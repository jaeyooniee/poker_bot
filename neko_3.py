"""
Neko_3: Neko_2 plus reading checks and playing position.

What is new over Neko_2:
- Checks are information. When an opponent checks while they could have
  bet, their strongest current hands become less likely: we drop the top
  share of their simulated hands, sized by how often they bet when given
  the chance and lowered by how often they check-raise (slowplay).
- Position. In position (acting last after the flop) we value-bet thinner,
  bet when checked to, c-bet more, and take free cards with draws against
  callers. Out of position we need a little more to call, probe after the
  opponent checked back, and check-raise strong hands against opponents
  who bet when checked to.
- More Monte Carlo samples when the opponent's range is narrowed, so a
  narrowed estimate rests on at least ~200 samples.

Run it with:  .venv/bin/python -m evals --model neko_3:Model
Submit with:  .venv/bin/python -m submit --team NAME --model neko_3:Model
"""

import math
import random

from card_utils import Card, evaluate, full_deck
from game_state import GameStateView
from player_action import DEALER, PlayerAction
from preflop_table import PREFLOP_EQUITY

PREFLOP_SAMPLES = 500
POSTFLOP_SAMPLES = 400
MIN_KEPT_SAMPLES = 200  # after narrowing, simulate until this many count
MAX_SAMPLES = 1500
MIN_HANDS_TO_CLASSIFY = 12
MAX_CHECK_DROP = 0.6  # never assume more than this share is ruled out

PRIORS = {
    "vpip": (0.60, 5),
    "pfr": (0.35, 5),
    "threebet": (0.15, 5),
    "fold_to_open": (0.35, 5),
    "post_aggr": (0.30, 8),
    "fold_to_bet": (0.40, 5),
    "lead": (0.35, 6),  # bets when nobody has bet yet this street
    "check_raise": (0.12, 6),  # raises after checking on the same street
}

ARCHETYPES = {
    "unknown": dict(open=0.80, open_bb=2.5, value=0.62, raise_=0.78,
                    size=0.66, margin=0.03, bluff=True, trap=0.0),
    "tag": dict(open=0.80, open_bb=2.5, value=0.62, raise_=0.78,
                size=0.66, margin=0.03, bluff=True, trap=0.0),
    "nit": dict(open=0.90, open_bb=2.5, value=0.65, raise_=0.82,
                size=0.6, margin=0.06, bluff=True, trap=0.0),
    "lag": dict(open=0.70, open_bb=2.5, value=0.60, raise_=0.78,
                size=0.66, margin=0.01, bluff=True, trap=0.0),
    "station": dict(open=0.55, open_bb=3.0, value=0.55, raise_=0.72,
                    size=0.9, margin=0.02, bluff=False, trap=0.0),
    "maniac": dict(open=0.55, open_bb=2.5, value=0.62, raise_=0.85,
                   size=0.66, margin=0.0, bluff=False, trap=0.6),
}

# Position adjustments after the flop.
IP_VALUE_SHIFT = -0.03  # in position: value-bet a little thinner
OOP_VALUE_SHIFT = 0.02
IP_MARGIN_SHIFT = -0.01  # in position: call a little lighter
OOP_MARGIN_SHIFT = 0.02

_RNG = random.Random(2026_11_07_3)


def hand_key(cards) -> str:
    """"AKs", "AKo" or "AA" for two hole cards."""
    a, b = sorted(cards, key=lambda c: c.value, reverse=True)
    if a.rank == b.rank:
        return a.rank + b.rank
    return a.rank + b.rank + ("s" if a.suit == b.suit else "o")


def _clamp(x, low, high):
    return min(high, max(low, x))


class Model:
    def __init__(self):
        self.hand = []
        self.player_id = None
        self.percentile = self._percentiles()
        self.stats = {}
        self._new_hand()

    # -- engine hooks --------------------------------------------------------

    def seed(self, hand: list[Card], player_id: int):
        self.hand = list(hand)
        self.player_id = player_id
        self._new_hand()

    def run(self, actions: list[PlayerAction],
            game_state: GameStateView) -> PlayerAction:
        self._observe(actions, game_state)
        if game_state.street == "preflop":
            return self._preflop(game_state)
        return self._postflop(game_state)

    def end_game(self, actions: list[PlayerAction],
                 game_state: GameStateView):
        self._observe(actions, game_state)
        teams = game_state.teams
        for p in self.h_seen:
            s = self._team(teams[p])
            s["hands"] += 1
            s["vpip"] += p in self.h_vpip
            s["pfr"] += p in self.h_pfr

    # -- preflop (as Neko_2) -------------------------------------------------

    def _preflop(self, gs):
        me = self.player_id
        key = hand_key(self.hand)
        pct = self.percentile[key]
        opp = self._main_opponent(gs)
        arch = self._arch(gs, opp)
        bb = gs.big_blind

        if self.raise_count == 0:
            if gs.call_amount > 0:  # first in from the small blind
                open_share = arch["open"]
                if (opp is not None
                        and self._team(gs.teams[opp])["hands"]
                        >= MIN_HANDS_TO_CLASSIFY
                        and self._stat(gs.teams[opp], "fold_to_open") >= 0.7):
                    open_share = 1.0
                if pct <= open_share:
                    size = arch["open_bb"]
                    if arch is ARCHETYPES["station"] and pct <= 0.15:
                        size = 3.5
                    return self._raise_to(gs, size * bb)
                return self._check_or_fold(gs)
            top = 0.35 if arch is ARCHETYPES["station"] else 0.25
            if pct <= top:
                size = 4 if arch is ARCHETYPES["station"] else 3
                return self._raise_to(gs, size * bb)
            return PlayerAction.call(me)

        raiser = self.last_raiser if self.last_raiser != me else opp
        width = self._range_width(gs, raiser) if raiser is not None else 1.0
        equity = self._equity(gs, {raiser: width} if raiser is not None
                              else {}, 1.0, 0.0, PREFLOP_SAMPLES)
        if equity >= 0.60 and gs.can_raise:
            if arch is ARCHETYPES["maniac"]:
                return self._raise_to(gs, gs.max_raise_to)
            return self._raise_to(gs, 3 * gs.current_bet)
        margin = self._call_margin(gs, arch, preflop=True, ip=False)
        if equity >= gs.pot_odds + margin:
            return PlayerAction.call(me)
        return self._check_or_fold(gs)

    # -- postflop ------------------------------------------------------------

    def _postflop(self, gs):
        me = self.player_id
        opp = self._main_opponent(gs)
        name = gs.teams[opp] if opp is not None else None
        arch = self._arch(gs, opp)
        ip = self._in_position(gs)
        widths = {p: self._range_width(gs, p)
                  for p in gs.players_in_hand if p != me}

        # Bets narrow their range to the strong end; checks cut the top.
        narrow, drop = 1.0, 0.0
        bettors = [p for p, n in self.h_post_raises.items()
                   if n and not gs.folded[p]]
        if bettors:
            p = bettors[0]
            aggr = self._stat(gs.teams[p], "post_aggr")
            narrow = max(0.15, aggr ** self.h_post_raises[p])
        elif opp is not None and self.h_checks.get(opp, 0):
            drop = self._check_drop(name, self.h_checks[opp])
        equity = self._equity(gs, widths, narrow, drop, POSTFLOP_SAMPLES)

        pot = gs.pot
        spr = gs.spr(me) if pot else 99.0
        fold_rate = self._stat(name, "fold_to_bet") if name else 0.4
        bluff_rate = 0.0
        if arch["bluff"]:
            bluff_rate = _clamp((fold_rate - 0.35) * 1.5, 0.0, 0.6)
        draw = self._has_draw(gs)
        opp_checked_now = opp in self.st_checked
        value = arch["value"] + (IP_VALUE_SHIFT if ip else OOP_VALUE_SHIFT)

        if gs.call_amount == 0:
            if spr < 1.5 and equity >= 0.55:
                return self._raise_to(gs, gs.max_raise_to)
            if spr < 3 and equity >= 0.75:
                return self._raise_to(gs, gs.max_raise_to)

            if equity >= value:
                first_to_act = not any(gs.acted)
                if first_to_act and not ip:
                    # Check-raise against players who bet when checked to.
                    lead = self._stat(name, "lead") if name else 0.35
                    trap = max(arch["trap"],
                               min(0.6, lead) if lead >= 0.45 else 0.0)
                    if _RNG.random() < trap:
                        return PlayerAction.call(me)  # check
                return self._bet(gs, arch["size"])

            if arch["bluff"]:
                # Continuation bet: more often in position.
                if self.pf_raiser == me and gs.street == "flop":
                    freq = 0.45 + (fold_rate - 0.35) * 1.5
                    freq *= 1.15 if ip else 0.8
                    if not self._board_dry(gs):
                        freq *= 0.7
                    if equity >= 0.45 or _RNG.random() < _clamp(freq, 0.2, 0.9):
                        return self._bet(gs, 0.4)
                # In position and they checked: take the pot.
                if ip and opp_checked_now:
                    freq = _clamp(0.35 + (fold_rate - 0.35) * 1.5, 0.15, 0.8)
                    if equity >= 0.45 or _RNG.random() < freq:
                        return self._bet(gs, 0.5)
                # Out of position after they checked back: probe.
                if (not ip and gs.street != "flop"
                        and self.h_checks.get(opp, 0) > 0):
                    freq = _clamp(0.3 + (fold_rate - 0.35) * 1.5, 0.1, 0.7)
                    if equity >= 0.45 or _RNG.random() < freq:
                        return self._bet(gs, 0.5)

            if draw and gs.street != "river":
                if ip and fold_rate < 0.35:
                    return PlayerAction.call(me)  # free card
                if arch["bluff"] and _RNG.random() < bluff_rate + 0.25:
                    return self._bet(gs, 0.6)
            if _RNG.random() < bluff_rate:
                return self._bet(gs, 0.5)
            return PlayerAction.call(me)  # check

        # Facing a bet.
        pot_odds = gs.pot_odds
        raise_at = arch["raise_"]
        if self.st_me_checked and not ip:
            raise_at = min(raise_at, 0.70)  # the planned check-raise
        if equity >= raise_at and gs.can_raise:
            return self._raise_to(gs, gs.current_bet + pot + gs.call_amount)
        if spr < 1.5 and equity >= 0.55 and gs.can_raise:
            return self._raise_to(gs, gs.max_raise_to)
        margin = self._call_margin(gs, arch, preflop=False, ip=ip)
        if equity >= pot_odds + margin:
            return PlayerAction.call(me)
        return self._check_or_fold(gs)

    def _call_margin(self, gs, arch, preflop, ip):
        """Equity over pot odds needed to call: none when nothing can follow
        (all-in or river); otherwise the archetype's margin, shifted by
        position."""
        me = self.player_id
        all_in = gs.call_amount >= gs.stacks[me]
        others_all_in = all(gs.stacks[p] == 0 for p in gs.players_in_hand
                            if p != me)
        if all_in or others_all_in or gs.street == "river":
            return 0.0
        margin = arch["margin"]
        if preflop:
            if gs.button != me:
                margin += 0.02  # we will act first after the flop
        else:
            margin += IP_MARGIN_SHIFT if ip else OOP_MARGIN_SHIFT
        return max(0.0, margin)

    def _check_drop(self, name, checks):
        """Share of the opponent's strongest hands ruled out by `checks`
        checks: each removes the share they would have bet (their lead
        rate) except what they slowplay (their check-raise rate)."""
        lead = self._stat(name, "lead")
        slow = self._stat(name, "check_raise")
        per_check = lead * (1.0 - slow)
        drop = 1.0 - (1.0 - per_check) ** checks
        return _clamp(drop, 0.0, MAX_CHECK_DROP)

    def _in_position(self, gs):
        """True when we act last after the flop among players still in."""
        n = gs.num_players
        order = [(gs.button + i) % n for i in range(1, n + 1)]
        live = [p for p in order if p in gs.players_in_hand]
        return bool(live) and live[-1] == self.player_id

    # -- equity --------------------------------------------------------------

    def _equity(self, gs, widths, narrow, drop, samples):
        """Share of the pot won against the opponents still in. Opponent p
        holds a hand from the top widths[p] of starting hands. Postflop,
        samples are ranked by the first opponent's current hand: narrow < 1
        keeps only that strongest share, drop > 0 removes that strongest
        share. Narrowed estimates are topped up to MIN_KEPT_SAMPLES."""
        me = self.player_id
        board = list(gs.board)
        opponents = [p for p in gs.players_in_hand if p != me]
        if not opponents:
            return 1.0
        if (not board and len(opponents) == 1
                and widths.get(opponents[0], 1.0) >= 0.999):
            return PREFLOP_EQUITY[hand_key(self.hand)]
        ranked = len(board) >= 3 and (narrow < 1.0 or drop > 0.0)
        if ranked:
            keep = narrow if narrow < 1.0 else 1.0 - drop
            samples = int(_clamp(MIN_KEPT_SAMPLES / max(keep, 0.01),
                                 samples, MAX_SAMPLES))
        dead = set(self.hand) | set(board)
        unseen = [c for c in full_deck() if c not in dead]
        missing = 5 - len(board)
        results = []
        for _ in range(samples):
            deck = unseen[:]
            _RNG.shuffle(deck)
            holes = [self._draw_hole(deck, widths.get(p, 1.0))
                     for p in opponents]
            full = board + deck[:missing]
            mine = evaluate(self.hand + full)
            theirs = [evaluate(h + full) for h in holes]
            best = max(theirs)
            if mine > best:
                share = 1.0
            elif mine == best:
                share = 1.0 / (1 + theirs.count(best))
            else:
                share = 0.0
            strength = evaluate(holes[0] + board) if ranked else 0
            results.append((strength, share))
        if ranked:
            results.sort(key=lambda r: r[0], reverse=True)
            if narrow < 1.0:
                results = results[:max(1, math.ceil(narrow * len(results)))]
            else:
                cut = min(len(results) - 1, int(drop * len(results)))
                results = results[cut:]
        return sum(share for _, share in results) / len(results)

    def _draw_hole(self, deck, width):
        for _ in range(40):
            if width >= 0.999 or len(deck) < 8:
                break
            hole = [deck[-1], deck[-2]]
            if self.percentile[hand_key(hole)] <= width:
                break
            for idx in (-1, -2):
                k = _RNG.randrange(len(deck) - 2)
                deck[idx], deck[k] = deck[k], deck[idx]
        return [deck.pop(), deck.pop()]

    @staticmethod
    def _percentiles():
        combos = {k: 6 if len(k) == 2 else 4 if k[2] == "s" else 12
                  for k in PREFLOP_EQUITY}
        result, seen = {}, 0
        for key in sorted(PREFLOP_EQUITY, key=PREFLOP_EQUITY.get,
                          reverse=True):
            seen += combos[key]
            result[key] = seen / 1326
        return result

    def _range_width(self, gs, p):
        if p is None:
            return 1.0
        name = gs.teams[p]
        if p == self.pf_raiser:
            if self.pf_raiser_count.get(p, 0) >= 2 or self.h_threebet.get(p):
                width = self._stat(name, "threebet")
            else:
                width = self._stat(name, "pfr")
        elif p in self.h_vpip:
            width = self._stat(name, "vpip")
        else:
            width = 1.0
        return _clamp(width, 0.05, 1.0)

    # -- board reading -------------------------------------------------------

    def _has_draw(self, gs):
        board = list(gs.board)
        if len(board) >= 5:
            return False
        cards = self.hand + board
        for suit in "CDHS":
            if (sum(c.suit == suit for c in cards) == 4
                    and any(c.suit == suit for c in self.hand)):
                return True
        values = {c.value for c in cards}
        if 14 in values:
            values.add(1)
        board_values = {c.value for c in board}
        outs = 0
        for v in range(2, 15):
            if v in values:
                continue
            trial = values | {v} | ({1} if v == 14 else set())
            for low in range(1, 11):
                window = set(range(low, low + 5))
                if window <= trial and not window <= (board_values | {v}):
                    outs += 1
                    break
        return outs >= 2

    def _board_dry(self, gs):
        board = list(gs.board)
        suits = [c.suit for c in board]
        if max(suits.count(s) for s in set(suits)) >= 2:
            return False
        values = sorted({c.value for c in board})
        return all(b - a > 2 for a, b in zip(values, values[1:]))

    # -- watching the hand ---------------------------------------------------

    def _new_hand(self):
        self.started = False
        self.board_count = 0
        self.bets = []
        self.cur = 0
        self.raise_count = 0
        self.last_raiser = None
        self.pf_raiser = None
        self.pf_raiser_count = {}
        self.h_seen = set()
        self.h_vpip = set()
        self.h_pfr = set()
        self.h_threebet = {}
        self.h_tb_chance = set()
        self.h_post_raises = {}
        # Checks: opponent -> streets they checked with the chance to bet,
        # since their last raise.
        self.h_checks = {}
        self.st_checked = set()  # opponents who checked this street
        self.st_me_checked = False

    def _start(self, gs):
        n = gs.num_players
        self.bets = [0] * n
        small = gs.button if n == 2 else (gs.button + 1) % n
        big = (small + 1) % n
        self.bets[small] = gs.small_blind
        self.bets[big] = gs.big_blind
        self.cur = gs.big_blind
        self.started = True

    def _observe(self, actions, gs):
        if not self.started:
            self._start(gs)
        me = self.player_id
        teams = gs.teams
        for a in actions:
            if a.kind == "reveal_card":
                if a.player_id == DEALER:
                    self.board_count += 1
                    if self.board_count in (3, 4, 5):
                        self.bets = [0] * len(self.bets)
                        self.cur = 0
                        self.raise_count = 0
                        self.last_raiser = None
                        self.st_checked = set()
                        self.st_me_checked = False
                continue
            if a.kind not in ("fold", "call", "raise"):
                continue
            p = a.player_id
            preflop = self.board_count < 3
            owed = self.cur - self.bets[p]
            if p == me:
                if not preflop and a.kind == "call" and owed == 0:
                    self.st_me_checked = True
            else:
                self._count(p, a.kind, owed, preflop, teams[p])
            if a.kind == "call":
                self.bets[p] = max(self.bets[p], self.cur)
            elif a.kind == "raise":
                self.bets[p] = a.amount
                if a.amount > self.cur:
                    self.cur = a.amount
                    self.raise_count += 1
                    self.last_raiser = p
                    if preflop:
                        self.pf_raiser = p
                        self.pf_raiser_count[p] = \
                            self.pf_raiser_count.get(p, 0) + 1

    def _count(self, p, kind, owed, preflop, name):
        s = self._team(name)
        if preflop:
            self.h_seen.add(p)
            if kind == "raise" or (kind == "call" and owed > 0):
                self.h_vpip.add(p)
            if kind == "raise":
                self.h_pfr.add(p)
            if self.raise_count >= 1 and p not in self.h_tb_chance:
                self.h_tb_chance.add(p)
                s["threebet_n"] += 1
                if kind == "raise":
                    s["threebet"] += 1
                    self.h_threebet[p] = True
            if (self.raise_count == 1 and owed > 0
                    and self.last_raiser == self.player_id):
                s["fold_to_open_n"] += 1
                s["fold_to_open"] += kind == "fold"
            return
        s["post_aggr_n"] += 1
        if kind == "raise":
            s["post_aggr"] += 1
            self.h_post_raises[p] = self.h_post_raises.get(p, 0) + 1
            self.h_checks[p] = 0  # a raise outweighs earlier checks
        if owed == 0:
            # Nobody has bet yet this street: they could bet or check.
            s["lead_n"] += 1
            s["lead"] += kind == "raise"
            if kind == "call" and p not in self.st_checked:
                self.st_checked.add(p)
                self.h_checks[p] = self.h_checks.get(p, 0) + 1
        else:
            s["fold_to_bet_n"] += 1
            s["fold_to_bet"] += kind == "fold"
            if p in self.st_checked:
                s["check_raise_n"] += 1
                s["check_raise"] += kind == "raise"

    # -- opponent model ------------------------------------------------------

    def _team(self, name):
        if name not in self.stats:
            self.stats[name] = {
                "hands": 0, "vpip": 0, "pfr": 0,
                "threebet": 0, "threebet_n": 0,
                "fold_to_open": 0, "fold_to_open_n": 0,
                "post_aggr": 0, "post_aggr_n": 0,
                "fold_to_bet": 0, "fold_to_bet_n": 0,
                "lead": 0, "lead_n": 0,
                "check_raise": 0, "check_raise_n": 0,
            }
        return self.stats[name]

    def _stat(self, name, stat):
        s = self._team(name)
        rate, weight = PRIORS[stat]
        if stat in ("vpip", "pfr"):
            hits, chances = s[stat], s["hands"]
        else:
            hits, chances = s[stat], s[stat + "_n"]
        return (hits + rate * weight) / (chances + weight)

    def _archetype(self, name):
        if name is None or self._team(name)["hands"] < MIN_HANDS_TO_CLASSIFY:
            return "unknown"
        vpip = self._stat(name, "vpip")
        pfr = self._stat(name, "pfr")
        aggr = self._stat(name, "post_aggr")
        fold = self._stat(name, "fold_to_bet")
        if pfr >= 0.65 or aggr >= 0.55:
            return "maniac"
        if vpip >= 0.65 and pfr < 0.2 and fold < 0.3:
            return "station"
        if vpip < 0.35:
            return "nit"
        if pfr >= 0.4:
            return "lag"
        return "tag"

    def _arch(self, gs, opp):
        if opp is None:
            return ARCHETYPES["unknown"]
        return ARCHETYPES[self._archetype(gs.teams[opp])]

    def _main_opponent(self, gs):
        if self.last_raiser is not None and self.last_raiser != self.player_id:
            return self.last_raiser
        others = [p for p in gs.players_in_hand if p != self.player_id]
        return others[0] if others else None

    # -- moves ---------------------------------------------------------------

    def _bet(self, gs, frac):
        return self._raise_to(gs, gs.current_bet + frac * gs.pot)

    def _raise_to(self, gs, total):
        if not gs.can_raise:
            return PlayerAction.call(self.player_id)
        low, high = gs.raise_range
        total = int(round(total))
        if total >= 0.6 * high:
            total = high
        return PlayerAction.raise_to(self.player_id,
                                     min(high, max(low, total)))

    def _check_or_fold(self, gs):
        if gs.call_amount == 0:
            return PlayerAction.call(self.player_id)
        return PlayerAction.fold(self.player_id)
