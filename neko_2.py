"""
Neko_2: heads-up bot built on opponent archetypes.

Based on the MIT Pokerbots 2024 heads-up approach: track each opponent's
VPIP, PFR, 3-bet, postflop aggression and fold rates (smoothed toward a
prior), classify them (nit, TAG, LAG, calling station, maniac), and adjust
opening range, bet size, value threshold, bluffing and call margin to that
type. Equity comes from the preflop table or Monte Carlo against the
opponent's estimated range, narrowed for each postflop bet or raise.

Run it with:  .venv/bin/python -m evals --model neko_2:Model
Submit with:  .venv/bin/python -m submit --team NAME --model neko_2:Model
"""

import math
import random

from card_utils import Card, evaluate, full_deck
from game_state import GameStateView
from player_action import DEALER, PlayerAction
from preflop_table import PREFLOP_EQUITY

PREFLOP_SAMPLES = 500
POSTFLOP_SAMPLES = 500
MIN_HANDS_TO_CLASSIFY = 12

# Smoothing priors: (rate, weight). A stat is (hits + rate*weight) /
# (chances + weight), so early on it stays near the prior.
PRIORS = {
    "vpip": (0.60, 5),  # voluntarily put chips in preflop
    "pfr": (0.35, 5),  # raised preflop
    "threebet": (0.15, 5),  # re-raised when facing a preflop raise
    "fold_to_open": (0.35, 5),  # folded to our preflop raise
    "post_aggr": (0.30, 8),  # postflop actions that are bets/raises
    "fold_to_bet": (0.40, 5),  # folded when facing a postflop bet
}

# How we play against each type.
#   open:  share of hands the small blind opens
#   open_bb: open size in big blinds
#   value: postflop equity needed to bet for value
#   raise: equity needed to raise a bet
#   size:  value bet as a share of the pot
#   margin: extra equity over pot odds needed to call (not all-in/river)
#   bluff: whether to bluff / semi-bluff / c-bet light at all
#   trap:  chance to check a strong hand first to let them bet
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

_RNG = random.Random(2026_11_07)


def hand_key(cards) -> str:
    """"AKs", "AKo" or "AA" for two hole cards."""
    a, b = sorted(cards, key=lambda c: c.value, reverse=True)
    if a.rank == b.rank:
        return a.rank + b.rank
    return a.rank + b.rank + ("s" if a.suit == b.suit else "o")


class Model:
    def __init__(self):
        self.hand = []
        self.player_id = None
        self.percentile = self._percentiles()
        self.stats = {}  # team name -> counters, kept for the whole match
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

    # -- preflop -------------------------------------------------------------

    def _preflop(self, gs):
        me = self.player_id
        key = hand_key(self.hand)
        pct = self.percentile[key]
        opp = self._main_opponent(gs)
        arch = ARCHETYPES[self._archetype(gs.teams[opp])] if opp is not None \
            else ARCHETYPES["unknown"]
        bb = gs.big_blind

        if self.raise_count == 0:
            if gs.call_amount > 0:  # first in from the small blind
                open_share = arch["open"]
                # They fold to most opens: any two cards make money.
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
            # Big blind after a limp: raise the top, otherwise check free.
            top = 0.35 if arch is ARCHETYPES["station"] else 0.25
            if pct <= top:
                size = 4 if arch is ARCHETYPES["station"] else 3
                return self._raise_to(gs, size * bb)
            return PlayerAction.call(me)

        # Facing a raise: equity against the raiser's range.
        raiser = self.last_raiser if self.last_raiser != me else opp
        width = self._range_width(gs, raiser) if raiser is not None else 1.0
        equity = self._equity(gs, {raiser: width} if raiser is not None
                              else {}, 1.0, PREFLOP_SAMPLES)
        if equity >= 0.60 and gs.can_raise:
            if arch is ARCHETYPES["maniac"]:
                return self._raise_to(gs, gs.max_raise_to)
            return self._raise_to(gs, 3 * gs.current_bet)
        margin = self._call_margin(gs, arch, preflop=True)
        if equity >= gs.pot_odds + margin:
            return PlayerAction.call(me)
        return self._check_or_fold(gs)

    # -- postflop ------------------------------------------------------------

    def _postflop(self, gs):
        me = self.player_id
        opp = self._main_opponent(gs)
        name = gs.teams[opp] if opp is not None else None
        arch = ARCHETYPES[self._archetype(name)] if name else \
            ARCHETYPES["unknown"]
        widths = {p: self._range_width(gs, p)
                  for p in gs.players_in_hand if p != me}
        narrow = 1.0
        bettors = [p for p, n in self.h_post_raises.items()
                   if n and not gs.folded[p]]
        if bettors:
            p = bettors[0]
            aggr = self._stat(gs.teams[p], "post_aggr")
            narrow = max(0.15, aggr ** self.h_post_raises[p])
        equity = self._equity(gs, widths, narrow, POSTFLOP_SAMPLES)

        pot = gs.pot
        spr = gs.spr(me) if pot else 99.0
        fold_rate = self._stat(name, "fold_to_bet") if name else 0.4
        bluff_rate = 0.0
        if arch["bluff"]:
            bluff_rate = min(0.6, max(0.0, (fold_rate - 0.35) * 1.5))
        draw = self._has_draw(gs)

        if gs.call_amount == 0:
            # Low stack-to-pot: commit with a good hand.
            if spr < 1.5 and equity >= 0.55:
                return self._raise_to(gs, gs.max_raise_to)
            if spr < 3 and equity >= 0.75:
                return self._raise_to(gs, gs.max_raise_to)
            if equity >= arch["value"]:
                first_to_act = not any(gs.acted)
                if first_to_act and _RNG.random() < arch["trap"]:
                    return PlayerAction.call(me)  # check, let them bet
                return self._bet(gs, arch["size"])
            # Continuation bet: we raised preflop, first bet on the flop.
            if (self.pf_raiser == me and gs.street == "flop"
                    and arch["bluff"]):
                freq = min(0.9, max(0.25, 0.45 + (fold_rate - 0.35) * 1.5))
                if not self._board_dry(gs):
                    freq *= 0.7
                if equity >= 0.45 or _RNG.random() < freq:
                    return self._bet(gs, 0.4)
            if draw and gs.street != "river" and arch["bluff"] \
                    and _RNG.random() < bluff_rate + 0.25:
                return self._bet(gs, 0.6)
            if _RNG.random() < bluff_rate:
                return self._bet(gs, 0.5)
            return PlayerAction.call(me)  # check

        pot_odds = gs.pot_odds
        if equity >= arch["raise_"] and gs.can_raise:
            return self._raise_to(gs, gs.current_bet + pot + gs.call_amount)
        if spr < 1.5 and equity >= 0.55 and gs.can_raise:
            return self._raise_to(gs, gs.max_raise_to)
        margin = self._call_margin(gs, arch, preflop=False)
        if equity >= pot_odds + margin:
            return PlayerAction.call(me)
        return self._check_or_fold(gs)

    def _call_margin(self, gs, arch, preflop):
        """Equity over pot odds needed to call. None when nothing can
        follow the call (all-in or river); otherwise the archetype's
        margin, plus a little out of position preflop."""
        me = self.player_id
        all_in = gs.call_amount >= gs.stacks[me]
        others_all_in = all(gs.stacks[p] == 0 for p in gs.players_in_hand
                            if p != me)
        if all_in or others_all_in or gs.street == "river":
            return 0.0
        margin = arch["margin"]
        if preflop and gs.button != me:
            margin += 0.02  # we will act first after the flop
        return margin

    # -- equity --------------------------------------------------------------

    def _equity(self, gs, widths, narrow, samples):
        """Share of the pot won against the opponents still in. Opponent p
        holds a hand from the top widths[p] of starting hands; with
        narrow < 1 (postflop) only that share of samples, where the first
        opponent's current hand is strongest, count."""
        me = self.player_id
        board = list(gs.board)
        opponents = [p for p in gs.players_in_hand if p != me]
        if not opponents:
            return 1.0
        # Exact and free: one opponent from any hand, preflop.
        if (not board and len(opponents) == 1
                and widths.get(opponents[0], 1.0) >= 0.999):
            return PREFLOP_EQUITY[hand_key(self.hand)]
        dead = set(self.hand) | set(board)
        unseen = [c for c in full_deck() if c not in dead]
        missing = 5 - len(board)
        use_narrow = narrow < 1.0 and len(board) >= 3
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
            strength = evaluate(holes[0] + board) if use_narrow else 0
            results.append((strength, share))
        if use_narrow:
            results.sort(key=lambda r: r[0], reverse=True)
            results = results[:max(1, math.ceil(narrow * len(results)))]
        return sum(share for _, share in results) / len(results)

    def _draw_hole(self, deck, width):
        """Two cards off the end of deck for an opponent, from the top
        `width` of starting hands when possible (rejection sampling)."""
        for _ in range(40):
            if width >= 0.999 or len(deck) < 8:
                break
            hole = [deck[-1], deck[-2]]
            if self.percentile[hand_key(hole)] <= width:
                break
            # Swap these two into random places and try the new end.
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
        """Share of starting hands player p is likely to hold, from how
        they entered the pot this hand."""
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
            width = 1.0  # checked the big blind: anything
        return min(1.0, max(0.05, width))

    # -- board reading -------------------------------------------------------

    def _has_draw(self, gs):
        """A flush draw or an open-ended / double gutshot straight draw
        that uses a hole card."""
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
        self.raise_count = 0  # raises this street
        self.last_raiser = None  # this street
        self.pf_raiser = None  # last preflop raiser, us included
        self.pf_raiser_count = {}
        self.h_seen = set()  # opponents who acted preflop
        self.h_vpip = set()
        self.h_pfr = set()
        self.h_threebet = {}
        self.h_tb_chance = set()
        self.h_post_raises = {}

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
                continue
            if a.kind not in ("fold", "call", "raise"):
                continue
            p = a.player_id
            preflop = self.board_count < 3
            owed = self.cur - self.bets[p]
            if p != me:
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
        else:
            s["post_aggr_n"] += 1
            if kind == "raise":
                s["post_aggr"] += 1
                self.h_post_raises[p] = self.h_post_raises.get(p, 0) + 1
            if owed > 0:
                s["fold_to_bet_n"] += 1
                s["fold_to_bet"] += kind == "fold"

    # -- opponent model ------------------------------------------------------

    def _team(self, name):
        if name not in self.stats:
            self.stats[name] = {
                "hands": 0, "vpip": 0, "pfr": 0,
                "threebet": 0, "threebet_n": 0,
                "fold_to_open": 0, "fold_to_open_n": 0,
                "post_aggr": 0, "post_aggr_n": 0,
                "fold_to_bet": 0, "fold_to_bet_n": 0,
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

    def _main_opponent(self, gs):
        """The opponent we mostly play against: the last raiser if it is
        not us, else the first other player still in."""
        if self.last_raiser is not None and self.last_raiser != self.player_id:
            return self.last_raiser
        others = [p for p in gs.players_in_hand if p != self.player_id]
        return others[0] if others else None

    # -- moves ---------------------------------------------------------------

    def _bet(self, gs, frac):
        """A bet of frac of the pot (when nothing is owed)."""
        return self._raise_to(gs, gs.current_bet + frac * gs.pot)

    def _raise_to(self, gs, total):
        """A raise to about total, kept legal; all-in when it would use
        most of the stack anyway; a call when raising is not allowed."""
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
