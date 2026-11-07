"""
Neko_7 (also in neko_7.py): Neko_6 with Neko_1's preflop.

What is new over Neko_6:
- Preflop is Neko_1's (the same as preflop_model.py): the small blind
  opens hands with table equity 0.42 or more (wider against players who
  fold to our opens), 2.5 big blinds (3 with a strong hand against players
  who rarely fold); the big blind raises a limp with equity 0.58 or more;
  facing a raise, equity against the raiser's preflop raise rate decides:
  re-raise from 0.62, call above the pot odds plus 0.02. No archetypes
  preflop, so a wide-opening button is no longer read as a maniac there.
  Postflop, opponent stats and everything else are Neko_6's.

Neko_6 over Neko_5 (kept):
- Varied bet sizes around one base size per opponent. Every bet is drawn
  at random: value bets between 75% and 125% of the base, bluffs (c-bet,
  stab, probe, semi-bluff, bluff) bigger, between 100% and 160%. The two
  ranges overlap, so a size alone does not tell strong from weak, and no
  size repeats exactly (Neko_5 always bet 0.4 to c-bet, 0.5 to bluff, 0.6
  to semi-bluff).
- Inducing. Against an opponent who calls fewer than INDUCE_BELOW of our
  value bets, that size is small (INDUCE_SIZE of the pot): strong hands
  get called by weaker ones, and bluffs cost less. Against one who calls
  most of them it is bigger. Calling stations still get big value bets
  (we never bluff them, so there is no size to hide).

Neko_5 over Neko_4 (kept):
- Decision success rates, per opponent team. Each kind of bluff (c-bet,
  stab, probe, semi-bluff, bluff) records whether the opponent folded to
  it; its frequency is scaled by how far that fold rate sits above the
  break-even rate for its size (never below 0, at most 1.5x). Value bets
  record whether they got called: rarely called -> smaller, usually
  called -> bigger. Calls with nothing to follow (river or all-in) record
  whether they won at showdown against the pot odds paid: losing calls
  raise the call margin, winning ones lower it a little.
- Safety switch. Every GUARD_WINDOW hands against a team we check our
  result with all-in luck removed (an all-in before the river counts at
  its equity, not its runout). Losing more than GUARD_LIMIT big blinds a
  hand switches between exploiting the read archetype and the default
  strategy; otherwise we keep what we are doing.

Neko_4 over Neko_3 (kept):
- Showdown learning. When an opponent shows their cards, every postflop
  bet or raise they made that hand is checked against the board as it was
  then: a hand weaker than a random hand on that board counts as a bluff.
  Each team's bluff rate is kept (smoothed toward a prior).
- Their bets are read as value plus bluffs. Facing a bet, their range is
  the strongest share they bet for value mixed with their weakest hands,
  weighted by their bluff rate, instead of only the strongest share.
- The share they bet comes from the situation: how often they bet when
  nobody has bet yet (lead), or how often they raise a bet, not their
  overall postflop aggression (which counted calls too and made Neko_3
  fold too much against frequent bettors).

Neko_3 over Neko_2 (kept):
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

Run it with:  python -m evals 
Submit with:  python -m submit --team NAME 
"""

import math
import random
from itertools import combinations

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
    "raise_vs_bet": (0.12, 6),  # raises when facing a postflop bet
    "bluff": (0.25, 4),  # shown bets that were weaker than a random hand
}

BLUFF_SHARE = 0.30  # their bluffs come from the weakest 30% of their range
MIN_VALUE_SHARE = 0.06  # never narrow their value range below this
SHOWDOWN_SAMPLES = 200  # random hands to rate a shown hand on a board

# Decision feedback (smoothing priors: rate, weight).
BLUFF_LABELS = ("cbet", "stab", "probe", "semibluff", "bluff")
BLUFF_PRIOR = (0.5, 6)  # share of our bluffs they fold to
VALUE_CALL_PRIOR = (0.4, 6)  # share of our value bets they call
CALL_PRIOR_WEIGHT = 5  # final calls: surplus over pot odds, prior 0

# One bet size for value and bluffs (share of the pot), per opponent.
BET_SIZE = 0.5
INDUCE_SIZE = 0.33  # they fold to our value bets: bet small, get calls
INDUCE_BELOW = 0.45  # ...when they call fewer of our value bets than this
BIG_SIZE = 0.66  # they call our value bets: bet bigger
BIG_ABOVE = 0.6
# Each bet is drawn at random around that size, so no two look alike.
# Bluffs are drawn bigger, overlapping the value range so a size alone
# does not tell them apart.
VALUE_SPREAD = (0.75, 1.25)  # value bets: 75%-125% of the size
BLUFF_SPREAD = (1.0, 1.6)  # bluffs: 100%-160% of the size

# Safety switch.
GUARD_WINDOW = 50  # hands per check
GUARD_LIMIT = -1.0  # big blinds a hand (-100 bb/100), luck removed
PREFLOP_ALLIN_SAMPLES = 600

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
        if self.start_stacks is None:
            # Chips each player had when the hand began.
            self.start_stacks = [game_state.stacks[p] + self.contrib[p]
                                 for p in range(game_state.num_players)]
        self.pending = None
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
        self._learn_showdown(actions, game_state)
        self._learn_calls(actions, game_state)
        self._guard_update(actions, game_state)

    # -- decision feedback ---------------------------------------------------

    def _shown(self, actions):
        """Opponents' hole cards shown at the showdown: id -> [Card, Card]."""
        shown = {}
        for a in actions:
            if (a.kind == "reveal_card" and a.player_id != DEALER
                    and a.player_id != self.player_id):
                shown.setdefault(a.player_id, []).append(a.card)
        return {p: h for p, h in shown.items() if len(h) == 2}

    def _learn_calls(self, actions, gs):
        """Final calls (river or all-in): did they win at showdown, against
        the pot odds we paid?"""
        if not self.h_calls or gs.folded[self.player_id]:
            return
        shown = self._shown(actions)
        if not shown:
            return
        board = list(gs.board)
        mine = evaluate(self.hand + board)
        best = max(evaluate(h + board) for h in shown.values())
        share = 1.0 if mine > best else 0.5 if mine == best else 0.0
        for name, pot_odds in self.h_calls:
            s = self._team(name)
            s["calls_n"] += 1
            s["calls_surplus"] += share - pot_odds

    def _success(self, name, label, prior):
        rate, weight = prior
        n, hits = self._team(name)["dec"].get(label, [0, 0])
        return (hits + rate * weight) / (n + weight)

    def _bluff_mult(self, name, label, size):
        """How much of a bluff's base frequency to keep: 1 at the prior
        fold rate, 0 at the break-even fold rate for this size, at most
        1.5."""
        if name is None:
            return 1.0
        breakeven = size / (1.0 + size)
        if label == "semibluff":
            breakeven *= 0.6  # it can still win when called
        folds = self._success(name, label, BLUFF_PRIOR)
        return _clamp((folds - breakeven) / (BLUFF_PRIOR[0] - breakeven),
                      0.0, 1.5)

    @staticmethod
    def _vary(size, spread):
        """A size drawn at random between spread[0] and spread[1] times
        size: the same situation never shows the same number twice."""
        return size * _RNG.uniform(*spread)

    def _bet_size(self, name):
        """The one bet size (share of the pot) we use against this team
        for value and bluffs alike: small to induce calls when they fold
        to our value bets, bigger when they call them."""
        if name is None:
            return BET_SIZE
        called = self._success(name, "value", VALUE_CALL_PRIOR)
        if called < INDUCE_BELOW:
            return INDUCE_SIZE
        if called > BIG_ABOVE:
            return BIG_SIZE
        return BET_SIZE

    def _value_size(self, name, size):
        """Value bet size from how often our value bets get called."""
        if name is None:
            return size
        called = self._success(name, "value", VALUE_CALL_PRIOR)
        if called < 0.3:
            size *= 0.7
        elif called > 0.6:
            size *= 1.2
        return _clamp(size, 0.33, 1.0)

    def _call_adjust(self, name):
        """Extra call margin from how our final calls did: a deficit
        against pot odds raises it, a surplus lowers it a little."""
        if name is None:
            return 0.0
        s = self._team(name)
        surplus = s["calls_surplus"] / (s["calls_n"] + CALL_PRIOR_WEIGHT)
        return _clamp(-0.5 * surplus, -0.03, 0.08)

    def _tracked_bet(self, gs, frac, label, name):
        """A bet whose result (fold or not) is recorded under label."""
        if name is not None:
            self.pending = (label, name)
        return self._bet(gs, frac)

    # -- safety switch -------------------------------------------------------

    def _guard_update(self, actions, gs):
        opp = self._main_opponent(gs)
        if opp is None:
            others = [p for p in range(gs.num_players) if p != self.player_id]
            opp = others[0] if others else None
        if opp is None:
            return
        g = self._team(gs.teams[opp])["guard"]
        g["n"] += 1
        g["sum"] += self._luckless_net(actions, gs) / gs.big_blind
        if g["n"] >= GUARD_WINDOW:
            if g["sum"] / g["n"] < GUARD_LIMIT:
                g["safe"] = not g["safe"]
                g["switches"] += 1
            g["n"], g["sum"] = 0, 0.0

    def _luckless_net(self, actions, gs):
        """Chips won this hand, with a heads-up all-in before the river
        counted at its equity instead of its runout."""
        me = self.player_id
        def put_in(p):
            # A call all-in for less puts in only what was left.
            chips = self.contrib[p]
            if self.start_stacks is not None:
                chips = min(chips, self.start_stacks[p])
            return chips

        won = sum(a.amount for a in actions
                  if a.kind == "win" and a.player_id == me)
        net = won - put_in(me)
        shown = self._shown(actions)
        if (len(shown) != 1 or self.last_action_board >= 5
                or gs.folded[me] or len(gs.players_in_hand) != 2):
            return net
        (opp, hole), = shown.items()
        board = list(gs.board)[:self.last_action_board]
        equity = self._exact_equity(hole, board)
        matched = min(put_in(me), put_in(opp))
        return matched * (2.0 * equity - 1.0)

    def _exact_equity(self, hole, board):
        """Our share of the pot against `hole` from `board` on: every
        runout from the flop on, Monte Carlo preflop."""
        dead = set(self.hand) | set(hole) | set(board)
        unseen = [c for c in full_deck() if c not in dead]
        missing = 5 - len(board)
        if len(board) >= 3:
            runouts = list(combinations(unseen, missing))
        else:
            runouts = [_RNG.sample(unseen, missing)
                       for _ in range(PREFLOP_ALLIN_SAMPLES)]
        score = 0.0
        for extra in runouts:
            full = board + list(extra)
            a = evaluate(self.hand + full)
            b = evaluate(list(hole) + full)
            score += 1.0 if a > b else 0.5 if a == b else 0.0
        return score / len(runouts)

    def _learn_showdown(self, actions, gs):
        """For every opponent who showed cards, rate each postflop bet or
        raise they made this hand: a bluff when their hand was weaker than
        a random hand on the board at that moment."""
        shown = {}
        for a in actions:
            if (a.kind == "reveal_card" and a.player_id != DEALER
                    and a.player_id != self.player_id):
                shown.setdefault(a.player_id, []).append(a.card)
        board = list(gs.board)
        for p, kind, board_size in self.h_aggr:
            hole = shown.get(p)
            if not hole or len(hole) != 2 or board_size > len(board):
                continue
            strength = self._strength(hole, board[:board_size])
            s = self._team(gs.teams[p])
            s["bluff_n"] += 1
            s["bluff"] += strength < 0.5

    def _strength(self, hole, board):
        """Share of random hands that `hole` beats on `board` now (ties
        half), our own cards excluded from theirs."""
        dead = set(hole) | set(board) | set(self.hand)
        unseen = [c for c in full_deck() if c not in dead]
        mine = evaluate(list(hole) + board)
        score = 0.0
        for _ in range(SHOWDOWN_SAMPLES):
            other = evaluate(_RNG.sample(unseen, 2) + board)
            score += 1.0 if mine > other else 0.5 if mine == other else 0.0
        return score / SHOWDOWN_SAMPLES

    # -- preflop (as Neko_1 / preflop_model.py) -------------------------------

    def _preflop(self, gs):
        """Neko_1's preflop, on Neko_7's opponent stats: open by table
        equity (wider against players who fold to our opens), raise a limp
        with strong hands, and against a raise compare equity versus the
        raiser's range (their preflop raise rate) with the pot odds."""
        me = self.player_id
        key = hand_key(self.hand)
        base = PREFLOP_EQUITY[key]
        bb = gs.big_blind
        opp = self._main_opponent(gs)
        name = gs.teams[opp] if opp is not None else None

        if self.raise_count == 0:
            fold_rate = self._stat(name, "fold_to_open") if name else 0.35
            if gs.call_amount > 0:  # first in from the small blind
                # Open wider against players who fold to raises.
                threshold = 0.42 - 0.15 * max(0.0, fold_rate - 0.4)
                if base >= threshold:
                    size = 3 * bb if base >= 0.6 and fold_rate < 0.3 \
                        else 2.5 * bb
                    return self._raise_to(gs, size)
                return self._check_or_fold(gs)
            # Big blind after a limp: raise strong hands, otherwise check.
            if base >= 0.58:
                return self._raise_to(gs, 3 * gs.current_bet)
            return PlayerAction.call(me)

        # Facing a raise: equity against the raiser's estimated range.
        raiser = self.last_raiser if self.last_raiser != me else opp
        name = gs.teams[raiser] if raiser is not None else None
        width = self._stat(name, "pfr") if name else 1.0
        width = min(1.0, max(0.08, width))
        equity = self._equity(gs, {raiser: width} if raiser is not None
                              else {}, 1.0, 0.0, PREFLOP_SAMPLES)
        if equity >= 0.62 and gs.can_raise:
            return self._raise_to(gs, 3 * gs.current_bet)
        if equity > gs.pot_odds + 0.02:
            self._note_final_call(gs, name)
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

        # Bets: their value hands (strong end) mixed with their bluffs
        # (weak end) by their bluff rate. Checks cut the top.
        narrow, drop, bluff = 1.0, 0.0, 0.0
        bettors = [p for p, n in self.h_post_raises.items()
                   if n and not gs.folded[p]]
        if bettors:
            p = bettors[0]
            narrow, bluff = self._bet_range(gs.teams[p], p)
        elif opp is not None and self.h_checks.get(opp, 0):
            drop = self._check_drop(name, self.h_checks[opp])
        equity = self._equity(gs, widths, narrow, drop, POSTFLOP_SAMPLES,
                              bluff)

        pot = gs.pot
        spr = gs.spr(me) if pot else 99.0
        fold_rate = self._stat(name, "fold_to_bet") if name else 0.4
        bluff_rate = 0.0
        if arch["bluff"]:
            bluff_rate = _clamp((fold_rate - 0.35) * 1.5, 0.0, 0.6)
        draw = self._has_draw(gs)
        opp_checked_now = opp in self.st_checked
        value = arch["value"] + (IP_VALUE_SHIFT if ip else OOP_VALUE_SHIFT)
        S = self._bet_size(name)  # one size for value and bluffs alike

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
                # Drawn around S, small (inducing) against players who fold
                # to our bets; calling stations get the big value size.
                if arch is ARCHETYPES["station"]:
                    size = arch["size"]
                else:
                    size = self._vary(S, VALUE_SPREAD)
                return self._tracked_bet(gs, size, "value", name)

            # Bluffs: drawn bigger than value bets, but overlapping them.
            B = S * sum(BLUFF_SPREAD) / 2  # their average size
            if arch["bluff"]:
                # Continuation bet: more often in position.
                if self.pf_raiser == me and gs.street == "flop":
                    if equity >= 0.45:
                        return self._bet(gs, self._vary(S, VALUE_SPREAD))
                    freq = 0.45 + (fold_rate - 0.35) * 1.5
                    freq *= 1.15 if ip else 0.8
                    if not self._board_dry(gs):
                        freq *= 0.7
                    freq = _clamp(freq, 0.2, 0.9)
                    freq *= self._bluff_mult(name, "cbet", B)
                    if _RNG.random() < freq:
                        return self._tracked_bet(
                            gs, self._vary(S, BLUFF_SPREAD), "cbet", name)
                # In position and they checked: take the pot.
                if ip and opp_checked_now:
                    if equity >= 0.45:
                        return self._bet(gs, self._vary(S, VALUE_SPREAD))
                    freq = _clamp(0.35 + (fold_rate - 0.35) * 1.5, 0.15, 0.8)
                    freq *= self._bluff_mult(name, "stab", B)
                    if _RNG.random() < freq:
                        return self._tracked_bet(
                            gs, self._vary(S, BLUFF_SPREAD), "stab", name)
                # Out of position after they checked back: probe.
                if (not ip and gs.street != "flop"
                        and self.h_checks.get(opp, 0) > 0):
                    if equity >= 0.45:
                        return self._bet(gs, self._vary(S, VALUE_SPREAD))
                    freq = _clamp(0.3 + (fold_rate - 0.35) * 1.5, 0.1, 0.7)
                    freq *= self._bluff_mult(name, "probe", B)
                    if _RNG.random() < freq:
                        return self._tracked_bet(
                            gs, self._vary(S, BLUFF_SPREAD), "probe", name)

            if draw and gs.street != "river":
                if ip and fold_rate < 0.35:
                    return PlayerAction.call(me)  # free card
                if arch["bluff"]:
                    freq = (bluff_rate + 0.25) * \
                        self._bluff_mult(name, "semibluff", B)
                    if _RNG.random() < freq:
                        return self._tracked_bet(
                            gs, self._vary(S, BLUFF_SPREAD), "semibluff", name)
            if _RNG.random() < bluff_rate * self._bluff_mult(name, "bluff", B):
                return self._tracked_bet(
                    gs, self._vary(S, BLUFF_SPREAD), "bluff", name)
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
        margin = self._call_margin(gs, arch, preflop=False, ip=ip, name=name)
        if equity >= pot_odds + margin:
            self._note_final_call(gs, name)
            return PlayerAction.call(me)
        return self._check_or_fold(gs)

    def _is_final_call(self, gs):
        """True when nothing can follow a call: river, or someone all-in."""
        me = self.player_id
        all_in = gs.call_amount >= gs.stacks[me]
        others_all_in = all(gs.stacks[p] == 0 for p in gs.players_in_hand
                            if p != me)
        return all_in or others_all_in or gs.street == "river"

    def _note_final_call(self, gs, name):
        if name is not None and gs.call_amount > 0 and self._is_final_call(gs):
            self.h_calls.append((name, gs.pot_odds))

    def _call_margin(self, gs, arch, preflop, ip, name=None):
        """Equity over pot odds needed to call: none when nothing can follow
        (all-in or river); otherwise the archetype's margin, shifted by
        position. Plus what our final calls against this team taught us."""
        me = self.player_id
        adjust = self._call_adjust(name)
        if self._is_final_call(gs):
            return max(0.0, adjust)
        margin = arch["margin"] + adjust
        if preflop:
            if gs.button != me:
                margin += 0.02  # we will act first after the flop
        else:
            margin += IP_MARGIN_SHIFT if ip else OOP_MARGIN_SHIFT
        return max(0.0, margin)

    def _bet_range(self, name, p):
        """(value share, bluff weight) for player p's bets this hand. Each
        bet keeps the share of their range they bet for value in that
        spot: how often they bet there times the share that is not a
        bluff. Bluffs are mixed back in by weight."""
        bluff = self._stat(name, "bluff")
        share = 1.0
        for who, kind, _ in self.h_aggr:
            if who != p:
                continue
            freq = self._stat(name, "lead" if kind == "lead"
                              else "raise_vs_bet")
            share *= freq * (1.0 - bluff)
        return max(MIN_VALUE_SHARE, min(1.0, share)), bluff

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

    def _equity(self, gs, widths, narrow, drop, samples, bluff=0.0):
        """Share of the pot won against the opponents still in. Opponent p
        holds a hand from the top widths[p] of starting hands. Postflop,
        samples are ranked by the first opponent's current hand: narrow < 1
        keeps that strongest share (their value bets), mixed with weight
        `bluff` with their weakest BLUFF_SHARE (their bluffs); drop > 0
        removes the strongest share. Narrowed estimates are topped up to
        MIN_KEPT_SAMPLES."""
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
                value = results[:max(1, math.ceil(narrow * len(results)))]
                value_eq = sum(s for _, s in value) / len(value)
                if bluff <= 0.0:
                    return value_eq
                weak = results[-max(1, int(BLUFF_SHARE * len(results))):]
                bluff_eq = sum(s for _, s in weak) / len(weak)
                return (1.0 - bluff) * value_eq + bluff * bluff_eq
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
        # Opponents' postflop bets/raises: (player, "lead" or "raise",
        # board size then), rated at showdown.
        self.h_aggr = []
        # Decision feedback and the safety switch.
        self.pending = None  # (label, team) of the bet we just made
        self.armed = None  # that bet, seen in the actions: await a reply
        self.h_calls = []  # (team, pot odds) of our final calls
        self.contrib = []  # chips each player put in this hand
        self.start_stacks = None
        self.last_action_board = 0  # board size at the last betting move

    def _start(self, gs):
        n = gs.num_players
        self.bets = [0] * n
        small = gs.button if n == 2 else (gs.button + 1) % n
        big = (small + 1) % n
        self.bets[small] = gs.small_blind
        self.bets[big] = gs.big_blind
        self.contrib = list(self.bets)
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
            self.last_action_board = self.board_count
            if p == me:
                if not preflop and a.kind == "call" and owed == 0:
                    self.st_me_checked = True
                # Our tracked bet is now on the table: await their reply.
                self.armed = self.pending if a.kind == "raise" else None
                self.pending = None
            else:
                self._count(p, a.kind, owed, preflop, teams[p])
                if self.armed is not None:
                    label, name = self.armed
                    d = self._team(name)["dec"].setdefault(label, [0, 0])
                    d[0] += 1
                    if label == "value":
                        d[1] += a.kind != "fold"  # they paid us
                    else:
                        d[1] += a.kind == "fold"  # the bluff worked
                    self.armed = None
            if a.kind == "call":
                self.contrib[p] += max(0, self.cur - self.bets[p])
                self.bets[p] = max(self.bets[p], self.cur)
            elif a.kind == "raise":
                self.contrib[p] += max(0, a.amount - self.bets[p])
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
            self.h_aggr.append((p, "lead" if owed == 0 else "raise",
                                self.board_count))
        if owed > 0:
            s["raise_vs_bet_n"] += 1
            s["raise_vs_bet"] += kind == "raise"
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
                "raise_vs_bet": 0, "raise_vs_bet_n": 0,
                "bluff": 0, "bluff_n": 0,
                # Decision feedback: label -> [times, successes].
                "dec": {},
                "calls_n": 0, "calls_surplus": 0.0,
                # Safety switch: hands and luck-free big blinds this
                # window, whether we are on the default strategy.
                "guard": {"n": 0, "sum": 0.0, "safe": False, "switches": 0},
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
        name = gs.teams[opp]
        if self._team(name)["guard"]["safe"]:
            return ARCHETYPES["unknown"]  # the safety switch is on
        return ARCHETYPES[self._archetype(name)]

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
