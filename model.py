import math
import random

from card_utils import Card, evaluate, full_deck
from game_state import GameStateView
from player_action import DEALER, PlayerAction
from preflop_table import PREFLOP_EQUITY

# Monte Carlo samples per decision (an evaluate call is ~40 us).
PREFLOP_SAMPLES = 300
POSTFLOP_SAMPLES = 400

# Bayesian priors for opponent stats: (prior rate, weight in hands/actions).
PRIOR_PF_RAISE = (0.35, 6)  # share of hands an opponent raises preflop
PRIOR_POST_AGGR = (0.3, 10)  # share of postflop actions that are raises
PRIOR_FOLD_TO_RAISE = (0.35, 6)  # share of our raises they fold to

_RNG = random.Random(20261107)


def hand_key(cards) -> str:
    """"AKs", "AKo" or "AA" for two hole cards."""
    a, b = sorted(cards, key=lambda c: c.value, reverse=True)
    if a.rank == b.rank:
        return a.rank + b.rank
    return a.rank + b.rank + ("s" if a.suit == b.suit else "o")


class Model:
    """
    Equity-driven heads-up bot with simple opponent modelling.

    Preflop: equity from a precomputed table (vs a random hand), or by Monte
    Carlo against the opponent's estimated raising range when they raised.
    Postflop: Monte Carlo equity against their range, narrowed to their
    strongest current hands when they have bet or raised this hand.
    Opponent stats (per team name) are learned in end_game and live as long
    as this object does.
    """

    def __init__(self):
        self.hand = []
        self.player_id = None
        # Share of all 1326 combos at least as strong as each starting hand.
        self.percentile = self._percentiles()
        # Learned per opponent team: counts.
        self.stats = {}
        # Per-hand bookkeeping, reset in seed.
        self._reset_hand()

    # -- engine hooks --------------------------------------------------------

    def seed(self, hand: list[Card], player_id: int):
        self.hand = list(hand)
        self.player_id = player_id
        self._reset_hand()

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
        for p in self.pf_acted:
            s = self._team(teams[p])
            s["pf_hands"] += 1
            if p in self.pf_raised:
                s["pf_raises"] += 1

    # -- decisions -----------------------------------------------------------

    def _preflop(self, gs):
        me = self.player_id
        key = hand_key(self.hand)
        bb = gs.big_blind
        aggressor = self._aggressor(gs)
        opened = gs.current_bet > bb  # someone has raised

        if not opened:
            fold_rate = self._fold_to_raise(gs)
            base = PREFLOP_EQUITY[key]
            if gs.call_amount > 0:  # small blind / first in
                # Open wider against players who fold to raises.
                threshold = 0.42 - 0.15 * max(0.0, fold_rate - 0.4)
                if base >= threshold:
                    size = 3 * bb if base >= 0.6 and fold_rate < 0.3 else 2.5 * bb
                    return self._raise(gs, size)
                return self._check_or_fold(gs)
            # Big blind after a limp: raise strong hands, otherwise check.
            if base >= 0.58:
                return self._raise(gs, 3 * gs.current_bet)
            return PlayerAction.call(me)

        # Facing a raise: equity against the raiser's estimated range.
        width = self._pf_raise_rate(gs.teams[aggressor]) if aggressor is not None else 1.0
        width = min(1.0, max(0.08, width))
        equity = self._equity(gs, width, 1.0, PREFLOP_SAMPLES)
        pot_odds = gs.pot_odds
        if equity >= 0.62 and gs.can_raise:
            return self._raise(gs, 3 * gs.current_bet)
        if equity > pot_odds + 0.02:
            return PlayerAction.call(me)
        return self._check_or_fold(gs)

    def _postflop(self, gs):
        me = self.player_id
        aggressor = self._aggressor(gs)
        width = 1.0
        narrow = 1.0
        if self.pf_aggressor is not None:
            team = gs.teams[self.pf_aggressor]
            width = min(1.0, max(0.08, self._pf_raise_rate(team)))
        raisers = [p for p in self.post_raises if not gs.folded[p]]
        if raisers:
            p = aggressor if aggressor in raisers else raisers[0]
            aggr = self._post_aggr(gs.teams[p])
            narrow = max(0.12, aggr ** self.post_raises[p])
        equity = self._equity(gs, width, narrow, POSTFLOP_SAMPLES)
        fold_rate = self._fold_to_raise(gs)
        pot = gs.pot

        if gs.call_amount == 0:
            if equity >= 0.65:
                frac = 1.0 if fold_rate < 0.25 else 0.66
                return self._raise(gs, gs.current_bet + frac * pot)
            if equity >= 0.5 and fold_rate > 0.5:
                return self._raise(gs, gs.current_bet + 0.5 * pot)
            if fold_rate > 0.65:  # they fold too much: bluff
                return self._raise(gs, gs.current_bet + 0.5 * pot)
            return PlayerAction.call(me)  # check

        pot_odds = gs.pot_odds
        if equity >= 0.75 and gs.can_raise:
            # Pot-sized raise: call, then add the pot.
            return self._raise(gs, gs.current_bet + pot + gs.call_amount)
        if equity > pot_odds + 0.02:
            return PlayerAction.call(me)
        return self._check_or_fold(gs)

    # -- equity --------------------------------------------------------------

    def _equity(self, gs, width, narrow, samples):
        """Share of the pot won against the opponents still in, by Monte
        Carlo. Opponent hands come from the top `width` of starting hands;
        with narrow < 1 (and a board), only the `narrow` share of samples
        where the opponent holds the strongest current hand count."""
        board = list(gs.board)
        opponents = max(1, len(gs.players_in_hand) - 1)
        dead = set(self.hand) | set(board)
        unseen = [c for c in full_deck() if c not in dead]
        missing = 5 - len(board)
        use_narrow = narrow < 1.0 and len(board) >= 3
        results = []
        for _ in range(samples):
            deck = unseen[:]
            _RNG.shuffle(deck)
            holes = []
            for _ in range(opponents):
                holes.append(self._draw_hole(deck, width))
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
        """Takes two cards off deck for an opponent, preferring hands in
        the top `width` of starting hands (rejection sampling)."""
        for _ in range(30):
            if len(deck) < 7:
                break
            i, j = len(deck) - 1, len(deck) - 2
            hole = [deck[i], deck[j]]
            if width >= 1.0 or self.percentile[hand_key(hole)] <= width:
                del deck[j:]
                return hole
            # Not in range: move this pair under the rest and try again.
            k = _RNG.randrange(len(deck) - 2)
            deck[i], deck[k] = deck[k], deck[i]
            k = _RNG.randrange(len(deck) - 2)
            deck[j], deck[k] = deck[k], deck[j]
        hole = [deck.pop(), deck.pop()]
        return hole

    @staticmethod
    def _percentiles():
        combos = {k: 6 if len(k) == 2 else 4 if k[2] == "s" else 12
                  for k in PREFLOP_EQUITY}
        result, seen = {}, 0
        for key in sorted(PREFLOP_EQUITY, key=PREFLOP_EQUITY.get, reverse=True):
            seen += combos[key]
            result[key] = seen / 1326
        return result

    # -- watching the hand ---------------------------------------------------

    def _reset_hand(self):
        self.board_count = 0
        self.pf_acted = set()  # opponents who acted preflop
        self.pf_raised = set()  # opponents who raised preflop
        self.pf_aggressor = None  # last preflop raiser (any player)
        self.post_raises = {}  # opponent id -> postflop raises this hand
        self.last_raiser = None
        self.my_raise_pending = False

    def _observe(self, actions, gs):
        teams = gs.teams
        me = self.player_id
        for a in actions:
            if a.kind == "reveal_card":
                if a.player_id == DEALER:
                    self.board_count += 1
                    self.last_raiser = None
                continue
            if a.kind not in ("fold", "call", "raise"):
                continue
            p = a.player_id
            preflop = self.board_count < 3
            if a.kind == "raise":
                self.last_raiser = p
                if preflop:
                    self.pf_aggressor = p
            if p == me:
                self.my_raise_pending = a.kind == "raise"
                continue
            s = self._team(teams[p])
            if self.my_raise_pending:
                s["resp_total"] += 1
                if a.kind == "fold":
                    s["resp_folds"] += 1
                self.my_raise_pending = False
            if preflop:
                self.pf_acted.add(p)
                if a.kind == "raise":
                    self.pf_raised.add(p)
            else:
                s["post_acts"] += 1
                if a.kind == "raise":
                    s["post_raises"] += 1
                    self.post_raises[p] = self.post_raises.get(p, 0) + 1

    def _aggressor(self, gs):
        if self.last_raiser is not None and self.last_raiser != self.player_id:
            return self.last_raiser
        return None

    # -- opponent stats ------------------------------------------------------

    def _team(self, name):
        if name not in self.stats:
            self.stats[name] = {
                "pf_hands": 0, "pf_raises": 0, "post_acts": 0,
                "post_raises": 0, "resp_total": 0, "resp_folds": 0,
            }
        return self.stats[name]

    @staticmethod
    def _rate(hits, total, prior):
        rate, weight = prior
        return (hits + rate * weight) / (total + weight)

    def _pf_raise_rate(self, team):
        s = self._team(team)
        return self._rate(s["pf_raises"], s["pf_hands"], PRIOR_PF_RAISE)

    def _post_aggr(self, team):
        s = self._team(team)
        return self._rate(s["post_raises"], s["post_acts"], PRIOR_POST_AGGR)

    def _fold_to_raise(self, gs):
        """Average fold-to-our-raise rate of the opponents still in."""
        others = [p for p in gs.players_in_hand if p != self.player_id]
        if not others:
            return 0.0
        rates = []
        for p in others:
            s = self._team(gs.teams[p])
            rates.append(self._rate(s["resp_folds"], s["resp_total"],
                                    PRIOR_FOLD_TO_RAISE))
        return sum(rates) / len(rates)

    # -- moves ---------------------------------------------------------------

    def _raise(self, gs, total):
        """A raise to about total, within the legal range; a call when we
        cannot raise. Commits all-in when the raise would use most of the
        stack anyway."""
        if not gs.can_raise:
            return PlayerAction.call(self.player_id)
        low, high = gs.raise_range
        total = int(round(total))
        if total >= 0.6 * high:
            total = high
        return PlayerAction.raise_to(self.player_id, min(high, max(low, total)))

    def _check_or_fold(self, gs):
        if gs.call_amount == 0:
            return PlayerAction.call(self.player_id)
        return PlayerAction.fold(self.player_id)
