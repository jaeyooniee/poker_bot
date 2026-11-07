"""
fight_bots: a heads-up no-limit hold'em bot for this arena.

The format it is built for (README / CLAUDE.md): one heads-up round robin,
each pairing a duplicate match, stacks reset to 500 every hand with blinds
10/20 (25 big blinds deep), best bb/100 wins. It still plays legally at a
full table or with short stacks, because tests/test_model.py checks both.

Where the strategy comes from:

* Postflop is the Gunslinger bot (tumbleweed repo, bots/gunslinger/bot.py)
  ported to this engine, heads-up strategy included in full:
    - hu_dry: heads-up (one opponent left), a board with no flush, paired
      or straight danger, and an opponent who is not a calling station.
      When I can bet first there, I bet medium hands (equity 0.45-0.75)
      for 40% of the pot 65% of the time, semi-bluff (equity >= 0.55) for
      50% of the pot 80% of the time, and bluff (equity < 0.30) for 66% of
      the pot 50% of the time.
    - no multiway equity discount heads-up (more than one opponent shrinks
      equity to eq ** (1 + 0.35 * (opponents - 1))).
    - equity by Monte Carlo against the range the opponent's play implies,
      that range re-ranked on the board by how big their bet is, and a
      brake that will not stack off one pair into an obvious flush,
      straight or full house.
    - reads on the opponent, kept per team name across hands: aggression
      factor (station below 1.0), fold to c-bet (overfolder from 0.60),
      3-bet, fold to 3-bet and fold to 4-bet frequency, tightness.
* Preflop is Gunslinger's decision tree, adapted to 25 big blinds heads-up:
    - open 2.5 big blinds (Gunslinger: 2.5 x pot), raise a limp to 2 x pot.
    - any preflop raise that would put 40% or more of my stack in goes
      all-in instead (Gunslinger capped raises at 85% of the stack, which
      at 25 big blinds just leaves a stack too short to fold).
    - new: facing an all-in, or a raise that commits 40% of my stack, the
      decision is equity against the raiser's modelled range versus the
      exact pot odds, and a raiser who raises half their actions or more
      is given a wide range, 80% or more any two cards (so all-in bots
      get called correctly).
    - new: heads-up, the small blind opens any two cards against a big
      blind seen folding to 65% or more of opens (10 spots at least).

Engine rules this file keeps (CLAUDE.md, "Rules for Claude"): only names in
GameStateView.PUBLIC are read, attributes are plain data (numbers, lists,
dicts, PlayerAction and Card, random.Random), imports are flat and on the
organisers' safe list, and every move is checked against the legal range
before it is returned.
"""

import random
from itertools import combinations

from card_utils import SUITS, Card, evaluate, full_deck
from game_state import GameStateView
from player_action import DEALER, PlayerAction

# -- preflop hand strength --------------------------------------------------

# (high rank, low rank, suited): equity against a random hand. Gunslinger's
# table, kept as-is because every threshold below was tuned against it.
PREFLOP_EQ = {
    ('2','2',False): 0.49, ('3','2',False): 0.315, ('3','2',True): 0.362,
    ('3','3',False): 0.53, ('4','2',False): 0.366, ('4','2',True): 0.366,
    ('4','3',False): 0.3555, ('4','3',True): 0.393, ('4','4',False): 0.555,
    ('5','2',False): 0.3605, ('5','2',True): 0.385, ('5','3',False): 0.3565,
    ('5','3',True): 0.3855, ('5','4',False): 0.392, ('5','4',True): 0.417,
    ('5','5',False): 0.6165, ('6','2',False): 0.3225, ('6','2',True): 0.347,
    ('6','3',False): 0.329, ('6','3',True): 0.3815, ('6','4',False): 0.368,
    ('6','4',True): 0.424, ('6','5',False): 0.4125, ('6','5',True): 0.4285,
    ('6','6',False): 0.625, ('7','2',False): 0.322, ('7','2',True): 0.4105,
    ('7','3',False): 0.358, ('7','3',True): 0.3925, ('7','4',False): 0.382,
    ('7','4',True): 0.421, ('7','5',False): 0.417, ('7','5',True): 0.419,
    ('7','6',False): 0.4225, ('7','6',True): 0.4475, ('7','7',False): 0.657,
    ('8','2',False): 0.361, ('8','2',True): 0.394, ('8','3',False): 0.3655,
    ('8','3',True): 0.437, ('8','4',False): 0.4005, ('8','4',True): 0.4145,
    ('8','5',False): 0.4185, ('8','5',True): 0.4595, ('8','6',False): 0.405,
    ('8','6',True): 0.442, ('8','7',False): 0.467, ('8','7',True): 0.465,
    ('8','8',False): 0.695, ('9','2',False): 0.4045, ('9','2',True): 0.451,
    ('9','3',False): 0.3825, ('9','3',True): 0.441, ('9','4',False): 0.4,
    ('9','4',True): 0.45, ('9','5',False): 0.4375, ('9','5',True): 0.4545,
    ('9','6',False): 0.422, ('9','6',True): 0.48, ('9','7',False): 0.48,
    ('9','7',True): 0.475, ('9','8',False): 0.472, ('9','8',True): 0.509,
    ('9','9',False): 0.7225, ('A','2',False): 0.5395, ('A','2',True): 0.5885,
    ('A','3',False): 0.509, ('A','3',True): 0.592, ('A','4',False): 0.5545,
    ('A','4',True): 0.5835, ('A','5',False): 0.593, ('A','5',True): 0.5945,
    ('A','6',False): 0.5515, ('A','6',True): 0.58, ('A','7',False): 0.567,
    ('A','7',True): 0.626, ('A','8',False): 0.6005, ('A','8',True): 0.6,
    ('A','9',False): 0.62, ('A','9',True): 0.6255, ('A','A',False): 0.853,
    ('A','J',False): 0.6695, ('A','J',True): 0.662, ('A','K',False): 0.643,
    ('A','K',True): 0.6535, ('A','Q',False): 0.5985, ('A','Q',True): 0.656,
    ('A','T',False): 0.622, ('A','T',True): 0.649, ('J','2',False): 0.447,
    ('J','2',True): 0.487, ('J','3',False): 0.4725, ('J','3',True): 0.4645,
    ('J','4',False): 0.468, ('J','4',True): 0.495, ('J','5',False): 0.476,
    ('J','5',True): 0.5065, ('J','6',False): 0.4585, ('J','6',True): 0.4745,
    ('J','7',False): 0.4985, ('J','7',True): 0.502, ('J','8',False): 0.529,
    ('J','8',True): 0.518, ('J','9',False): 0.5185, ('J','9',True): 0.553,
    ('J','J',False): 0.75, ('J','T',False): 0.5835, ('J','T',True): 0.5955,
    ('K','2',False): 0.506, ('K','2',True): 0.5195, ('K','3',False): 0.503,
    ('K','3',True): 0.5325, ('K','4',False): 0.523, ('K','4',True): 0.555,
    ('K','5',False): 0.5365, ('K','5',True): 0.5735, ('K','6',False): 0.526,
    ('K','6',True): 0.5865, ('K','7',False): 0.548, ('K','7',True): 0.5785,
    ('K','8',False): 0.5395, ('K','8',True): 0.5745, ('K','9',False): 0.563,
    ('K','9',True): 0.6115, ('K','J',False): 0.59, ('K','J',True): 0.6155,
    ('K','K',False): 0.819, ('K','Q',False): 0.6195, ('K','Q',True): 0.649,
    ('K','T',False): 0.6035, ('K','T',True): 0.6155, ('Q','2',False): 0.456,
    ('Q','2',True): 0.529, ('Q','3',False): 0.504, ('Q','3',True): 0.505,
    ('Q','4',False): 0.4835, ('Q','4',True): 0.513, ('Q','5',False): 0.5215,
    ('Q','5',True): 0.5325, ('Q','6',False): 0.5085, ('Q','6',True): 0.5515,
    ('Q','7',False): 0.5225, ('Q','7',True): 0.5345, ('Q','8',False): 0.565,
    ('Q','8',True): 0.56, ('Q','9',False): 0.5535, ('Q','9',True): 0.563,
    ('Q','J',False): 0.582, ('Q','J',True): 0.6, ('Q','Q',False): 0.787,
    ('Q','T',False): 0.592, ('Q','T',True): 0.599, ('T','2',False): 0.424,
    ('T','2',True): 0.4465, ('T','3',False): 0.4315, ('T','3',True): 0.459,
    ('T','4',False): 0.4355, ('T','4',True): 0.4895, ('T','5',False): 0.4495,
    ('T','5',True): 0.4595, ('T','6',False): 0.459, ('T','6',True): 0.512,
    ('T','7',False): 0.5155, ('T','7',True): 0.503, ('T','8',False): 0.49,
    ('T','8',True): 0.539, ('T','9',False): 0.544, ('T','9',True): 0.547,
    ('T','T',False): 0.752,
}


def hand_class(hole):
    """Two Cards as (high rank, low rank, suited), the PREFLOP_EQ key."""
    high, low = hole
    if high.value < low.value:
        high, low = low, high
    if high.value == low.value:
        return (high.rank, low.rank, False)
    return (high.rank, low.rank, high.suit == low.suit)


def preflop_equity(hole):
    return PREFLOP_EQ.get(hand_class(hole), 0.40)


def _build_range(min_eq):
    return [k for k, v in PREFLOP_EQ.items() if v >= min_eq]


RANGE_PREMIUM = _build_range(0.70)
RANGE_STRONG = _build_range(0.62)
RANGE_TIGHT = _build_range(0.55)
RANGE_STANDARD = _build_range(0.50)
RANGE_WIDE = _build_range(0.46)
RANGE_VERY_WIDE = _build_range(0.40)
RANGE_ANY = list(PREFLOP_EQ)
# Widest first: one step right is one step tighter.
RANGE_LADDER = (RANGE_VERY_WIDE, RANGE_WIDE, RANGE_STANDARD, RANGE_TIGHT,
                RANGE_STRONG, RANGE_PREMIUM)


def _class_combos(key):
    r1, r2, suited = key
    if r1 == r2:
        return [(Card(r1, a), Card(r2, b)) for a, b in combinations(SUITS, 2)]
    if suited:
        return [(Card(r1, s), Card(r2, s)) for s in SUITS]
    return [(Card(r1, a), Card(r2, b)) for a in SUITS for b in SUITS if a != b]


# Every class's two-card combos, built once at import.
COMBOS = {key: _class_combos(key) for key in PREFLOP_EQ}
FULL_DECK = full_deck()


def expand_combos(range_classes, blocked):
    """The combos of range_classes that share no card with blocked."""
    return [
        (c1, c2)
        for key in range_classes
        for c1, c2 in COMBOS[key]
        if c1 not in blocked and c2 not in blocked
    ]


def equity_vs_range(rng, hole, board, opp_combos, n_trials):
    """Share of the pot I win against a random combo of opp_combos, with
    the rest of the board dealt at random; ties count half."""
    if not opp_combos:
        return 0.5
    hole = list(hole)
    board = list(board)
    used = set(hole) | set(board)
    deck = [c for c in FULL_DECK if c not in used]
    need = 5 - len(board)
    wins = ties = total = 0
    for _ in range(n_trials):
        c1, c2 = rng.choice(opp_combos)
        if c1 in used or c2 in used:
            continue
        remaining = [c for c in deck if c != c1 and c != c2]
        full_board = board + (rng.sample(remaining, need) if need else [])
        mine = evaluate(hole + full_board)
        theirs = evaluate([c1, c2] + full_board)
        if mine > theirs:
            wins += 1
        elif mine == theirs:
            ties += 1
        total += 1
    if total == 0:
        return 0.5
    return (wins + ties / 2.0) / total


# -- board texture (Gunslinger) ---------------------------------------------
# Category numbers are card_utils.CATEGORIES indices.
TRIPS, STRAIGHT, FLUSH = 3, 4, 5


def _made_rank(hole, board):
    if len(board) < 3:
        return 0
    return evaluate(list(hole) + list(board))[0]


def _board_danger(board):
    """{"flush", "boat", "straight"}: what the board itself threatens."""
    danger = set()
    if len(board) < 3:
        return danger
    suits = [c.suit for c in board]
    if max(suits.count(s) for s in set(suits)) >= 3:
        danger.add("flush")
    values = [c.value for c in board]
    if len(set(values)) < len(values):
        danger.add("boat")
    vals = sorted(set(values))
    if 14 in vals:
        vals = sorted(set(vals) | {1})
    for v in vals:
        if len([w for w in vals if v <= w <= v + 4]) >= 3:
            danger.add("straight")
            break
    return danger


def _vulnerable(my_rank, danger):
    if "flush" in danger and my_rank < FLUSH:
        return True
    if "boat" in danger and my_rank <= TRIPS:
        return True
    if "straight" in danger and my_rank < STRAIGHT:
        return True
    return False


def _board_condition_range(opp_combos, board, bet_ratio):
    """Keeps the part of the range that is strongest on this board: the
    bigger the bet, the stronger the part kept."""
    if len(board) < 3 or len(opp_combos) <= 8:
        return opp_combos
    if bet_ratio >= 1.0:
        keep = 0.40
    elif bet_ratio >= 0.7:
        keep = 0.55
    elif bet_ratio >= 0.4:
        keep = 0.72
    else:
        keep = 0.90
    board = list(board)
    ranked = sorted(opp_combos, key=lambda c: evaluate([c[0], c[1]] + board),
                    reverse=True)
    return ranked[:max(8, int(len(ranked) * keep))]


# -- opponent reads (Gunslinger), kept per team name ------------------------


def _ratio(num, den, default, min_den):
    return (num / den) if den >= min_den else default


def estimate_opp_tightness(s):
    """0 = loose/aggressive, 1 = tight/passive, 0.5 = no sample yet."""
    if not s or s.get("actions", 0) < 6:
        return 0.5
    actions = max(1, s["actions"])
    raise_freq = s.get("raises", 0) / actions
    fold_freq = s.get("folds", 0) / actions
    return max(0.05, min(0.95, 0.5 + 0.5 * fold_freq - 0.5 * raise_freq))


def opp_threebet_freq(s, default=0.10):
    s = s or {}
    return _ratio(s.get("tb_made", 0), s.get("tb_opp", 0), default, 9)


def opp_fold_to_fourbet(s, default=0.50):
    s = s or {}
    return _ratio(s.get("f4_fold", 0), s.get("f4_opp", 0), default, 6)


def opp_fold_to_cbet(s, default=0.45):
    s = s or {}
    return _ratio(s.get("fc_fold", 0), s.get("fc_opp", 0), default, 7)


def opp_af(s, default=2.0):
    """Aggression factor raises/calls: below 1 a station, above 3 a maniac."""
    s = s or {}
    agg, cal = s.get("agg", 0), s.get("cal", 0)
    if agg + cal < 6:
        return default
    return agg / max(1, cal)


def opp_fold_to_open(s):
    """How often they fold facing a single preflop raise, or None before
    10 such spots: new here, for stealing."""
    s = s or {}
    if s.get("tb_opp", 0) < 10:
        return None
    return s.get("fo_fold", 0) / s["tb_opp"]


def maniac_range(s):
    """New here: the range of a player who raises most of the time (all-in
    bots), or None. Half their actions or more: very wide; 80% or more:
    any two cards."""
    if not s or s.get("actions", 0) < 12:
        return None
    raise_freq = s.get("raises", 0) / s["actions"]
    if raise_freq >= 0.8:
        return RANGE_ANY
    if raise_freq >= 0.5:
        return RANGE_VERY_WIDE
    return None


def opp_range(s, was_raiser, was_caller):
    tight = estimate_opp_tightness(s)
    if was_raiser:
        wide = maniac_range(s)
        if wide is not None:
            return wide
        if tight > 0.70:
            return RANGE_PREMIUM
        if tight > 0.55:
            return RANGE_STRONG
        if tight > 0.40:
            return RANGE_TIGHT
        return RANGE_STANDARD
    if was_caller:
        if tight > 0.60:
            return RANGE_TIGHT
        return RANGE_STANDARD
    if tight > 0.50:
        return RANGE_WIDE
    return RANGE_VERY_WIDE


def raiser_range(s, n_raises):
    """The range behind a preflop raise: the opener's range, one step
    tighter for a 3-bet (unless they 3-bet light), two for a 4-bet."""
    wide = maniac_range(s)
    if wide is not None:
        return wide
    step = RANGE_LADDER.index(opp_range(s, True, False))
    if n_raises == 2 and opp_threebet_freq(s) < 0.20:
        step += 1
    elif n_raises >= 3:
        step += 2
    return RANGE_LADDER[min(step, len(RANGE_LADDER) - 1)]


# -- replaying the hand's actions -------------------------------------------


def replay(actions, button, num_players, small_blind, big_blind):
    """
    Walks the hand's actions in order and returns (events, info).

    events: one (street, player, kind, facing, n_raises, pf_aggressor) per
    betting move, where kind is "fold", "check", "call" or "raise", facing
    means chips were owed, n_raises counts raises earlier on that street,
    and pf_aggressor is the last preflop raiser so far (or None).

    info: where the hand stands now: street (0 preflop .. 3 river),
    pf_raises, acted_pf (players who made a preflop move), last_raiser and
    callers (players who called since the last raise).
    """
    sb = button if num_players == 2 else (button + 1) % num_players
    bb = (sb + 1) % num_players
    street = 0
    board = 0
    bets = {sb: small_blind, bb: big_blind}
    current = big_blind
    n_raises = 0
    pf_raises = 0
    pf_aggressor = None
    acted_pf = set()
    last_raiser = None
    callers = []
    events = []
    for action in actions:
        if action.player_id == DEALER:
            if action.kind == "reveal_card":
                board += 1
                if board in (3, 4, 5):
                    street = board - 2
                    bets = {}
                    current = 0
                    n_raises = 0
            continue
        if action.kind not in ("fold", "call", "raise"):
            continue  # showdown cards and pot awards
        pid = action.player_id
        facing = bets.get(pid, 0) < current
        kind = action.kind
        if kind == "call" and not facing:
            kind = "check"
        events.append((street, pid, kind, facing, n_raises, pf_aggressor))
        if street == 0:
            acted_pf.add(pid)
        if kind == "raise":
            bets[pid] = action.amount
            current = max(current, action.amount)
            n_raises += 1
            last_raiser = pid
            callers = []
            if street == 0:
                pf_raises += 1
                pf_aggressor = pid
        elif kind == "call":
            bets[pid] = current
            callers.append(pid)
    info = {
        "street": street,
        "pf_raises": pf_raises,
        "acted_pf": acted_pf,
        "last_raiser": last_raiser,
        "callers": callers,
    }
    return events, info


def learn(opp, events, teams, me):
    """Adds one finished hand's events to the per-team counters in opp."""
    for street, pid, kind, facing, n_raises, pf_aggressor in events:
        if pid == me:
            continue
        s = opp.setdefault(teams[pid], {})

        def bump(key):
            s[key] = s.get(key, 0) + 1

        bump("actions")
        if kind == "fold":
            bump("folds")
        elif kind == "check":
            bump("checks")
        elif kind == "call":
            bump("calls")
            bump("cal")
        elif kind == "raise":
            bump("raises")
            bump("agg")

        if street == 0 and facing:
            if n_raises == 1:  # facing an open: a 3-bet spot
                bump("tb_opp")
                if kind == "raise":
                    bump("tb_made")
                elif kind == "fold":
                    bump("fo_fold")
            elif n_raises == 2:  # facing a 3-bet
                bump("f3_opp")
                if kind == "fold":
                    bump("f3_fold")
            elif n_raises >= 3:  # facing a 4-bet or more
                bump("f4_opp")
                if kind == "fold":
                    bump("f4_fold")
        elif street == 1:
            if pid == pf_aggressor and n_raises == 0 and not facing:
                bump("cb_opp")  # a c-bet spot
                if kind == "raise":
                    bump("cb_made")
            elif (pf_aggressor is not None and pid != pf_aggressor
                  and facing and n_raises >= 1):
                bump("fc_opp")  # faced a flop bet
                if kind == "fold":
                    bump("fc_fold")


# -- the bot ----------------------------------------------------------------

# A preflop raise putting at least this share of my stack in goes all-in.
COMMIT_FRACTION = 0.40
PREFLOP_TRIALS = 300
POSTFLOP_TRIALS = 160


class Model:
    def __init__(self):
        self.hand = []
        self.player_id = None
        self.log = []  # every action of this hand so far
        self.opp = {}  # team name -> counters, kept across hands
        self.rng = random.Random(20261107)
        # True only in local tests: errors raise instead of check/fold.
        self.strict = False

    def seed(self, hand, player_id):
        self.hand = list(hand)
        self.player_id = player_id
        self.log = []

    def run(self, actions, game_state: GameStateView) -> PlayerAction:
        self.log.extend(actions)
        try:
            return self._decide(game_state)
        except Exception:
            if self.strict:
                raise
            return self._check_or_fold(game_state)

    def end_game(self, actions, game_state: GameStateView):
        self.log.extend(actions)
        try:
            events, _ = self._replay(game_state)
            learn(self.opp, events, game_state.teams, self.player_id)
        except Exception:
            if self.strict:
                raise

    # -- helpers ------------------------------------------------------------

    def _replay(self, gs):
        return replay(self.log, gs.button, gs.num_players, gs.small_blind,
                      gs.big_blind)

    def _stats(self, gs, pid):
        if pid is None:
            return None
        return self.opp.get(gs.teams[pid])

    def _check_or_fold(self, gs):
        if gs.call_amount == 0:
            return PlayerAction.call(self.player_id)
        return PlayerAction.fold(self.player_id)

    def _identify(self, gs, info):
        """(primary, is_raiser, is_caller): the last raiser still in, else
        the first caller since then still in, else (None, False, False)."""
        me = self.player_id
        in_hand = set(gs.players_in_hand)
        raiser = info["last_raiser"]
        if raiser is not None and raiser != me and raiser in in_hand:
            return raiser, True, False
        for pid in info["callers"]:
            if pid != me and pid in in_hand:
                return pid, False, True
        return None, False, False

    def _jam(self, gs):
        if gs.can_raise:
            return PlayerAction.raise_to(self.player_id, gs.max_raise_to)
        return PlayerAction.call(self.player_id)

    def _preflop_raise(self, gs, target):
        """Raise to target, all-in once it commits COMMIT_FRACTION of my
        stack, kept in the legal range; a call when raising is not legal."""
        me = self.player_id
        if not gs.can_raise:
            return PlayerAction.call(me)
        low, high = gs.raise_range
        total = int(target)
        if total >= COMMIT_FRACTION * (gs.bets[me] + gs.stacks[me]):
            total = high
        return PlayerAction.raise_to(me, min(high, max(low, total)))

    def _postflop_raise(self, gs, target):
        """Raise (or bet) to target within the legal range; a call (or
        check) when raising is not legal."""
        me = self.player_id
        if not gs.can_raise:
            return PlayerAction.call(me)
        low, high = gs.raise_range
        return PlayerAction.raise_to(me, min(high, max(low, int(target))))

    # -- decisions ----------------------------------------------------------

    def _decide(self, gs):
        _, info = self._replay(gs)
        primary, is_raiser, is_caller = self._identify(gs, info)
        if gs.street == "preflop":
            return self._preflop(gs, info, primary)
        if primary is None:
            # Nobody bet or called yet this hand: read the first other
            # player still in (heads-up, the only opponent).
            others = [p for p in gs.players_in_hand if p != self.player_id]
            primary = others[0] if others else None
        return self._postflop(gs, primary, is_raiser, is_caller)

    def _preflop(self, gs, info, primary):
        me = self.player_id
        hole = self.hand
        pot = gs.pot
        owed = gs.call_amount
        stack = gs.stacks[me]
        bet_this = gs.bets[me]
        current_bet = gs.current_bet
        min_raise = gs.min_raise_to if gs.can_raise else current_bet
        bb = gs.big_blind
        stats = self._stats(gs, primary)

        eq = preflop_equity(hole)
        my_total = stack + bet_this  # chips at risk this hand
        bb_remaining = max(1, my_total / bb)
        n_raises = info["pf_raises"]
        # Players still to act behind me this round (Gunslinger's position).
        n_behind = sum(1 for p in gs.active_players
                       if p != me and p not in info["acted_pf"])
        opp_tight = estimate_opp_tightness(stats) if primary is not None \
            else 0.5

        if bb_remaining < 25 and eq >= 0.70 and my_total <= 15 * bb:
            return self._jam(gs)

        if n_raises >= 1 and owed == 0:
            return PlayerAction.call(me)  # nothing owed: check

        if n_raises >= 1:
            others_in = [p for p in gs.players_in_hand if p != me]
            opp_all_in = any(gs.status(p) == "all_in" for p in others_in)
            if opp_all_in or bet_this + owed >= COMMIT_FRACTION * my_total:
                return self._vs_big_bet(gs, stats, n_raises, opp_all_in)

        if n_raises == 0:
            if owed == 0:
                # Big blind after limps: raise strong hands, check the rest.
                if eq >= 0.55:
                    return self._preflop_raise(gs, max(min_raise, 2 * pot))
                return PlayerAction.call(me)
            open_thresh = 0.50 - 0.10 * (1 - opp_tight)
            open_thresh += 0.015 * (n_behind - 2)  # wider in late position
            open_thresh = max(0.42, min(0.62, open_thresh))
            # New here: heads-up, steal with any two cards from a big
            # blind that folds to opens 65% of the time or more.
            others_in = [p for p in gs.players_in_hand if p != me]
            if len(others_in) == 1:
                folds = opp_fold_to_open(self._stats(gs, others_in[0]))
                if folds is not None and folds >= 0.65:
                    open_thresh = 0.0
            if eq >= open_thresh:
                return self._preflop_raise(gs, max(min_raise, int(2.5 * bb)))
            return PlayerAction.fold(me)

        if n_raises == 1:
            # Facing a single open: value 3-bet, blocker 3-bet bluff, call
            # or set-mine, else fold.
            bet_ratio = owed / max(1, pot)
            three_thresh = 0.64 - 0.07 * (1 - opp_tight)
            if bet_ratio <= 0.5:
                call_thresh = 0.48 - 0.08 * (1 - opp_tight)
                call_max_ratio = 0.5
            else:
                call_thresh = 0.55 - 0.08 * (1 - opp_tight)
                call_max_ratio = 0.65
            in_position = n_behind == 0
            has_blocker = any(c.rank in ("A", "K") for c in hole)
            if eq >= three_thresh:
                return self._preflop_raise(
                    gs, max(min_raise, current_bet * 3))
            if (in_position and has_blocker and 0.45 <= eq < three_thresh
                    and owed <= pot * 1.2 and self.rng.random() < 0.30):
                return self._preflop_raise(
                    gs, max(min_raise, current_bet * 3))
            if eq >= call_thresh and bet_ratio <= call_max_ratio:
                return PlayerAction.call(me)
            is_pair = hole[0].rank == hole[1].rank
            if (is_pair and bb_remaining >= 22 and owed <= 0.06 * my_total
                    and owed <= pot * 1.2):
                return PlayerAction.call(me)
            return PlayerAction.fold(me)

        # Facing a 3-bet or more.
        tb = opp_threebet_freq(stats)
        f4 = opp_fold_to_fourbet(stats)
        has_blocker = any(c.rank in ("A", "K") for c in hole)
        is_ak = {c.rank for c in hole} == {"A", "K"}
        big_bet = owed > pot or owed > 0.5 * my_total

        if n_raises >= 3:  # facing a 4-bet or more: never 5-bet bluff
            if eq >= 0.80:
                if big_bet:
                    return PlayerAction.call(me)
                return self._preflop_raise(
                    gs, max(min_raise, int(current_bet * 2.2)))
            if eq >= 0.74 and owed <= pot * 0.7:
                return PlayerAction.call(me)
            return PlayerAction.fold(me)

        if eq >= 0.78:  # premium: get it in
            if big_bet:
                return PlayerAction.call(me)
            return self._preflop_raise(
                gs, max(min_raise, int(current_bet * 2.3)))
        if (eq >= 0.70 or is_ak) and not big_bet:  # value 4-bet
            return self._preflop_raise(
                gs, max(min_raise, int(current_bet * 2.3)))
        # 4-bet bluff only against a player who has folded to 4-bets.
        if (has_blocker and 0.47 <= eq < 0.66 and not big_bet
                and owed <= pot * 1.0 and f4 >= 0.55
                and self.rng.random() < 0.5):
            return self._preflop_raise(
                gs, max(min_raise, int(current_bet * 2.2)))
        if tb >= 0.20 and eq >= 0.56 and owed <= pot * 0.7:
            return PlayerAction.call(me)
        if eq >= 0.66 and owed <= pot * 0.7:
            return PlayerAction.call(me)
        return PlayerAction.fold(me)

    def _vs_big_bet(self, gs, stats, n_raises, opp_all_in):
        """Facing an all-in or a raise that commits me: my equity against
        the raiser's range versus the price of calling."""
        me = self.player_id
        hole = self.hand
        combos = expand_combos(raiser_range(stats, n_raises), set(hole))
        eq = equity_vs_range(self.rng, hole, [], combos, PREFLOP_TRIALS)
        stack = gs.stacks[me]
        bet_this = gs.bets[me]
        call = min(gs.call_amount, stack)
        # Chips the raiser bet beyond what I can match come back to them.
        returned = max(0, gs.current_bet - (bet_this + call))
        need = call / max(1, gs.pot + call - returned)
        if gs.can_raise and not opp_all_in and eq >= 0.58:
            return self._jam(gs)
        if eq >= need + 0.02:
            return PlayerAction.call(me)
        return PlayerAction.fold(me)

    def _postflop(self, gs, primary, is_raiser, is_caller):
        """Gunslinger's postflop, heads-up strategy included."""
        me = self.player_id
        hole = self.hand
        board = list(gs.board)
        pot = gs.pot
        owed = gs.call_amount
        stack = gs.stacks[me]
        bet_this = gs.bets[me]
        can_check = owed == 0
        current_bet = gs.current_bet
        min_raise = gs.min_raise_to if gs.can_raise else current_bet
        stats = self._stats(gs, primary)

        opp_combos = expand_combos(opp_range(stats, is_raiser, is_caller),
                                   set(hole) | set(board))
        # Re-rank their range on this board before measuring against it.
        if owed > 0:
            opp_combos = _board_condition_range(opp_combos, board,
                                                owed / max(1, pot))
        eq = equity_vs_range(self.rng, hole, board, opp_combos,
                             POSTFLOP_TRIALS)

        # Board-texture brake.
        my_rank = _made_rank(hole, board)
        danger = _board_danger(board)
        vulnerable = _vulnerable(my_rank, danger)
        commit_frac = owed / max(1, stack + bet_this)
        big_commit = owed > pot or commit_frac > 0.45

        # Opponents still in the pot (all-in ones too: they can still win).
        n_active = sum(1 for p in gs.players_in_hand if p != me)
        if n_active > 1:
            # Multiway discount; heads-up the raw equity stands.
            eq = eq ** (1 + 0.35 * (n_active - 1))

        odds_needed = owed / max(1, pot + owed)

        af = opp_af(stats)
        fc = opp_fold_to_cbet(stats)
        station = af < 1.0  # passive caller: never bluff, value bigger
        overfolder = fc >= 0.60  # folds to c-bets: barrel more

        if can_check:
            if eq >= 0.75 or (station and eq >= 0.62):  # value
                ot = estimate_opp_tightness(stats) if primary is not None \
                    else 0.5
                size_frac = 0.75 if (ot > 0.5 and not station) else 1.0
                if vulnerable:
                    size_frac = min(size_frac, 0.5)
                return self._postflop_raise(
                    gs, max(min_raise, int(pot * size_frac)))
            semibluff_p = 0.0 if station else (0.55 if overfolder else 0.4)
            bluff_p = 0.0 if station else (0.22 if overfolder else 0.12)
            # Heads-up on a dry board against a non-station: take the lead.
            hu_dry = n_active == 1 and not danger and not station
            if hu_dry:
                semibluff_p = max(semibluff_p, 0.80)
                bluff_p = max(bluff_p, 0.50)
                # Bet medium hands small instead of check-folding them.
                if (0.45 <= eq < 0.75 and not vulnerable
                        and self.rng.random() < 0.65):
                    return self._postflop_raise(
                        gs, max(min_raise, int(pot * 0.4)))
            if eq >= 0.55 and self.rng.random() < semibluff_p:
                return self._postflop_raise(
                    gs, max(min_raise, int(pot * 0.5)))
            if eq < 0.30 and not danger and self.rng.random() < bluff_p:
                return self._postflop_raise(
                    gs, max(min_raise, int(pot * 0.66)))
            return PlayerAction.call(me)  # check

        # Facing a bet. Do not raise a vulnerable hand into a big commitment.
        if eq >= 0.78 and not (vulnerable and big_commit):
            return self._postflop_raise(
                gs, max(min_raise, int(pot * 1.0) + current_bet))
        if eq >= odds_needed + 0.02:
            if eq < 0.30 and owed / max(1, pot) > 1.5:
                return PlayerAction.fold(me)
            # Do not call off a big share of the stack with a hand the
            # obvious nuts beat.
            if big_commit and vulnerable and eq < odds_needed + 0.12:
                return PlayerAction.fold(me)
            return PlayerAction.call(me)
        return PlayerAction.fold(me)
