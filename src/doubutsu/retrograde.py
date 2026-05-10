"""Retrograde analysis for Doubutsushougi.

All positions are identified by the 61-bit perfect hash from ``hash.py``.
The analysis works entirely via **backward BFS** seeded from terminal
positions that are generated combinatorially — no forward BFS is used.

Algorithm
---------
1. Generate all simple terminal states (one lion missing, or try-win with
   only the two lions) and seed the evaluation table with value ``-1``
   (current mover loses).
2. Pop each evaluated state from the queue and generate its *predecessor*
   states using reverse-move rules.
3. For each predecessor:
   - If a predecessor is itself terminal, evaluate it immediately.
   - If it has a successor that is LOSS → mark it WIN (value ``+1``).
   - Track how many successors remain un-evaluated; when the count reaches
     zero every successor was WIN → mark it LOSS (value ``-1``).
4. Positions never reached are classified as DRAW (value ``0``).
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

from .game import (
    BOARD_COLS,
    BOARD_ROWS,
    HAND_TYPES,
    Piece,
    PieceType,
    Player,
    State,
)
from .hash import position_to_index

# ---------------------------------------------------------------------------
# Terminal detection
# ---------------------------------------------------------------------------

def _last_rank(player: Player) -> int:
    """Row that *player* is trying to reach (matches game.py semantics)."""
    return 0 if player == Player.BLACK else BOARD_ROWS - 1


def is_terminal(state: State) -> bool:
    """Return True if *state* is a game-over position.

    Matches the two conditions checked in ``State.apply_move``:
    * the current mover's lion has been captured (it is absent from the board), or
    * the previous mover's lion stands on the try-rank and cannot be captured.
    """
    T = state.turn
    opp = T.opponent

    # Lion capture: the current mover's own lion is gone.
    if state._find_lion(T) is None:
        return True

    # Try-win: the *previous* mover's lion is on the try-rank (row that
    # player.opponent is aiming for, per apply_move) and is uncapturable.
    opp_lion = state._find_lion(opp)
    if opp_lion is not None:
        try_rank = _last_rank(T)          # == _last_rank(player.opponent) in apply_move
        if opp_lion[0] == try_rank:
            if not state._capturable_by(opp, opp_lion[0], opp_lion[1]):
                return True

    return False


def terminal_value(_state: State) -> int:
    """Value of a terminal state: always -1 (the player to move has lost)."""
    return -1


# ---------------------------------------------------------------------------
# Reverse-move generation
# ---------------------------------------------------------------------------

def _predecessor_states(state: State) -> List[State]:
    """Return all states that can reach *state* in exactly one legal move."""
    results: List[State] = []
    mover = state.turn.opponent   # who made the last move
    receiver = state.turn          # who will move next (did not move)

    receiver_lion_missing = state._find_lion(receiver) is None

    for to_r in range(BOARD_ROWS):
        for to_c in range(BOARD_COLS):
            piece_after = state.board[to_r][to_c]
            if piece_after is None or piece_after.owner != mover:
                continue

            # ---- reverse a BOARD MOVE ----
            # Determine which piece types could have arrived here.
            if (
                piece_after.type == PieceType.HEN
                and to_r == _last_rank(mover)
            ):
                # Could have promoted from CHICK on arrival, or was already HEN.
                pieces_before = [Piece(mover, PieceType.CHICK), piece_after]
            else:
                pieces_before = [piece_after]

            for piece_before in pieces_before:
                # CHICK cannot legally stand on its own promotion rank.
                # (It would have promoted immediately, so this square is invalid
                # as a *from* square for a chick.)
                for dr, dc in state._deltas_for_piece(piece_before):
                    from_r = to_r - dr
                    from_c = to_c - dc
                    if not state.inside(from_r, from_c):
                        continue
                    if state.board[from_r][from_c] is not None:
                        continue  # must have been empty (piece left there)

                    # What was at (to_r, to_c) before the move?
                    if receiver_lion_missing:
                        # Terminal by lion capture: the receiver's lion was here.
                        captured_options: List[Optional[Piece]] = [
                            Piece(receiver, PieceType.LION)
                        ]
                    else:
                        # No lion capture. Either nothing, or a non-lion piece.
                        captured_options = [None]
                        for cap_type in HAND_TYPES:
                            if state.hands[mover][cap_type] > 0:
                                # The piece on board could have been cap_type or
                                # HEN (which enters mover's hand as CHICK).
                                captured_options.append(Piece(receiver, cap_type))
                                if cap_type == PieceType.CHICK:
                                    captured_options.append(
                                        Piece(receiver, PieceType.HEN)
                                    )

                    for captured in captured_options:
                        pred = state.clone()
                        pred.turn = mover
                        pred.winner = None
                        pred.board[from_r][from_c] = piece_before
                        pred.board[to_r][to_c] = captured

                        if captured is not None and captured.type != PieceType.LION:
                            hand_type = (
                                PieceType.CHICK
                                if captured.type == PieceType.HEN
                                else captured.type
                            )
                            if pred.hands[mover][hand_type] <= 0:
                                continue
                            pred.hands[mover][hand_type] -= 1

                        # Validate: piece_before=CHICK cannot sit on its promotion rank.
                        if (
                            piece_before.type == PieceType.CHICK
                            and from_r == _last_rank(mover)
                        ):
                            continue

                        results.append(pred)

            # ---- reverse a DROP MOVE ----
            if (
                piece_after.type in HAND_TYPES
                and piece_after.type != PieceType.HEN   # hens cannot be dropped
            ):
                drop_type = piece_after.type
                # Illegal to drop a chick onto its promotion rank.
                if drop_type == PieceType.CHICK and to_r == _last_rank(mover):
                    continue

                pred = state.clone()
                pred.turn = mover
                pred.winner = None
                pred.board[to_r][to_c] = None
                pred.hands[mover][drop_type] += 1
                results.append(pred)

    return results


# ---------------------------------------------------------------------------
# Terminal seeds
# ---------------------------------------------------------------------------

def _seed_terminals() -> List[State]:
    """Generate simple terminal positions without any graph search.

    Lion-capture terminals
        One player's lion is absent; the other is on the board; no other pieces.
    Try-win terminals
        Both lions present, no other pieces; the previous mover's lion is on
        the try-rank and cannot be captured by the current mover.
    """
    seeds: List[State] = []

    def _make_board():
        return [[None] * BOARD_COLS for _ in range(BOARD_ROWS)]

    # -- Lion-capture terminals --
    for lion_sq in range(BOARD_ROWS * BOARD_COLS):
        lr, lc = divmod(lion_sq, BOARD_COLS)
        for survivor, loser in ((Player.BLACK, Player.WHITE),
                                (Player.WHITE, Player.BLACK)):
            board = _make_board()
            board[lr][lc] = Piece(survivor, PieceType.LION)
            seeds.append(
                State(
                    board=board,
                    turn=loser,       # loser is "to move" in a terminal state
                    hands={
                        Player.BLACK: State.empty_hand(),
                        Player.WHITE: State.empty_hand(),
                    },
                )
            )

    # -- Try-win terminals (both lions, no other pieces) --
    for bl_sq in range(BOARD_ROWS * BOARD_COLS):
        bl_r, bl_c = divmod(bl_sq, BOARD_COLS)
        for wl_sq in range(BOARD_ROWS * BOARD_COLS):
            if wl_sq == bl_sq:
                continue
            wl_r, wl_c = divmod(wl_sq, BOARD_COLS)

            board = _make_board()
            board[bl_r][bl_c] = Piece(Player.BLACK, PieceType.LION)
            board[wl_r][wl_c] = Piece(Player.WHITE, PieceType.LION)
            hands = {Player.BLACK: State.empty_hand(), Player.WHITE: State.empty_hand()}

            # Try for BLACK (BL at _last_rank(WHITE)), turn=WHITE
            if bl_r == _last_rank(Player.WHITE):
                s = State(board=[r[:] for r in board], turn=Player.WHITE, hands={
                    Player.BLACK: dict(hands[Player.BLACK]),
                    Player.WHITE: dict(hands[Player.WHITE]),
                })
                if is_terminal(s):
                    seeds.append(s)

            # Try for WHITE (WL at _last_rank(BLACK)), turn=BLACK
            if wl_r == _last_rank(Player.BLACK):
                s = State(board=[r[:] for r in board], turn=Player.BLACK, hands={
                    Player.BLACK: dict(hands[Player.BLACK]),
                    Player.WHITE: dict(hands[Player.WHITE]),
                })
                if is_terminal(s):
                    seeds.append(s)

    return seeds


# ---------------------------------------------------------------------------
# Retrograde analysis
# ---------------------------------------------------------------------------

def retrograde_analysis() -> Dict[int, int]:
    """Compute WIN/LOSS/DRAW for all positions reachable backward from terminals.

    Returns
    -------
    dict mapping ``position_to_index(state)`` to ``1`` (WIN), ``-1`` (LOSS),
    or absent (DRAW) for every discovered position.
    """
    eval_table: Dict[int, int] = {}
    remaining: Dict[int, int] = {}
    queue: deque = deque()

    # Seed with all simple terminal positions.
    seen_seeds: set = set()
    for term in _seed_terminals():
        idx = position_to_index(term)
        if idx not in seen_seeds and idx not in eval_table:
            seen_seeds.add(idx)
            eval_table[idx] = terminal_value(term)
            queue.append(term)

    # Backward BFS.
    processed: set = set()
    while queue:
        P_prime = queue.popleft()
        P_prime_idx = position_to_index(P_prime)
        if P_prime_idx in processed:
            continue
        processed.add(P_prime_idx)
        P_prime_val = eval_table[P_prime_idx]

        seen_in_iter: set = set()
        for P in _predecessor_states(P_prime):
            P_idx = position_to_index(P)
            if P_idx in seen_in_iter:
                continue
            seen_in_iter.add(P_idx)

            if P_idx in eval_table:
                continue

            # If P is itself terminal, evaluate immediately.
            if is_terminal(P):
                eval_table[P_idx] = terminal_value(P)
                queue.append(P)
                continue

            first_encounter = P_idx not in remaining
            if first_encounter:
                # Compute total successors and how many are already WIN.
                total = 0
                win_count = 0
                has_loss = False
                for move in P.legal_moves():
                    S = P.apply_move(move)
                    S_idx = position_to_index(S)
                    total += 1
                    if S_idx in eval_table:
                        v = eval_table[S_idx]
                        if v == -1:
                            has_loss = True
                        elif v == 1:
                            win_count += 1

                if total == 0:
                    eval_table[P_idx] = -1
                    queue.append(P)
                    continue

                if has_loss:
                    eval_table[P_idx] = 1
                    queue.append(P)
                    continue

                remaining[P_idx] = total - win_count
                if remaining[P_idx] == 0:
                    eval_table[P_idx] = -1
                    queue.append(P)
                # P_prime's contribution already counted above; skip double-decrement.
                continue

            # P was encountered before; P_prime is a new evaluated successor.
            if P_prime_val == -1:   # new LOSS successor → P is WIN
                eval_table[P_idx] = 1
                remaining.pop(P_idx, None)
                queue.append(P)
            elif P_prime_val == 1:  # new WIN successor → decrement
                remaining[P_idx] -= 1
                if remaining[P_idx] == 0:
                    eval_table[P_idx] = -1
                    remaining.pop(P_idx)
                    queue.append(P)

    return eval_table
