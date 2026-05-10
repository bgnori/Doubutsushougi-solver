from __future__ import annotations

from functools import lru_cache
from math import comb
from typing import Dict, List, Optional, Tuple

from .game import BOARD_COLS, BOARD_ROWS, Piece, PieceType, Player, State

_BOARD_SIZE = BOARD_ROWS * BOARD_COLS
_LION_PAIR_COUNT = _BOARD_SIZE * (_BOARD_SIZE - 1)

_BLACK_CHICK = 0
_WHITE_CHICK = 1
_BLACK_HEN = 2
_WHITE_HEN = 3


# ---------------------------------------------------------------------------
# Dense perfect index over material-complete raw placements
# ---------------------------------------------------------------------------


def position_key(state: State) -> int:
    """Return a dense, collision-free index for a material-complete position.

    The indexed space is:
        * canonical black-to-move orientation,
    * one black lion and one white lion on distinct board squares,
    * two giraffes, two elephants, and two chick-family pieces distributed
      between board and hands,
    * no overlapping board pieces.

    Move-legality constraints such as check, double-check, and simultaneous
    try-state are deliberately not filtered here; retrograde analysis classifies
    those states later. ``State.winner`` is analysis metadata and is ignored.
    """
    state = _canonical_state(state)

    black_lion, white_lion = _lion_positions(state)
    lion_rank = _rank_lions(black_lion, white_lion)
    available = _available_cells(black_lion, white_lion)

    g_cells = _piece_cells(state, PieceType.GIRAFFE)
    e_cells = _piece_cells(state, PieceType.ELEPHANT)
    c_cells = _chick_family_cells(state)

    h_g = _hand_total(state, PieceType.GIRAFFE)
    h_e = _hand_total(state, PieceType.ELEPHANT)
    h_c = _hand_total(state, PieceType.CHICK)
    _validate_hand_count(h_g, "giraffe")
    _validate_hand_count(h_e, "elephant")
    _validate_hand_count(h_c, "chick-family")

    b_g = 2 - h_g
    b_e = 2 - h_e
    b_c = 2 - h_c
    if len(g_cells) != b_g:
        raise ValueError(f"Found {len(g_cells)} giraffes on board, expected {b_g}")
    if len(e_cells) != b_e:
        raise ValueError(f"Found {len(e_cells)} elephants on board, expected {b_e}")
    if len(c_cells) != b_c:
        raise ValueError(f"Found {len(c_cells)} chick-family pieces on board, expected {b_c}")

    hand_offset, block_size = _hand_block_offset_and_size(h_g, h_e, h_c)

    g_indices = _cell_indices_in(g_cells, available)
    g_cell_rank = _rank_combination(g_indices, len(available), b_g)
    after_g = _remove_cells(available, g_cells)

    e_indices = _cell_indices_in(e_cells, after_g)
    e_cell_rank = _rank_combination(e_indices, len(after_g), b_e)
    after_e = _remove_cells(after_g, e_cells)

    c_indices = _cell_indices_in(c_cells, after_e)
    c_cell_rank = _rank_combination(c_indices, len(after_e), b_c)

    g_owner_rank = _non_promoted_ownership_rank(state, PieceType.GIRAFFE, g_cells, h_g)
    e_owner_rank = _non_promoted_ownership_rank(state, PieceType.ELEPHANT, e_cells, h_e)
    c_owner_rank = _chick_family_ownership_rank(state, c_cells, h_c)

    e_cell_count = comb(10 - b_g, b_e)
    c_cell_count = comb(10 - b_g - b_e, b_c)
    g_owner_count = (h_g + 1) * (1 << b_g)
    e_owner_count = (h_e + 1) * (1 << b_e)
    c_owner_count = (h_c + 1) * (4 ** b_c)

    inner = g_cell_rank
    inner = inner * e_cell_count + e_cell_rank
    inner = inner * c_cell_count + c_cell_rank
    inner = inner * g_owner_count + g_owner_rank
    inner = inner * e_owner_count + e_owner_rank
    inner = inner * c_owner_count + c_owner_rank

    if inner >= block_size:
        raise ValueError("Internal hash rank exceeded hand-count block size")

    return lion_rank * _states_per_lion_pair() + hand_offset + inner


# Alias for retrograde analysis.
position_to_index = position_key


@lru_cache(maxsize=1)
def position_key_space_size() -> int:
    """Return the dense index-space size of ``position_key``."""
    return _states_per_turn()


def state_from_key(key: int) -> State:
    """Reconstruct the canonical black-to-move State represented by ``key``."""
    if key < 0 or key >= position_key_space_size():
        raise ValueError(f"Key out of range: {key}")

    remainder = key
    turn = Player.BLACK

    lion_rank, remainder = divmod(remainder, _states_per_lion_pair())
    black_lion, white_lion = _unrank_lions(lion_rank)
    available = _available_cells(black_lion, white_lion)

    h_g, h_e, h_c, remainder = _unrank_hand_block(remainder)
    b_g = 2 - h_g
    b_e = 2 - h_e
    b_c = 2 - h_c

    e_cell_count = comb(10 - b_g, b_e)
    c_cell_count = comb(10 - b_g - b_e, b_c)
    g_owner_count = (h_g + 1) * (1 << b_g)
    e_owner_count = (h_e + 1) * (1 << b_e)
    c_owner_count = (h_c + 1) * (4 ** b_c)

    remainder, c_owner_rank = divmod(remainder, c_owner_count)
    remainder, e_owner_rank = divmod(remainder, e_owner_count)
    remainder, g_owner_rank = divmod(remainder, g_owner_count)
    remainder, c_cell_rank = divmod(remainder, c_cell_count)
    g_cell_rank, e_cell_rank = divmod(remainder, e_cell_count)

    g_indices = _unrank_combination(g_cell_rank, len(available), b_g)
    g_cells = tuple(available[i] for i in g_indices)
    after_g = _remove_cells(available, g_cells)

    e_indices = _unrank_combination(e_cell_rank, len(after_g), b_e)
    e_cells = tuple(after_g[i] for i in e_indices)
    after_e = _remove_cells(after_g, e_cells)

    c_indices = _unrank_combination(c_cell_rank, len(after_e), b_c)
    c_cells = tuple(after_e[i] for i in c_indices)

    return _build_state(
        turn=turn,
        black_lion=black_lion,
        white_lion=white_lion,
        g_cells=g_cells,
        e_cells=e_cells,
        c_cells=c_cells,
        h_g=h_g,
        h_e=h_e,
        h_c=h_c,
        g_owner_rank=g_owner_rank,
        e_owner_rank=e_owner_rank,
        c_owner_rank=c_owner_rank,
    )


# ---------------------------------------------------------------------------
# Canonicalization and space-size blocks
# ---------------------------------------------------------------------------


def _canonical_state(state: State) -> State:
    if state.turn == Player.BLACK:
        return state

    board: List[List[Optional[Piece]]] = [[None] * BOARD_COLS for _ in range(BOARD_ROWS)]
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            piece = state.board[row][col]
            if piece is None:
                continue
            new_row = BOARD_ROWS - 1 - row
            new_col = BOARD_COLS - 1 - col
            board[new_row][new_col] = Piece(piece.owner.opponent, piece.type)

    return State(
        board=board,
        turn=Player.BLACK,
        hands={
            Player.BLACK: dict(state.hands[Player.WHITE]),
            Player.WHITE: dict(state.hands[Player.BLACK]),
        },
        winner=None if state.winner is None else state.winner.opponent,
    )


@lru_cache(maxsize=1)
def _states_per_turn() -> int:
    return _LION_PAIR_COUNT * _states_per_lion_pair()


@lru_cache(maxsize=1)
def _states_per_lion_pair() -> int:
    return sum(size for _, _, _, _, size in _hand_blocks())


@lru_cache(maxsize=1)
def _hand_blocks() -> Tuple[Tuple[int, int, int, int, int], ...]:
    blocks: List[Tuple[int, int, int, int, int]] = []
    offset = 0
    for h_g in range(3):
        for h_e in range(3):
            for h_c in range(3):
                size = _hand_block_size(h_g, h_e, h_c)
                blocks.append((h_g, h_e, h_c, offset, size))
                offset += size
    return tuple(blocks)


@lru_cache(maxsize=None)
def _hand_block_size(h_g: int, h_e: int, h_c: int) -> int:
    b_g = 2 - h_g
    b_e = 2 - h_e
    b_c = 2 - h_c
    return (
        comb(10, b_g)
        * comb(10 - b_g, b_e)
        * comb(10 - b_g - b_e, b_c)
        * (h_g + 1)
        * (1 << b_g)
        * (h_e + 1)
        * (1 << b_e)
        * (h_c + 1)
        * (4 ** b_c)
    )


def _hand_block_offset_and_size(h_g: int, h_e: int, h_c: int) -> Tuple[int, int]:
    for block_h_g, block_h_e, block_h_c, offset, size in _hand_blocks():
        if (block_h_g, block_h_e, block_h_c) == (h_g, h_e, h_c):
            return offset, size
    raise ValueError(f"Invalid hand-count block: {(h_g, h_e, h_c)}")


def _unrank_hand_block(rank: int) -> Tuple[int, int, int, int]:
    for h_g, h_e, h_c, _offset, size in _hand_blocks():
        if rank < size:
            return h_g, h_e, h_c, rank
        rank -= size
    raise ValueError("Hand-count block rank out of range")


# ---------------------------------------------------------------------------
# Lions and board-cell combinations
# ---------------------------------------------------------------------------


def _rank_lions(black_lion: int, white_lion: int) -> int:
    if not (0 <= black_lion < _BOARD_SIZE) or not (0 <= white_lion < _BOARD_SIZE):
        raise ValueError(f"Invalid lion placement: black={black_lion}, white={white_lion}")
    if black_lion == white_lion:
        raise ValueError(f"Overlapping lions at cell {black_lion}")
    white_rank = white_lion if white_lion < black_lion else white_lion - 1
    return black_lion * (_BOARD_SIZE - 1) + white_rank


def _unrank_lions(rank: int) -> Tuple[int, int]:
    black_lion, white_rank = divmod(rank, _BOARD_SIZE - 1)
    white_lion = white_rank if white_rank < black_lion else white_rank + 1
    return black_lion, white_lion


def _available_cells(black_lion: int, white_lion: int) -> Tuple[int, ...]:
    return tuple(cell for cell in range(_BOARD_SIZE) if cell not in (black_lion, white_lion))


def _rank_combination(indices: Tuple[int, ...], n: int, k: int) -> int:
    if len(indices) != k:
        raise ValueError(f"Expected {k} combination indices, found {len(indices)}")
    if tuple(sorted(indices)) != indices or len(set(indices)) != len(indices):
        raise ValueError(f"Combination indices must be strictly increasing: {indices}")
    if any(index < 0 or index >= n for index in indices):
        raise ValueError(f"Combination indices out of range for n={n}: {indices}")

    rank = 0
    previous = -1
    for i, index in enumerate(indices):
        for skipped in range(previous + 1, index):
            rank += comb(n - skipped - 1, k - i - 1)
        previous = index
    return rank


def _unrank_combination(rank: int, n: int, k: int) -> Tuple[int, ...]:
    if k == 0:
        if rank != 0:
            raise ValueError("Rank out of range for empty combination")
        return ()

    result: List[int] = []
    start = 0
    remaining = k
    while remaining:
        for value in range(start, n):
            count = comb(n - value - 1, remaining - 1)
            if rank < count:
                result.append(value)
                start = value + 1
                remaining -= 1
                break
            rank -= count
        else:
            raise ValueError("Combination rank out of range")
    return tuple(result)


def _cell_indices_in(cells: Tuple[int, ...], universe: Tuple[int, ...]) -> Tuple[int, ...]:
    index_by_cell = {cell: index for index, cell in enumerate(universe)}
    try:
        return tuple(index_by_cell[cell] for cell in cells)
    except KeyError as exc:
        raise ValueError(f"Cell {exc.args[0]} is already occupied") from exc


def _remove_cells(cells: Tuple[int, ...], removed: Tuple[int, ...]) -> Tuple[int, ...]:
    removed_set = set(removed)
    return tuple(cell for cell in cells if cell not in removed_set)


# ---------------------------------------------------------------------------
# State extraction
# ---------------------------------------------------------------------------


def _cell_index(row: int, col: int) -> int:
    return row * BOARD_COLS + col


def _lion_positions(state: State) -> Tuple[int, int]:
    black_lions: List[int] = []
    white_lions: List[int] = []
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            piece = state.board[row][col]
            if piece is None or piece.type != PieceType.LION:
                continue
            index = _cell_index(row, col)
            if piece.owner == Player.BLACK:
                black_lions.append(index)
            else:
                white_lions.append(index)

    if len(black_lions) != 1 or len(white_lions) != 1:
        raise ValueError("A position must contain exactly one black lion and one white lion on the board")
    return black_lions[0], white_lions[0]


def _piece_cells(state: State, piece_type: PieceType) -> Tuple[int, ...]:
    cells = []
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            piece = state.board[row][col]
            if piece is not None and piece.type == piece_type:
                cells.append(_cell_index(row, col))
    return tuple(cells)


def _chick_family_cells(state: State) -> Tuple[int, ...]:
    cells = []
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            piece = state.board[row][col]
            if piece is not None and piece.type in (PieceType.CHICK, PieceType.HEN):
                cells.append(_cell_index(row, col))
    return tuple(cells)


def _hand_count(state: State, owner: Player, piece_type: PieceType) -> int:
    return state.hands.get(owner, {}).get(piece_type, 0)


def _hand_total(state: State, piece_type: PieceType) -> int:
    black_count = _hand_count(state, Player.BLACK, piece_type)
    white_count = _hand_count(state, Player.WHITE, piece_type)
    if black_count < 0 or white_count < 0:
        raise ValueError(f"Negative hand count for {piece_type.value}")
    return black_count + white_count


def _validate_hand_count(count: int, family_name: str) -> None:
    if count < 0 or count > 2:
        raise ValueError(f"Invalid {family_name} hand count: {count}")


def _non_promoted_ownership_rank(
    state: State,
    piece_type: PieceType,
    cells: Tuple[int, ...],
    hand_total: int,
) -> int:
    black_hand = _hand_count(state, Player.BLACK, piece_type)
    white_hand = _hand_count(state, Player.WHITE, piece_type)
    if black_hand + white_hand != hand_total:
        raise ValueError(f"Invalid {piece_type.value} hand count")

    bits = 0
    for i, cell in enumerate(cells):
        row, col = divmod(cell, BOARD_COLS)
        piece = state.board[row][col]
        if piece is None or piece.type != piece_type:
            raise ValueError(f"Expected {piece_type.value} at cell {cell}")
        if piece.owner == Player.WHITE:
            bits |= 1 << i
    return black_hand * (1 << len(cells)) + bits


def _chick_family_ownership_rank(state: State, cells: Tuple[int, ...], hand_total: int) -> int:
    black_hand = _hand_count(state, Player.BLACK, PieceType.CHICK)
    white_hand = _hand_count(state, Player.WHITE, PieceType.CHICK)
    if black_hand + white_hand != hand_total:
        raise ValueError("Invalid chick-family hand count")

    digits = 0
    multiplier = 1
    for cell in cells:
        row, col = divmod(cell, BOARD_COLS)
        piece = state.board[row][col]
        if piece is None or piece.type not in (PieceType.CHICK, PieceType.HEN):
            raise ValueError(f"Expected chick-family piece at cell {cell}")
        digits += _chick_family_digit(piece) * multiplier
        multiplier *= 4
    return black_hand * (4 ** len(cells)) + digits


def _chick_family_digit(piece: Piece) -> int:
    if piece.type == PieceType.CHICK:
        return _BLACK_CHICK if piece.owner == Player.BLACK else _WHITE_CHICK
    if piece.type == PieceType.HEN:
        return _BLACK_HEN if piece.owner == Player.BLACK else _WHITE_HEN
    raise ValueError(f"Not a chick-family piece: {piece}")


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


def _build_state(
    turn: Player,
    black_lion: int,
    white_lion: int,
    g_cells: Tuple[int, ...],
    e_cells: Tuple[int, ...],
    c_cells: Tuple[int, ...],
    h_g: int,
    h_e: int,
    h_c: int,
    g_owner_rank: int,
    e_owner_rank: int,
    c_owner_rank: int,
) -> State:
    board: List[List[Optional[Piece]]] = [[None] * BOARD_COLS for _ in range(BOARD_ROWS)]
    hands: Dict[Player, Dict[PieceType, int]] = {
        Player.BLACK: State.empty_hand(),
        Player.WHITE: State.empty_hand(),
    }

    def place(cell: int, piece: Piece) -> None:
        row, col = divmod(cell, BOARD_COLS)
        if board[row][col] is not None:
            raise ValueError(f"Decoded overlapping pieces at cell {cell}")
        board[row][col] = piece

    place(black_lion, Piece(Player.BLACK, PieceType.LION))
    place(white_lion, Piece(Player.WHITE, PieceType.LION))
    _place_non_promoted_family(place, hands, PieceType.GIRAFFE, g_cells, h_g, g_owner_rank)
    _place_non_promoted_family(place, hands, PieceType.ELEPHANT, e_cells, h_e, e_owner_rank)
    _place_chick_family(place, hands, c_cells, h_c, c_owner_rank)

    return State(board=board, turn=turn, hands=hands, winner=None)


def _place_non_promoted_family(
    place,
    hands: Dict[Player, Dict[PieceType, int]],
    piece_type: PieceType,
    cells: Tuple[int, ...],
    hand_total: int,
    owner_rank: int,
) -> None:
    board_owner_count = 1 << len(cells)
    black_hand, owner_bits = divmod(owner_rank, board_owner_count)
    white_hand = hand_total - black_hand
    if black_hand < 0 or white_hand < 0:
        raise ValueError(f"Invalid decoded hand count for {piece_type.value}")
    hands[Player.BLACK][piece_type] = black_hand
    hands[Player.WHITE][piece_type] = white_hand

    for i, cell in enumerate(cells):
        owner = Player.WHITE if owner_bits & (1 << i) else Player.BLACK
        place(cell, Piece(owner, piece_type))


def _place_chick_family(
    place,
    hands: Dict[Player, Dict[PieceType, int]],
    cells: Tuple[int, ...],
    hand_total: int,
    owner_rank: int,
) -> None:
    board_state_count = 4 ** len(cells)
    black_hand, board_digits = divmod(owner_rank, board_state_count)
    white_hand = hand_total - black_hand
    if black_hand < 0 or white_hand < 0:
        raise ValueError("Invalid decoded chick-family hand count")
    hands[Player.BLACK][PieceType.CHICK] = black_hand
    hands[Player.WHITE][PieceType.CHICK] = white_hand

    for cell in cells:
        board_digits, digit = divmod(board_digits, 4)
        place(cell, _piece_from_chick_family_digit(digit))


def _piece_from_chick_family_digit(digit: int) -> Piece:
    if digit == _BLACK_CHICK:
        return Piece(Player.BLACK, PieceType.CHICK)
    if digit == _WHITE_CHICK:
        return Piece(Player.WHITE, PieceType.CHICK)
    if digit == _BLACK_HEN:
        return Piece(Player.BLACK, PieceType.HEN)
    if digit == _WHITE_HEN:
        return Piece(Player.WHITE, PieceType.HEN)
    raise ValueError(f"Invalid chick-family digit: {digit}")
