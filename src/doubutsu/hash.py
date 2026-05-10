from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from .game import BOARD_COLS, BOARD_ROWS, Piece, PieceType, Player, State

_BLACK_HAND = -2
_WHITE_HAND = -1

_BLACK_CHICK_OFFSET = 0
_WHITE_CHICK_OFFSET = 12
_BLACK_HEN_OFFSET = 24
_WHITE_HEN_OFFSET = 36

_BOARD_SIZE = BOARD_ROWS * BOARD_COLS
_PAIR_TABLE_CACHE_SIZE = None  # at most 4096 distinct occupied masks


# ---------------------------------------------------------------------------
# Dense perfect index over material-complete positions
# ---------------------------------------------------------------------------


def position_key(state: State) -> int:
    """Return a dense, collision-free index for a material-complete position.

    This hash enumerates raw piece-placement positions, not legal game states.
    It intentionally includes positions such as double-check positions and
    simultaneous try-state positions; those are classified later by retrograde
    analysis.

    The index excludes states that cannot be represented by a board+hand
    placement of the physical Doubutsushougi pieces:
    * two pieces on the same board square,
    * a lion in hand,
    * a missing lion,
    * missing or extra non-lion pieces.

    ``State.winner`` is deliberately ignored.  Win/loss/draw is analysis data,
    not part of the position index.
    """
    key = _turn_code(state.turn) * _states_per_turn()

    black_lion, white_lion = _lion_positions(state)
    occupied_mask = _lion_occupied_mask(black_lion, white_lion)

    for bl, wl in _lion_pair_options():
        count = _giraffe_prefix_count(_lion_occupied_mask(bl, wl))
        if bl == black_lion and wl == white_lion:
            break
        key += count
    else:
        raise ValueError(f"Invalid lion placement: black={black_lion}, white={white_lion}")

    giraffe_placement = _non_promoted_pair(state, PieceType.GIRAFFE)
    giraffe_table, _ = _non_promoted_pair_table(occupied_mask)
    giraffe_rank = giraffe_table.get(giraffe_placement)
    if giraffe_rank is None:
        raise ValueError("Invalid giraffe placement")

    for earlier_giraffe in _non_promoted_pair_options(occupied_mask)[:giraffe_rank]:
        key += _elephant_prefix_count(occupied_mask | _pair_board_mask(earlier_giraffe))
    occupied_mask |= _pair_board_mask(giraffe_placement)

    elephant_placement = _non_promoted_pair(state, PieceType.ELEPHANT)
    elephant_table, _ = _non_promoted_pair_table(occupied_mask)
    elephant_rank = elephant_table.get(elephant_placement)
    if elephant_rank is None:
        raise ValueError("Invalid elephant placement")

    for earlier_elephant in _non_promoted_pair_options(occupied_mask)[:elephant_rank]:
        _, chick_total = _chick_pair_table(occupied_mask | _pair_board_mask(earlier_elephant))
        key += chick_total
    occupied_mask |= _pair_board_mask(elephant_placement)

    chick_placement = _chick_pair(state)
    chick_table, _ = _chick_pair_table(occupied_mask)
    chick_rank = chick_table.get(chick_placement)
    if chick_rank is None:
        raise ValueError("Invalid chick placement")

    return key + chick_rank


# Alias for retrograde analysis.
position_to_index = position_key


@lru_cache(maxsize=1)
def position_key_space_size() -> int:
    """Return the dense index-space size of ``position_key``."""
    return 2 * _states_per_turn()


def state_from_key(key: int) -> State:
    """Reconstruct the unique material-complete State represented by ``key``."""
    if key < 0 or key >= position_key_space_size():
        raise ValueError(f"Key out of range: {key}")

    turn_code, remainder = divmod(key, _states_per_turn())
    turn = Player.BLACK if turn_code == 0 else Player.WHITE

    black_lion, white_lion, occupied_mask, remainder = _unrank_lions(remainder)
    giraffe_placement, occupied_mask, remainder = _unrank_non_promoted_pair(remainder, occupied_mask)
    elephant_placement, occupied_mask, remainder = _unrank_elephant_pair(remainder, occupied_mask)
    chick_placement = _chick_pair_options(occupied_mask)[remainder]

    return _build_state(
        turn=turn,
        black_lion=black_lion,
        white_lion=white_lion,
        giraffe_placement=giraffe_placement,
        elephant_placement=elephant_placement,
        chick_placement=chick_placement,
    )


# ---------------------------------------------------------------------------
# Prefix counts and unranking
# ---------------------------------------------------------------------------


def _turn_code(turn: Player) -> int:
    return 0 if turn == Player.BLACK else 1


@lru_cache(maxsize=1)
def _states_per_turn() -> int:
    return sum(_giraffe_prefix_count(_lion_occupied_mask(bl, wl)) for bl, wl in _lion_pair_options())


@lru_cache(maxsize=1)
def _lion_pair_options() -> Tuple[Tuple[int, int], ...]:
    return tuple((black_lion, white_lion) for black_lion in range(_BOARD_SIZE) for white_lion in range(_BOARD_SIZE) if white_lion != black_lion)


def _lion_occupied_mask(black_lion: int, white_lion: int) -> int:
    return (1 << black_lion) | (1 << white_lion)


@lru_cache(maxsize=None)
def _giraffe_prefix_count(occupied_mask: int) -> int:
    return sum(
        _elephant_prefix_count(occupied_mask | _pair_board_mask(giraffe_placement))
        for giraffe_placement in _non_promoted_pair_options(occupied_mask)
    )


@lru_cache(maxsize=None)
def _elephant_prefix_count(occupied_mask: int) -> int:
    total = 0
    for elephant_placement in _non_promoted_pair_options(occupied_mask):
        _, chick_total = _chick_pair_table(occupied_mask | _pair_board_mask(elephant_placement))
        total += chick_total
    return total


def _unrank_lions(remainder: int) -> Tuple[int, int, int, int]:
    for black_lion, white_lion in _lion_pair_options():
        occupied_mask = _lion_occupied_mask(black_lion, white_lion)
        count = _giraffe_prefix_count(occupied_mask)
        if remainder < count:
            return black_lion, white_lion, occupied_mask, remainder
        remainder -= count
    raise ValueError("Lion rank out of range")


def _unrank_non_promoted_pair(remainder: int, occupied_mask: int) -> Tuple[Tuple[int, int], int, int]:
    for placement in _non_promoted_pair_options(occupied_mask):
        next_occupied_mask = occupied_mask | _pair_board_mask(placement)
        count = _elephant_prefix_count(next_occupied_mask)
        if remainder < count:
            return placement, next_occupied_mask, remainder
        remainder -= count
    raise ValueError("Non-promoted pair rank out of range")


def _unrank_elephant_pair(remainder: int, occupied_mask: int) -> Tuple[Tuple[int, int], int, int]:
    for placement in _non_promoted_pair_options(occupied_mask):
        next_occupied_mask = occupied_mask | _pair_board_mask(placement)
        _, chick_total = _chick_pair_table(next_occupied_mask)
        if remainder < chick_total:
            return placement, next_occupied_mask, remainder
        remainder -= chick_total
    raise ValueError("Elephant pair rank out of range")


# ---------------------------------------------------------------------------
# Piece extraction and pair tables
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


def _non_promoted_pair(state: State, piece_type: PieceType) -> Tuple[int, int]:
    placements: List[int] = []

    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            piece = state.board[row][col]
            if piece is None or piece.type != piece_type:
                continue
            index = _cell_index(row, col)
            placements.append(index if piece.owner == Player.BLACK else _BOARD_SIZE + index)

    for _ in range(state.hands[Player.BLACK][piece_type]):
        placements.append(_BLACK_HAND)
    for _ in range(state.hands[Player.WHITE][piece_type]):
        placements.append(_WHITE_HAND)

    if len(placements) != 2:
        raise ValueError(f"Found {len(placements)} {piece_type.value} pieces, expected exactly 2")
    placements.sort()
    return placements[0], placements[1]


def _chick_pair(state: State) -> Tuple[int, int]:
    placements: List[int] = []

    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            piece = state.board[row][col]
            if piece is None:
                continue
            index = _cell_index(row, col)
            if piece.type == PieceType.CHICK:
                placements.append(_BLACK_CHICK_OFFSET + index if piece.owner == Player.BLACK else _WHITE_CHICK_OFFSET + index)
            elif piece.type == PieceType.HEN:
                placements.append(_BLACK_HEN_OFFSET + index if piece.owner == Player.BLACK else _WHITE_HEN_OFFSET + index)

    for _ in range(state.hands[Player.BLACK][PieceType.CHICK]):
        placements.append(_BLACK_HAND)
    for _ in range(state.hands[Player.WHITE][PieceType.CHICK]):
        placements.append(_WHITE_HAND)

    if len(placements) != 2:
        raise ValueError(f"Found {len(placements)} chick-family pieces, expected exactly 2")
    placements.sort()
    return placements[0], placements[1]


@lru_cache(maxsize=None)
def _pair_board_mask(pair: Tuple[int, int]) -> int:
    mask = 0
    for placement in pair:
        cell = _placement_cell(placement)
        if cell is not None:
            mask |= 1 << cell
    return mask


def _placement_cell(placement: int) -> Optional[int]:
    if placement < 0:
        return None
    return placement % _BOARD_SIZE


def _valid_pair(first: int, second: int) -> bool:
    first_cell = _placement_cell(first)
    second_cell = _placement_cell(second)
    return first_cell is None or second_cell is None or first_cell != second_cell


@lru_cache(maxsize=_PAIR_TABLE_CACHE_SIZE)
def _non_promoted_pair_table(occupied_mask: int) -> Tuple[Dict[Tuple[int, int], int], int]:
    options = [_BLACK_HAND, _WHITE_HAND]
    for cell in range(_BOARD_SIZE):
        if occupied_mask & (1 << cell):
            continue
        options.append(cell)
        options.append(_BOARD_SIZE + cell)
    options.sort()
    return _exact_pair_table_from_options(options)


@lru_cache(maxsize=_PAIR_TABLE_CACHE_SIZE)
def _non_promoted_pair_options(occupied_mask: int) -> Tuple[Tuple[int, int], ...]:
    table, total = _non_promoted_pair_table(occupied_mask)
    options: List[Tuple[int, int]] = [(_BLACK_HAND, _BLACK_HAND) for _ in range(total)]
    for placement, rank in table.items():
        options[rank] = placement
    return tuple(options)


@lru_cache(maxsize=_PAIR_TABLE_CACHE_SIZE)
def _chick_pair_table(occupied_mask: int) -> Tuple[Dict[Tuple[int, int], int], int]:
    options = [_BLACK_HAND, _WHITE_HAND]
    for cell in range(_BOARD_SIZE):
        if occupied_mask & (1 << cell):
            continue
        options.extend(
            (
                _BLACK_CHICK_OFFSET + cell,
                _WHITE_CHICK_OFFSET + cell,
                _BLACK_HEN_OFFSET + cell,
                _WHITE_HEN_OFFSET + cell,
            )
        )
    options.sort()
    return _exact_pair_table_from_options(options)


@lru_cache(maxsize=_PAIR_TABLE_CACHE_SIZE)
def _chick_pair_options(occupied_mask: int) -> Tuple[Tuple[int, int], ...]:
    table, total = _chick_pair_table(occupied_mask)
    options: List[Tuple[int, int]] = [(_BLACK_HAND, _BLACK_HAND) for _ in range(total)]
    for placement, rank in table.items():
        options[rank] = placement
    return tuple(options)


def _exact_pair_table_from_options(options: List[int]) -> Tuple[Dict[Tuple[int, int], int], int]:
    rank_map: Dict[Tuple[int, int], int] = {}
    rank = 0

    for option in options:
        if _placement_cell(option) is not None:
            continue
        rank_map[(option, option)] = rank
        rank += 1

    for i, first in enumerate(options):
        for second in options[i + 1 :]:
            if not _valid_pair(first, second):
                continue
            rank_map[(first, second)] = rank
            rank += 1
    return rank_map, rank


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


def _decode_non_promoted_piece(placement_val: int) -> Tuple[Optional[int], Player]:
    if placement_val == _BLACK_HAND:
        return None, Player.BLACK
    if placement_val == _WHITE_HAND:
        return None, Player.WHITE
    if placement_val < _BOARD_SIZE:
        return placement_val, Player.BLACK
    return placement_val - _BOARD_SIZE, Player.WHITE


def _decode_chick_piece(placement_val: int) -> Tuple[Optional[int], Player, PieceType]:
    if placement_val == _BLACK_HAND:
        return None, Player.BLACK, PieceType.CHICK
    if placement_val == _WHITE_HAND:
        return None, Player.WHITE, PieceType.CHICK
    if _BLACK_CHICK_OFFSET <= placement_val < _BLACK_CHICK_OFFSET + _BOARD_SIZE:
        return placement_val - _BLACK_CHICK_OFFSET, Player.BLACK, PieceType.CHICK
    if _WHITE_CHICK_OFFSET <= placement_val < _WHITE_CHICK_OFFSET + _BOARD_SIZE:
        return placement_val - _WHITE_CHICK_OFFSET, Player.WHITE, PieceType.CHICK
    if _BLACK_HEN_OFFSET <= placement_val < _BLACK_HEN_OFFSET + _BOARD_SIZE:
        return placement_val - _BLACK_HEN_OFFSET, Player.BLACK, PieceType.HEN
    if _WHITE_HEN_OFFSET <= placement_val < _WHITE_HEN_OFFSET + _BOARD_SIZE:
        return placement_val - _WHITE_HEN_OFFSET, Player.WHITE, PieceType.HEN
    raise ValueError(f"Invalid chick placement value: {placement_val}")


def _build_state(
    turn: Player,
    black_lion: int,
    white_lion: int,
    giraffe_placement: Tuple[int, int],
    elephant_placement: Tuple[int, int],
    chick_placement: Tuple[int, int],
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

    for val in giraffe_placement:
        cell, owner = _decode_non_promoted_piece(val)
        if cell is None:
            hands[owner][PieceType.GIRAFFE] += 1
        else:
            place(cell, Piece(owner, PieceType.GIRAFFE))

    for val in elephant_placement:
        cell, owner = _decode_non_promoted_piece(val)
        if cell is None:
            hands[owner][PieceType.ELEPHANT] += 1
        else:
            place(cell, Piece(owner, PieceType.ELEPHANT))

    for val in chick_placement:
        cell, owner, piece_type = _decode_chick_piece(val)
        if cell is None:
            hands[owner][PieceType.CHICK] += 1
        else:
            place(cell, Piece(owner, piece_type))

    return State(board=board, turn=turn, hands=hands, winner=None)
