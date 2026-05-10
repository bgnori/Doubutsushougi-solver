from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from .game import BOARD_COLS, BOARD_ROWS, PieceType, Player, State

_NONE_WINNER = 0
_BLACK_WINNER = 1
_WHITE_WINNER = 2
_WINNER_CODES = {
    None: _NONE_WINNER,
    Player.BLACK: _BLACK_WINNER,
    Player.WHITE: _WHITE_WINNER,
}

_BLACK_HAND = -2
_WHITE_HAND = -1

_BLACK_CHICK_OFFSET = 0
_WHITE_CHICK_OFFSET = 12
_BLACK_HEN_OFFSET = 24
_WHITE_HEN_OFFSET = 36

_BOARD_SIZE = BOARD_ROWS * BOARD_COLS
_ABSENT_LION = _BOARD_SIZE
_PAIR_TABLE_CACHE_SIZE = 512


def position_key(state: State) -> int:
    key = _winner_code(state.winner)
    key = key * 2 + (0 if state.turn == Player.BLACK else 1)

    black_lion, white_lion = _lion_positions(state)
    black_lion_code = _ABSENT_LION if black_lion is None else black_lion
    white_lion_code = _ABSENT_LION if white_lion is None else white_lion

    key = key * (_ABSENT_LION + 1) + black_lion_code
    white_lion_rank, white_lion_total = _rank_second_lion(black_lion_code, white_lion_code)
    key = key * white_lion_total + white_lion_rank

    occupied_mask = 0
    if black_lion is not None:
        occupied_mask |= 1 << black_lion
    if white_lion is not None:
        occupied_mask |= 1 << white_lion

    giraffe_rank, giraffe_total, giraffe_mask = _encode_non_promoted_pair(
        state=state,
        piece_type=PieceType.GIRAFFE,
        occupied_mask=occupied_mask,
    )
    key = key * giraffe_total + giraffe_rank
    occupied_mask |= giraffe_mask

    elephant_rank, elephant_total, elephant_mask = _encode_non_promoted_pair(
        state=state,
        piece_type=PieceType.ELEPHANT,
        occupied_mask=occupied_mask,
    )
    key = key * elephant_total + elephant_rank
    occupied_mask |= elephant_mask

    chick_rank, chick_total, _ = _encode_chick_pair(state=state, occupied_mask=occupied_mask)
    key = key * chick_total + chick_rank
    return key


def _winner_code(winner: Optional[Player]) -> int:
    return _WINNER_CODES[winner]


def _cell_index(row: int, col: int) -> int:
    return row * BOARD_COLS + col


def _lion_positions(state: State) -> Tuple[Optional[int], Optional[int]]:
    black_lion = None
    white_lion = None
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            piece = state.board[row][col]
            if piece is None or piece.type != PieceType.LION:
                continue
            index = _cell_index(row, col)
            if piece.owner == Player.BLACK:
                black_lion = index
            else:
                white_lion = index
    return black_lion, white_lion


def _rank_second_lion(first_lion_code: int, second_lion_code: int) -> Tuple[int, int]:
    first_lion_occupies_square = first_lion_code != _ABSENT_LION
    total = (_ABSENT_LION + 1) - (1 if first_lion_occupies_square else 0)
    rank = 0
    for candidate in range(_ABSENT_LION + 1):
        if candidate != _ABSENT_LION and candidate == first_lion_code:
            continue
        if candidate == second_lion_code:
            return rank, total
        rank += 1
    raise ValueError(f"Invalid lion placement: black={first_lion_code}, white={second_lion_code}")


def _encode_non_promoted_pair(state: State, piece_type: PieceType, occupied_mask: int) -> Tuple[int, int, int]:
    placements = _non_promoted_pair(state, piece_type)
    table, total = _non_promoted_pair_table(occupied_mask)
    rank = table.get(placements)
    if rank is None:
        raise ValueError(f"Invalid {piece_type.value} placement")
    return rank, total, _pair_board_mask(placements)


def _encode_chick_pair(state: State, occupied_mask: int) -> Tuple[int, int, int]:
    placements = _chick_pair(state)
    table, total = _chick_pair_table(occupied_mask)
    rank = table.get(placements)
    if rank is None:
        raise ValueError("Invalid chick placement")
    return rank, total, _pair_board_mask(placements)


def _non_promoted_pair(state: State, piece_type: PieceType) -> Tuple[int, ...]:
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

    if len(placements) > 2:
        raise ValueError(f"Found {len(placements)} {piece_type.value} pieces, expected at most 2")
    placements.sort()
    return tuple(placements)


def _chick_pair(state: State) -> Tuple[int, ...]:
    placements: List[int] = []

    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            piece = state.board[row][col]
            if piece is None:
                continue
            index = _cell_index(row, col)
            if piece.type == PieceType.CHICK:
                placements.append(
                    _BLACK_CHICK_OFFSET + index if piece.owner == Player.BLACK else _WHITE_CHICK_OFFSET + index
                )
            elif piece.type == PieceType.HEN:
                placements.append(_BLACK_HEN_OFFSET + index if piece.owner == Player.BLACK else _WHITE_HEN_OFFSET + index)

    for _ in range(state.hands[Player.BLACK][PieceType.CHICK]):
        placements.append(_BLACK_HAND)
    for _ in range(state.hands[Player.WHITE][PieceType.CHICK]):
        placements.append(_WHITE_HAND)

    if len(placements) > 2:
        raise ValueError(f"Found {len(placements)} chick-family pieces, expected at most 2")
    placements.sort()
    return tuple(placements)


def _pair_board_mask(pair: Tuple[int, ...]) -> int:
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
def _non_promoted_pair_table(occupied_mask: int) -> Tuple[Dict[Tuple[int, ...], int], int]:
    options = [_BLACK_HAND, _WHITE_HAND]
    for cell in range(_BOARD_SIZE):
        if occupied_mask & (1 << cell):
            continue
        options.append(cell)
        options.append(_BOARD_SIZE + cell)
    options.sort()

    rank_map: Dict[Tuple[int, ...], int] = {(): 0}
    rank = 1
    for option in options:
        rank_map[(option,)] = rank
        rank += 1

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


@lru_cache(maxsize=_PAIR_TABLE_CACHE_SIZE)
def _chick_pair_table(occupied_mask: int) -> Tuple[Dict[Tuple[int, ...], int], int]:
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

    rank_map: Dict[Tuple[int, ...], int] = {(): 0}
    rank = 1
    for option in options:
        rank_map[(option,)] = rank
        rank += 1

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
