from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from .game import BOARD_COLS, BOARD_ROWS, Piece, PieceType, Player, State

_NONE_WINNER = 0
_BLACK_WINNER = 1
_WHITE_WINNER = 2
_WINNER_CODES = {
    None: _NONE_WINNER,
    Player.BLACK: _BLACK_WINNER,
    Player.WHITE: _WHITE_WINNER,
}
_WINNER_CODES_INV = {v: k for k, v in _WINNER_CODES.items()}

_BLACK_HAND = -2
_WHITE_HAND = -1

_BLACK_CHICK_OFFSET = 0
_WHITE_CHICK_OFFSET = 12
_BLACK_HEN_OFFSET = 24
_WHITE_HEN_OFFSET = 36

_BOARD_SIZE = BOARD_ROWS * BOARD_COLS
_ABSENT_LION = _BOARD_SIZE
_PAIR_TABLE_CACHE_SIZE = None  # unlimited: at most 4096 distinct occ values (12-bit board)


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

# Alias for retrograde analysis
position_to_index = position_key


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


@lru_cache(maxsize=None)
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


# ---------------------------------------------------------------------------
# Inverse: reconstruct a State from its position_key hash value
# ---------------------------------------------------------------------------


def state_from_key(target_key: int) -> State:
    """Reconstruct a State from the integer produced by ``position_key``."""
    bl_total = _ABSENT_LION + 1  # 13

    def _search(lion_pairs):
        for black_lion_code, white_lion_code, wl_rank, wl_total, occupied_mask in lion_pairs:
            giraffe_table, giraffe_total = _non_promoted_pair_table(occupied_mask)
            giraffe_items = [
                (gp, gr, _pair_board_mask(gp)) for gp, gr in giraffe_table.items()
            ]

            for giraffe_placement, giraffe_rank, giraffe_mask in giraffe_items:
                occ_after_g = occupied_mask | giraffe_mask

                elephant_table, elephant_total = _non_promoted_pair_table(occ_after_g)
                occ_to_chick: Dict[int, int] = {}
                elephant_items = []
                for ep, er in elephant_table.items():
                    occ_e = occ_after_g | _pair_board_mask(ep)
                    if occ_e not in occ_to_chick:
                        _, ct = _chick_pair_table(occ_e)
                        occ_to_chick[occ_e] = ct
                    elephant_items.append((ep, er, occ_e))

                for elephant_placement, elephant_rank, occ_after_e in elephant_items:
                    chick_total = occ_to_chick[occ_after_e]

                    c_rank = target_key % chick_total
                    key_after_c = target_key // chick_total

                    if key_after_c % elephant_total != elephant_rank:
                        continue
                    key_after_e = key_after_c // elephant_total

                    if key_after_e % giraffe_total != giraffe_rank:
                        continue
                    key_after_g = key_after_e // giraffe_total

                    if key_after_g % wl_total != wl_rank:
                        continue
                    key_after_wl = key_after_g // wl_total

                    if key_after_wl % bl_total != black_lion_code:
                        continue
                    key_after_bl = key_after_wl // bl_total

                    if key_after_bl < 0 or key_after_bl > 5:
                        continue

                    winner_code = key_after_bl // 2
                    turn_code = key_after_bl % 2

                    chick_table, _ = _chick_pair_table(occ_after_e)
                    chick_inv = {v: k for k, v in chick_table.items()}
                    if c_rank not in chick_inv:
                        continue
                    chick_placement = chick_inv[c_rank]

                    winner = _WINNER_CODES_INV[winner_code]
                    turn = Player.BLACK if turn_code == 0 else Player.WHITE

                    return _build_state(
                        winner=winner,
                        turn=turn,
                        black_lion_code=black_lion_code,
                        white_lion_code=white_lion_code,
                        giraffe_placement=giraffe_placement,
                        elephant_placement=elephant_placement,
                        chick_placement=chick_placement,
                    )
        return None

    # ---------------------------------------------------------------------------
    # Pre-filter: estimate key_2 = wt * BL_TOTAL + bl (range 0..77) to narrow
    # the set of (bl, wl) pairs that undergo the full giraffe/elephant search.
    #
    # key_2 ≈ target_key / (wl_total * g_total * E_EC) where E_EC is the typical
    # product e_total * c_total.  Use E_EC = 80_000 as a geometric-mean estimate
    # that covers both dense and sparse game states.  Accept pairs whose key_2
    # candidates (one per wt ∈ 0..5) lie within 4× of the estimate.
    # ---------------------------------------------------------------------------
    _E_EC_TYPICAL = 80_000  # typical e_total * c_total product

    candidates = []
    seen = set()
    for bl in range(bl_total):
        for wl in range(bl_total):
            if bl != _ABSENT_LION and wl != _ABSENT_LION and bl == wl:
                continue
            wl_rank, wl_total = _rank_second_lion(bl, wl)
            occ = 0
            if bl != _ABSENT_LION:
                occ |= 1 << bl
            if wl != _ABSENT_LION:
                occ |= 1 << wl
            _, g_total = _non_promoted_pair_table(occ)

            # Estimate of key_2 using the formula:
            #   target_key ≈ key_2 * wl_total * g_total * E_EC
            rough_scale = wl_total * g_total * _E_EC_TYPICAL
            rough_key2 = target_key // rough_scale if rough_scale else 0

            lo = rough_key2 // 4
            hi = rough_key2 * 4 + 1
            # Ensure key_2 always in valid range, even if estimate is 0
            if rough_key2 == 0:
                lo, hi = 0, 5

            accepted = False
            for wt in range(6):
                key2_cand = wt * bl_total + bl
                if lo <= key2_cand <= hi:
                    accepted = True
                    break
            if accepted:
                candidates.append((bl, wl, wl_rank, wl_total, occ))
                seen.add((bl, wl))

    result = _search(candidates)
    if result is not None:
        return result

    # Fallback: search pairs not covered by the pre-filter.
    remaining = []
    for bl in range(bl_total):
        for wl in range(bl_total):
            if bl != _ABSENT_LION and wl != _ABSENT_LION and bl == wl:
                continue
            if (bl, wl) not in seen:
                wl_rank, wl_total = _rank_second_lion(bl, wl)
                occ = 0
                if bl != _ABSENT_LION:
                    occ |= 1 << bl
                if wl != _ABSENT_LION:
                    occ |= 1 << wl
                remaining.append((bl, wl, wl_rank, wl_total, occ))

    result = _search(remaining)
    if result is not None:
        return result

    raise ValueError(f"No valid state found for key {target_key}")


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
    return placement_val - _WHITE_HEN_OFFSET, Player.WHITE, PieceType.HEN


def _build_state(
    winner: Optional[Player],
    turn: Player,
    black_lion_code: int,
    white_lion_code: int,
    giraffe_placement: Tuple[int, ...],
    elephant_placement: Tuple[int, ...],
    chick_placement: Tuple[int, ...],
) -> State:
    board: List[List[Optional[Piece]]] = [[None] * BOARD_COLS for _ in range(BOARD_ROWS)]
    hands: Dict[Player, Dict[PieceType, int]] = {
        Player.BLACK: State.empty_hand(),
        Player.WHITE: State.empty_hand(),
    }

    def place(cell: int, piece: Piece) -> None:
        row, col = divmod(cell, BOARD_COLS)
        board[row][col] = piece

    if black_lion_code != _ABSENT_LION:
        place(black_lion_code, Piece(Player.BLACK, PieceType.LION))
    if white_lion_code != _ABSENT_LION:
        place(white_lion_code, Piece(Player.WHITE, PieceType.LION))

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

    return State(board=board, turn=turn, hands=hands, winner=winner)
