"""Perfect hash for Doubutsushougi positions.

Each valid position is encoded as a 61-bit integer with no collisions:

  bits  0-47 : board  (12 cells × 4 bits, piece code 0-10)
  bits 48-59 : hands  (6 × 2 bits: BHG, WHG, BHE, WHE, BHC, WHC)
  bit  60    : turn   (0 = Black, 1 = White)

The encoding is injective over all valid positions.
"""
from __future__ import annotations

from typing import Optional

from .game import (
    BOARD_COLS,
    BOARD_ROWS,
    HAND_TYPES,
    Piece,
    PieceType,
    Player,
    State,
)

# Piece → 4-bit code (0 = empty)
_PIECE_TO_CODE = {
    (Player.BLACK, PieceType.LION): 1,
    (Player.WHITE, PieceType.LION): 2,
    (Player.BLACK, PieceType.GIRAFFE): 3,
    (Player.WHITE, PieceType.GIRAFFE): 4,
    (Player.BLACK, PieceType.ELEPHANT): 5,
    (Player.WHITE, PieceType.ELEPHANT): 6,
    (Player.BLACK, PieceType.CHICK): 7,
    (Player.WHITE, PieceType.CHICK): 8,
    (Player.BLACK, PieceType.HEN): 9,
    (Player.WHITE, PieceType.HEN): 10,
}

# 4-bit code → Piece (0 = empty)
_CODE_TO_PIECE = {v: Piece(k[0], k[1]) for k, v in _PIECE_TO_CODE.items()}


def position_to_index(state: State) -> int:
    """Return a unique 61-bit integer for *state*."""
    board_enc = 0
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            sq = r * BOARD_COLS + c
            piece = state.board[r][c]
            code = _PIECE_TO_CODE.get((piece.owner, piece.type), 0) if piece else 0
            board_enc |= code << (4 * sq)

    h = state.hands
    hand_enc = (
        (h[Player.BLACK][PieceType.GIRAFFE])
        | (h[Player.WHITE][PieceType.GIRAFFE] << 2)
        | (h[Player.BLACK][PieceType.ELEPHANT] << 4)
        | (h[Player.WHITE][PieceType.ELEPHANT] << 6)
        | (h[Player.BLACK][PieceType.CHICK] << 8)
        | (h[Player.WHITE][PieceType.CHICK] << 10)
    )

    turn_enc = 1 if state.turn == Player.WHITE else 0
    return board_enc | (hand_enc << 48) | (turn_enc << 60)


def index_to_position(idx: int) -> State:
    """Decode a 61-bit index produced by *position_to_index* back to a State."""
    turn_enc = (idx >> 60) & 1
    hand_enc = (idx >> 48) & 0xFFF
    board_enc = idx & ((1 << 48) - 1)

    board = []
    for r in range(BOARD_ROWS):
        row = []
        for c in range(BOARD_COLS):
            sq = r * BOARD_COLS + c
            code = (board_enc >> (4 * sq)) & 0xF
            row.append(_CODE_TO_PIECE.get(code))
        board.append(row)

    hands = {
        Player.BLACK: {
            PieceType.GIRAFFE: hand_enc & 3,
            PieceType.ELEPHANT: (hand_enc >> 4) & 3,
            PieceType.CHICK: (hand_enc >> 8) & 3,
        },
        Player.WHITE: {
            PieceType.GIRAFFE: (hand_enc >> 2) & 3,
            PieceType.ELEPHANT: (hand_enc >> 6) & 3,
            PieceType.CHICK: (hand_enc >> 10) & 3,
        },
    }

    turn = Player.WHITE if turn_enc else Player.BLACK
    return State(board=board, turn=turn, hands=hands)
