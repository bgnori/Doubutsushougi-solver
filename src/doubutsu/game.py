from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, List, Optional, Tuple

BOARD_ROWS = 4
BOARD_COLS = 3


class Player(str, Enum):
    BLACK = "b"
    WHITE = "w"

    @property
    def opponent(self) -> "Player":
        return Player.WHITE if self == Player.BLACK else Player.BLACK

    @property
    def forward(self) -> int:
        return -1 if self == Player.BLACK else 1


class PieceType(str, Enum):
    LION = "L"
    GIRAFFE = "G"
    ELEPHANT = "E"
    CHICK = "C"
    HEN = "H"


HAND_TYPES = (PieceType.CHICK, PieceType.GIRAFFE, PieceType.ELEPHANT)

PIECE_DELTAS = {
    PieceType.LION: [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ],
    PieceType.GIRAFFE: [(-1, 0), (1, 0), (0, -1), (0, 1)],
    PieceType.ELEPHANT: [(-1, -1), (-1, 1), (1, -1), (1, 1)],
}


@dataclass(frozen=True)
class Piece:
    owner: Player
    type: PieceType


@dataclass(frozen=True)
class Move:
    to_row: int
    to_col: int
    from_row: Optional[int] = None
    from_col: Optional[int] = None
    drop_piece: Optional[PieceType] = None

    @property
    def is_drop(self) -> bool:
        return self.drop_piece is not None

    def to_notation(self) -> str:
        if self.is_drop:
            return f"drop {self.drop_piece.value}@{self.to_row},{self.to_col}"
        return f"{self.from_row},{self.from_col}->{self.to_row},{self.to_col}"


@dataclass
class State:
    board: List[List[Optional[Piece]]]
    turn: Player
    hands: Dict[Player, Dict[PieceType, int]]
    winner: Optional[Player] = None

    @staticmethod
    def empty_hand() -> Dict[PieceType, int]:
        return {PieceType.CHICK: 0, PieceType.GIRAFFE: 0, PieceType.ELEPHANT: 0}

    @classmethod
    def initial(cls) -> "State":
        board: List[List[Optional[Piece]]] = [[None for _ in range(BOARD_COLS)] for _ in range(BOARD_ROWS)]
        board[0] = [
            Piece(Player.WHITE, PieceType.GIRAFFE),
            Piece(Player.WHITE, PieceType.LION),
            Piece(Player.WHITE, PieceType.ELEPHANT),
        ]
        board[1][1] = Piece(Player.WHITE, PieceType.CHICK)
        board[2][1] = Piece(Player.BLACK, PieceType.CHICK)
        board[3] = [
            Piece(Player.BLACK, PieceType.ELEPHANT),
            Piece(Player.BLACK, PieceType.LION),
            Piece(Player.BLACK, PieceType.GIRAFFE),
        ]
        return cls(
            board=board,
            turn=Player.BLACK,
            hands={Player.BLACK: cls.empty_hand(), Player.WHITE: cls.empty_hand()},
        )

    @classmethod
    def from_sfen(cls, text: str) -> "State":
        parts = text.split()
        if len(parts) != 4:
            raise ValueError("Position must have 4 fields: board turn black_hand white_hand")
        board_part, turn_part, black_hand, white_hand = parts
        rows = board_part.split("/")
        if len(rows) != BOARD_ROWS:
            raise ValueError("Board must contain exactly 4 rows")

        board: List[List[Optional[Piece]]] = []
        for row_text in rows:
            if len(row_text) != BOARD_COLS:
                raise ValueError("Each row must contain exactly 3 cells")
            row: List[Optional[Piece]] = []
            for ch in row_text:
                if ch == ".":
                    row.append(None)
                    continue
                owner = Player.BLACK if ch.isupper() else Player.WHITE
                p = PieceType(ch.upper())
                row.append(Piece(owner, p))
            board.append(row)

        turn = Player(turn_part)
        hands = {
            Player.BLACK: cls.empty_hand(),
            Player.WHITE: cls.empty_hand(),
        }
        for ch in black_hand:
            if black_hand == "-":
                break
            pt = PieceType(ch.upper())
            if pt not in HAND_TYPES:
                raise ValueError("Only C/G/E are allowed in hand")
            hands[Player.BLACK][pt] += 1
        for ch in white_hand:
            if white_hand == "-":
                break
            pt = PieceType(ch.upper())
            if pt not in HAND_TYPES:
                raise ValueError("Only C/G/E are allowed in hand")
            hands[Player.WHITE][pt] += 1

        return cls(board=board, turn=turn, hands=hands)

    def to_sfen(self) -> str:
        def board_char(piece: Optional[Piece]) -> str:
            if piece is None:
                return "."
            ch = piece.type.value
            return ch if piece.owner == Player.BLACK else ch.lower()

        board = "/".join("".join(board_char(c) for c in row) for row in self.board)

        def hand_to_text(owner: Player) -> str:
            out = []
            for pt in HAND_TYPES:
                out.extend(pt.value for _ in range(self.hands[owner][pt]))
            return "".join(out) or "-"

        return f"{board} {self.turn.value} {hand_to_text(Player.BLACK)} {hand_to_text(Player.WHITE)}"

    def clone(self) -> "State":
        return State(
            board=[[c for c in row] for row in self.board],
            turn=self.turn,
            hands={
                Player.BLACK: dict(self.hands[Player.BLACK]),
                Player.WHITE: dict(self.hands[Player.WHITE]),
            },
            winner=self.winner,
        )

    def inside(self, row: int, col: int) -> bool:
        return 0 <= row < BOARD_ROWS and 0 <= col < BOARD_COLS

    def is_terminal(self) -> bool:
        return self.winner is not None

    def _chick_deltas(self, owner: Player) -> List[Tuple[int, int]]:
        return [(owner.forward, 0)]

    def _hen_deltas(self, owner: Player) -> List[Tuple[int, int]]:
        forward = owner.forward
        return [
            (forward, 0),
            (forward, -1),
            (forward, 1),
            (0, -1),
            (0, 1),
            (-forward, 0),
        ]

    def _deltas_for_piece(self, piece: Piece) -> Iterable[Tuple[int, int]]:
        if piece.type in PIECE_DELTAS:
            return PIECE_DELTAS[piece.type]
        if piece.type == PieceType.CHICK:
            return self._chick_deltas(piece.owner)
        return self._hen_deltas(piece.owner)

    def _last_rank(self, owner: Player) -> int:
        return 0 if owner == Player.BLACK else BOARD_ROWS - 1

    def _promotion(self, piece: Piece, to_row: int) -> Piece:
        if piece.type == PieceType.CHICK and to_row == self._last_rank(piece.owner):
            return Piece(piece.owner, PieceType.HEN)
        return piece

    def _capturable_by(self, defender: Player, row: int, col: int) -> bool:
        attacker = defender.opponent
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                p = self.board[r][c]
                if p is None or p.owner != attacker:
                    continue
                for dr, dc in self._deltas_for_piece(p):
                    nr, nc = r + dr, c + dc
                    if nr == row and nc == col:
                        if self.inside(nr, nc):
                            target = self.board[nr][nc]
                            if target is None or target.owner == defender:
                                return True
        return False

    def _find_lion(self, owner: Player) -> Optional[Tuple[int, int]]:
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                p = self.board[r][c]
                if p and p.owner == owner and p.type == PieceType.LION:
                    return (r, c)
        return None

    def legal_moves(self) -> List[Move]:
        if self.winner is not None:
            return []

        moves: List[Move] = []
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                piece = self.board[r][c]
                if piece is None or piece.owner != self.turn:
                    continue
                for dr, dc in self._deltas_for_piece(piece):
                    nr, nc = r + dr, c + dc
                    if not self.inside(nr, nc):
                        continue
                    target = self.board[nr][nc]
                    if target is not None and target.owner == self.turn:
                        continue
                    moves.append(Move(from_row=r, from_col=c, to_row=nr, to_col=nc))

        hand = self.hands[self.turn]
        for pt in HAND_TYPES:
            if hand[pt] <= 0:
                continue
            for r in range(BOARD_ROWS):
                for c in range(BOARD_COLS):
                    if self.board[r][c] is not None:
                        continue
                    if pt == PieceType.CHICK and r == self._last_rank(self.turn):
                        continue
                    moves.append(Move(to_row=r, to_col=c, drop_piece=pt))
        return moves

    def apply_move(self, move: Move) -> "State":
        if move not in self.legal_moves():
            raise ValueError(f"Illegal move: {move}")

        nxt = self.clone()
        player = nxt.turn

        if move.is_drop:
            assert move.drop_piece is not None
            nxt.hands[player][move.drop_piece] -= 1
            nxt.board[move.to_row][move.to_col] = Piece(player, move.drop_piece)
        else:
            assert move.from_row is not None and move.from_col is not None
            piece = nxt.board[move.from_row][move.from_col]
            assert piece is not None
            target = nxt.board[move.to_row][move.to_col]
            if target is not None:
                if target.type == PieceType.LION:
                    nxt.winner = player
                else:
                    captured_type = PieceType.CHICK if target.type == PieceType.HEN else target.type
                    if captured_type in HAND_TYPES:
                        nxt.hands[player][captured_type] += 1
            nxt.board[move.from_row][move.from_col] = None
            nxt.board[move.to_row][move.to_col] = self._promotion(piece, move.to_row)

        if nxt.winner is None:
            lion_pos = nxt._find_lion(player)
            if lion_pos is not None and lion_pos[0] == nxt._last_rank(player.opponent):
                if not nxt._capturable_by(player, lion_pos[0], lion_pos[1]):
                    nxt.winner = player

        nxt.turn = player.opponent
        return nxt
