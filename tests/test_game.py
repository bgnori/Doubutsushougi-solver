import unittest

from src.doubutsu.game import Move, Piece, PieceType, Player, State


class GameTest(unittest.TestCase):
    def test_initial_legal_moves_exist(self):
        state = State.initial()
        self.assertTrue(len(state.legal_moves()) > 0)

    def test_chick_promotion(self):
        state = State.from_sfen("l../.C./.../.L. b - -")
        move = Move(from_row=1, from_col=1, to_row=0, to_col=1)
        nxt = state.apply_move(move)
        piece = nxt.board[0][1]
        self.assertIsNotNone(piece)
        self.assertEqual(piece.type, PieceType.HEN)

    def test_drop_from_hand(self):
        state = State.from_sfen("l../.../.../.L. b C -")
        move = Move(to_row=2, to_col=1, drop_piece=PieceType.CHICK)
        nxt = state.apply_move(move)
        self.assertEqual(nxt.board[2][1], Piece(Player.BLACK, PieceType.CHICK))
        self.assertEqual(nxt.hands[Player.BLACK][PieceType.CHICK], 0)

    def test_capture_lion_ends_game(self):
        state = State.from_sfen(".l./.C./.../.L. b - -")
        move = Move(from_row=1, from_col=1, to_row=0, to_col=1)
        nxt = state.apply_move(move)
        self.assertEqual(nxt.winner, Player.BLACK)


if __name__ == "__main__":
    unittest.main()
