import unittest

from src.doubutsu.game import Move, PieceType, State
from src.doubutsu.hash import position_key


class HashTest(unittest.TestCase):
    def test_position_key_is_stable_for_equivalent_states(self):
        state = State.from_sfen("gle/.c./.C./ELG b - -")
        clone = State.from_sfen(state.to_sfen())
        self.assertEqual(position_key(state), position_key(clone))

    def test_position_key_changes_with_board_and_hand_state(self):
        state = State.from_sfen("l../.../.../.L. b C -")
        dropped = state.apply_move(Move(to_row=2, to_col=1, drop_piece=PieceType.CHICK))
        self.assertNotEqual(position_key(state), position_key(dropped))

    def test_position_key_includes_turn_and_winner(self):
        black_to_move = State.from_sfen(".l./.C./.../.L. b - -")
        white_to_move = State.from_sfen(".l./.C./.../.L. w - -")
        self.assertNotEqual(position_key(black_to_move), position_key(white_to_move))

        winning = black_to_move.apply_move(Move(from_row=1, from_col=1, to_row=0, to_col=1))
        same_position_without_winner = winning.clone()
        same_position_without_winner.winner = None
        self.assertNotEqual(position_key(winning), position_key(same_position_without_winner))


if __name__ == "__main__":
    unittest.main()
