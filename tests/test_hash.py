import unittest
from collections import defaultdict

from src.doubutsu.game import BOARD_COLS, BOARD_ROWS, Move, Piece, PieceType, Player, State
from src.doubutsu.hash import position_key, position_key_space_size, state_from_key


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
        same_position_without_winner = State(
            board=[[cell for cell in row] for row in winning.board],
            turn=winning.turn,
            hands={
                player: dict(winning.hands[player])
                for player in winning.hands
            },
            winner=None,
        )
        self.assertNotEqual(position_key(winning), position_key(same_position_without_winner))

    def test_position_key_has_no_lion_prefix_collisions(self):
        keys = defaultdict(list)
        absent_lion = BOARD_ROWS * BOARD_COLS
        for turn in Player:
            for winner in (None, Player.BLACK, Player.WHITE):
                for black_lion in range(absent_lion + 1):
                    for white_lion in range(absent_lion + 1):
                        if black_lion != absent_lion and white_lion != absent_lion and black_lion == white_lion:
                            continue

                        board = [[None for _ in range(BOARD_COLS)] for _ in range(BOARD_ROWS)]
                        if black_lion != absent_lion:
                            row, col = divmod(black_lion, BOARD_COLS)
                            board[row][col] = Piece(Player.BLACK, PieceType.LION)
                        if white_lion != absent_lion:
                            row, col = divmod(white_lion, BOARD_COLS)
                            board[row][col] = Piece(Player.WHITE, PieceType.LION)

                        state = State(
                            board=board,
                            turn=turn,
                            hands={Player.BLACK: State.empty_hand(), Player.WHITE: State.empty_hand()},
                            winner=winner,
                        )
                        keys[position_key(state)].append((turn, winner, black_lion, white_lion))

        collisions = {key: states for key, states in keys.items() if len(states) > 1}
        self.assertEqual({}, collisions)

    def test_state_from_key_round_trips_representative_keys(self):
        keys = [
            0,
            1,
            position_key_space_size() // 2,
            position_key_space_size() - 1,
            position_key(State.initial()),
        ]
        for key in keys:
            with self.subTest(key=key):
                self.assertEqual(key, position_key(state_from_key(key)))


if __name__ == "__main__":
    unittest.main()
