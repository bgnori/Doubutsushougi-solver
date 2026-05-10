import unittest
from collections import defaultdict
from math import comb

from src.doubutsu.game import BOARD_COLS, BOARD_ROWS, Move, Piece, PieceType, Player, State
from src.doubutsu.hash import position_key, position_key_space_size, state_from_key


class HashTest(unittest.TestCase):
    def test_position_key_is_stable_for_equivalent_states(self):
        state = State.from_sfen("gle/.c./.C./ELG b - -")
        clone = State.from_sfen(state.to_sfen())
        self.assertEqual(position_key(state), position_key(clone))

    def test_position_key_changes_with_board_and_hand_state(self):
        state = State.from_sfen("l../.../.../.L. b CCGGEE -")
        dropped = state.apply_move(Move(to_row=2, to_col=1, drop_piece=PieceType.CHICK))
        self.assertNotEqual(position_key(state), position_key(dropped))

    def test_position_key_includes_turn_but_not_winner(self):
        black_to_move = State.from_sfen("gle/.c./.C./ELG b - -")
        white_to_move = State.from_sfen("gle/.c./.C./ELG w - -")
        self.assertNotEqual(position_key(black_to_move), position_key(white_to_move))

        marked_winner = State(
            board=[[cell for cell in row] for row in black_to_move.board],
            turn=black_to_move.turn,
            hands={
                player: dict(black_to_move.hands[player])
                for player in black_to_move.hands
            },
            winner=Player.BLACK,
        )
        self.assertEqual(position_key(black_to_move), position_key(marked_winner))

    def test_position_key_has_no_lion_prefix_collisions(self):
        keys = defaultdict(list)
        for turn in Player:
            for black_lion in range(BOARD_ROWS * BOARD_COLS):
                for white_lion in range(BOARD_ROWS * BOARD_COLS):
                    if black_lion == white_lion:
                        continue

                    board = [[None for _ in range(BOARD_COLS)] for _ in range(BOARD_ROWS)]
                    row, col = divmod(black_lion, BOARD_COLS)
                    board[row][col] = Piece(Player.BLACK, PieceType.LION)
                    row, col = divmod(white_lion, BOARD_COLS)
                    board[row][col] = Piece(Player.WHITE, PieceType.LION)

                    state = State(
                        board=board,
                        turn=turn,
                        hands={
                            Player.BLACK: {
                                PieceType.CHICK: 2,
                                PieceType.GIRAFFE: 2,
                                PieceType.ELEPHANT: 2,
                            },
                            Player.WHITE: State.empty_hand(),
                        },
                    )
                    keys[position_key(state)].append((turn, black_lion, white_lion))

        collisions = {key: states for key, states in keys.items() if len(states) > 1}
        self.assertEqual({}, collisions)

    def test_position_key_rejects_material_incomplete_states(self):
        with self.assertRaises(ValueError):
            position_key(State.from_sfen("l../.../.../.L. b C -"))

        with self.assertRaises(ValueError):
            position_key(State.from_sfen(".../.../.../.L. b CCGGEE -"))

        with self.assertRaises(ValueError):
            position_key(State.from_sfen("ll./.../.../.L. b CCGGEE -"))

    def test_chicks_and_hens_are_one_two_piece_family(self):
        state = State.from_sfen("lH./.../.../.Lh b GGEE -")
        key = position_key(state)
        restored = state_from_key(key)

        self.assertEqual(key, position_key(restored))
        self.assertEqual(state.to_sfen(), restored.to_sfen())

    def test_hens_cannot_be_in_hand(self):
        with self.assertRaises(ValueError):
            State.from_sfen("l../.../.../.L. b HCCGGEE -")

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

    def test_position_key_space_size_matches_hand_count_formula(self):
        per_lion = 0
        for h_g in range(3):
            for h_e in range(3):
                for h_c in range(3):
                    b_g = 2 - h_g
                    b_e = 2 - h_e
                    b_c = 2 - h_c
                    per_lion += (
                        comb(10, b_g)
                        * comb(10 - b_g, b_e)
                        * comb(10 - b_g - b_e, b_c)
                        * (h_g + 1)
                        * (2 ** b_g)
                        * (h_e + 1)
                        * (2 ** b_e)
                        * (h_c + 1)
                        * (4 ** b_c)
                    )

        self.assertEqual(11_878_227, per_lion)
        self.assertEqual(2 * 12 * 11 * per_lion, position_key_space_size())


if __name__ == "__main__":
    unittest.main()
