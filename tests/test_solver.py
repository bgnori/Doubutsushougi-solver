import unittest

from src.doubutsu.game import State
from src.doubutsu.solver import solve


class SolverTest(unittest.TestCase):
    def test_solver_finds_immediate_win(self):
        state = State.from_sfen(".l./.C./.../.L. b GGEE C")
        result = solve(state, max_depth=2)
        self.assertEqual(result.value, 1)
        self.assertIsNotNone(result.best_move)

    def test_solver_reports_loss_on_terminal_position(self):
        state = State.from_sfen(".l./.C./.../.L. b GGEE C")
        result_move = solve(state, max_depth=1).best_move
        terminal = state.apply_move(result_move)
        self.assertIsNotNone(result_move)
        result = solve(terminal, max_depth=2)
        self.assertEqual(result.value, -1)

    def test_solver_returns_unknown_at_depth_zero(self):
        state = State.initial()
        result = solve(state, max_depth=0)
        self.assertEqual(result.value, 0)
        self.assertIsNone(result.best_move)


if __name__ == "__main__":
    unittest.main()
