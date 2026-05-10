import unittest

from src.doubutsu.game import State
from src.doubutsu.solver import solve


class SolverTest(unittest.TestCase):
    def test_solver_finds_immediate_win(self):
        state = State.from_sfen(".l./.C./.../.L. b - -")
        result = solve(state, max_depth=2)
        self.assertEqual(result.value, 1)
        self.assertIsNotNone(result.best_move)


if __name__ == "__main__":
    unittest.main()
