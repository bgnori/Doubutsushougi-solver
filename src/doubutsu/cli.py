from __future__ import annotations

import argparse

from .game import State
from .solver import evaluation_to_text, solve


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Doubutsushougi solver")
    parser.add_argument(
        "--position",
        default="initial",
        help="initial or SFEN-like string: <board> <turn> <black_hand> <white_hand>",
    )
    parser.add_argument("--depth", type=int, default=8, help="search depth")
    return parser


def parse_state(position: str) -> State:
    if position == "initial":
        return State.initial()
    return State.from_sfen(position)


def main() -> int:
    args = build_parser().parse_args()
    state = parse_state(args.position)
    result = solve(state, max_depth=args.depth)

    print(f"position: {state.to_sfen()}")
    print(f"value: {evaluation_to_text(result.value, state.turn)}")
    if result.best_move:
        print(f"best_move: {result.best_move.to_notation()}")
    else:
        print("best_move: none")
    if result.principal_variation:
        print("pv:")
        for i, mv in enumerate(result.principal_variation, start=1):
            print(f"  {i}. {mv.to_notation()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
