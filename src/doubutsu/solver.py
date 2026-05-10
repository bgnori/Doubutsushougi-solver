from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .game import Move, Player, State


@dataclass
class SolveResult:
    value: int
    best_move: Optional[Move]
    principal_variation: List[Move]
    searched_depth: int


def _terminal_value(state: State) -> int:
    if state.winner is None:
        return 0
    return 1 if state.winner == state.turn else -1


def solve(state: State, max_depth: int = 8) -> SolveResult:
    memo: Dict[Tuple[str, int], Tuple[int, List[Move]]] = {}

    def negamax(cur: State, depth: int) -> Tuple[int, List[Move]]:
        key = (cur.to_sfen(), depth)
        if key in memo:
            return memo[key]

        if cur.is_terminal():
            result = (_terminal_value(cur), [])
            memo[key] = result
            return result
        if depth == 0:
            result = (0, [])
            memo[key] = result
            return result

        moves = cur.legal_moves()
        if not moves:
            result = (-1, [])
            memo[key] = result
            return result

        best_value = -2
        best_line: List[Move] = []

        for move in moves:
            nxt = cur.apply_move(move)
            child_value, child_line = negamax(nxt, depth - 1)
            score = -child_value
            if score > best_value:
                best_value = score
                best_line = [move] + child_line
            if best_value == 1:
                break

        result = (best_value, best_line)
        memo[key] = result
        return result

    value, line = negamax(state, max_depth)
    return SolveResult(
        value=value,
        best_move=line[0] if line else None,
        principal_variation=line,
        searched_depth=max_depth,
    )


def evaluation_to_text(value: int, turn: Player) -> str:
    # At finite depth, 0 means either true draw or unresolved/unknown result.
    if value > 0:
        return f"{turn.value}:win"
    if value < 0:
        return f"{turn.value}:loss"
    return f"{turn.value}:draw_or_unknown"
