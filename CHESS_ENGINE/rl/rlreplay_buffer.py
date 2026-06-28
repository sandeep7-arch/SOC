"""Replay artifacts for native NNUE training.

The C++ trainer ingests rows shaped as:

    <fen>,<centipawn evaluation from White's perspective>

or the backward-compatible weighted form:

    <fen>,<centipawn evaluation from White's perspective>,<sample weight>

This module keeps Python self-play data in that format instead of maintaining a
parallel Python feature encoder.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

TERMINAL_RESULT_CP = 600.0
MIN_SAMPLE_WEIGHT = 0.10
MAX_SAMPLE_WEIGHT = 3.00


@dataclass(frozen=True)
class FenExperience:
    fen: str
    target_cp: float
    move: str = ""
    ply: int = 0
    game_id: str = ""
    weight: float = 1.0


class ReplayBuffer:
    def __init__(self, capacity: int = 1_000_000) -> None:
        self.capacity = capacity
        self.buffer: deque[FenExperience] = deque(maxlen=capacity)

    def add(
        self,
        fen: str,
        target_cp: float,
        move: str = "",
        ply: int = 0,
        game_id: str = "",
        weight: float = 1.0,
    ) -> None:
        self.buffer.append(
            FenExperience(
                fen=fen,
                target_cp=float(max(-4000.0, min(4000.0, target_cp))),
                move=move,
                ply=ply,
                game_id=game_id,
                weight=float(max(MIN_SAMPLE_WEIGHT, min(MAX_SAMPLE_WEIGHT, weight))),
            )
        )

    def add_rl_game(
        self,
        history: list[dict[str, Any]],
        result: float = 0.0,
        game_id: str = "",
        blend_terminal: float = 0.0,
    ) -> None:
        total = max(1, len(history))
        terminal_cp = normalize_terminal_result(result)
        for row in history:
            progress = (int(row.get("ply", 0)) + 1) / total
            search_cp = float(row.get("raw_cp", 0.0))
            target = (1.0 - blend_terminal) * search_cp + blend_terminal * terminal_cp
            step_reward = abs(float(row.get("step_reward", 0.0)))
            tactical_weight = 1.0 + min(1.0, step_reward / 300.0)
            if bool(row.get("is_blunder", False)):
                tactical_weight += 0.50
            self.add(
                fen=str(row["fen"]),
                target_cp=target,
                move=str(row.get("move", "")),
                ply=int(row.get("ply", 0)),
                game_id=game_id,
                weight=(0.25 + 0.75 * progress) * tactical_weight,
            )

    def extend(self, experiences: Iterable[FenExperience]) -> None:
        for exp in experiences:
            self.buffer.append(exp)

    def sample(self, batch_size: int) -> list[FenExperience]:
        actual = min(batch_size, len(self.buffer))
        return random.sample(list(self.buffer), actual)

    def export_fen_file(
        self,
        path: str | Path,
        append: bool = True,
        include_weights: bool = True,
    ) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with output_path.open(mode, encoding="utf-8") as handle:
            for exp in self.buffer:
                if include_weights:
                    handle.write(f"{exp.fen},{exp.target_cp:.2f},{exp.weight:.4f}\n")
                else:
                    handle.write(f"{exp.fen},{exp.target_cp:.2f}\n")
        return output_path

    def clear(self) -> None:
        self.buffer.clear()

    def size(self) -> int:
        return len(self.buffer)

    def is_empty(self) -> bool:
        return not self.buffer


def normalize_terminal_result(result: float) -> float:
    """Accept either cp-like terminal values or normalized -1/0/+1 results."""
    value = float(result)
    if -1.0 <= value <= 1.0:
        return value * TERMINAL_RESULT_CP
    return max(-4000.0, min(4000.0, value))
