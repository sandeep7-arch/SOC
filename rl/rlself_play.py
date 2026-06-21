"""Self-play data generation for the native SOC engine.

The Python RL layer owns orchestration and logging. Search, legality, FEN
parsing, feature encoding, and NNUE training remain in the native engine stack.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from rl.rlgame_logger import GameLogger
from rl.rlreward import RewardEngine
from uci import STARTPOS_FEN, UciPosition


@dataclass(frozen=True)
class SearchSample:
    fen: str
    move: str
    score_cp: float
    white_cp: float
    is_white_to_move: bool
    ply: int
    step_reward: float
    is_blunder: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "fen": self.fen,
            "move": self.move,
            "raw_cp": self.white_cp,
            "search_cp": self.score_cp,
            "is_white_to_move": self.is_white_to_move,
            "ply": self.ply,
            "step_reward": self.step_reward,
            "is_blunder": self.is_blunder,
        }


class SelfPlayEngine:
    """Generate self-play games through the current native UCI engine wrapper."""

    def __init__(
        self,
        engine: Any,
        reward_engine: RewardEngine,
        game_logger: GameLogger | None = None,
        max_ply: int = 200,
        time_limit_per_move_ms: int = 50,
        max_search_depth: int = 8,
        start_fen: str = STARTPOS_FEN,
    ) -> None:
        self.engine = engine
        self.reward_engine = reward_engine
        self.logger = game_logger
        self.max_ply = max_ply
        self.time_limit_ms = time_limit_per_move_ms
        self.max_depth = max_search_depth
        self.start_fen = start_fen

    def execute_match(self) -> tuple[list[dict[str, Any]], float, str]:
        game_id = f"game_{uuid.uuid4().hex[:12]}"
        position = UciPosition.from_fen(self.start_fen)
        history: list[dict[str, Any]] = []
        previous_white_cp: float | None = None

        if hasattr(self.engine, "clear_caches"):
            self.engine.clear_caches()

        for ply in range(self.max_ply):
            fen = position.to_fen()
            is_white_to_move = position.turn == "w"
            move, score_cp = self.engine.get_best_move_with_score(
                fen,
                depth=self.max_depth,
                time_limit_ms=float(self.time_limit_ms),
            )

            if move in ("", "0000", "none", "null"):
                result, reason = self._terminal_result(position.turn)
                self._log(game_id, history, result, reason)
                return history, result, reason

            white_cp = float(score_cp if is_white_to_move else -score_cp)
            step_reward, is_blunder = self.reward_engine.calculate_step_reward(
                current_cp=white_cp,
                previous_cp=previous_white_cp,
                is_white_to_move=is_white_to_move,
            )

            sample = SearchSample(
                fen=fen,
                move=move,
                score_cp=float(score_cp),
                white_cp=white_cp,
                is_white_to_move=is_white_to_move,
                ply=ply,
                step_reward=step_reward,
                is_blunder=is_blunder,
            )
            history.append(sample.as_dict())

            try:
                position.apply_uci_move(move)
            except ValueError as exc:
                reason = f"Illegal native move {move}: {exc}"
                self._log(game_id, history, 0.0, reason)
                return history, 0.0, reason

            previous_white_cp = white_cp

        reason = "Max ply threshold reached"
        self._log(game_id, history, 0.0, reason)
        return history, 0.0, reason

    def _terminal_result(self, side_to_move: str) -> tuple[float, str]:
        # Search returns no move for mate or stalemate. Without a legal move API exposed
        # to Python, keep the training target neutral and preserve the reason explicitly.
        return 0.0, f"No legal move for {'white' if side_to_move == 'w' else 'black'}"

    def _log(
        self,
        game_id: str,
        history: list[dict[str, Any]],
        result: float,
        reason: str,
    ) -> None:
        if self.logger is not None:
            self.logger.log_game(game_id, history, result, reason)
