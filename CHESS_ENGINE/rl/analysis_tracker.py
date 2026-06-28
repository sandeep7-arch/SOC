"""Game analysis metrics for RL self-play and engine-guided runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from uci import UciPosition, square_index


PIECE_VALUES = {
    "P": 100,
    "N": 320,
    "B": 330,
    "R": 500,
    "Q": 900,
    "K": 0,
}


@dataclass
class AnalysisSummary:
    games: int = 0
    positions: int = 0
    blunders: int = 0
    mistakes: int = 0
    inaccuracies: int = 0
    captures: int = 0
    promotions: int = 0
    checks_or_mate_threats: int = 0
    favorable_trades: int = 0
    unfavorable_trades: int = 0
    structural_inaccuracies: int = 0
    pressure_positions: int = 0
    defensive_recoveries: int = 0
    endgame_conversion_chances: int = 0
    endgame_conversions: int = 0
    average_abs_eval_swing_cp: float = 0.0
    aggression_rate: float = 0.0
    defensive_resourcefulness: float = 0.0
    endgame_conversion_rate: float = 0.0
    blunder_rate: float = 0.0
    mistake_rate: float = 0.0
    inaccuracy_rate: float = 0.0


class AnalysisTracker:
    """Accumulates practical chess-quality metrics from logged search rows."""

    def __init__(
        self,
        blunder_cp: float = -150.0,
        mistake_cp: float = -75.0,
        inaccuracy_cp: float = -30.0,
    ) -> None:
        self.blunder_cp = blunder_cp
        self.mistake_cp = mistake_cp
        self.inaccuracy_cp = inaccuracy_cp
        self.summary = AnalysisSummary()
        self._abs_eval_swing_total = 0.0

    def add_game(self, history: list[dict[str, Any]], result: float) -> None:
        self.summary.games += 1
        self.summary.positions += len(history)

        previous_bad_ply = False
        for row in history:
            step_reward = float(row.get("step_reward", 0.0))
            raw_cp = float(row.get("raw_cp", 0.0))
            move = str(row.get("move", ""))
            fen = str(row.get("fen", ""))

            self._abs_eval_swing_total += abs(step_reward)
            self._classify_error(step_reward, move)
            self._classify_move_shape(fen, move, step_reward)

            if abs(raw_cp) >= 250.0:
                self.summary.pressure_positions += 1
                if previous_bad_ply and step_reward >= 50.0:
                    self.summary.defensive_recoveries += 1
            previous_bad_ply = step_reward <= self.mistake_cp

            if self._is_endgame(fen) and abs(raw_cp) >= 300.0:
                self.summary.endgame_conversion_chances += 1
                if self._result_matches_advantage(raw_cp, result):
                    self.summary.endgame_conversions += 1

    def finish(self) -> AnalysisSummary:
        positions = max(1, self.summary.positions)
        self.summary.average_abs_eval_swing_cp = self._abs_eval_swing_total / positions
        self.summary.aggression_rate = (
            self.summary.captures + self.summary.promotions + self.summary.checks_or_mate_threats
        ) / positions
        self.summary.defensive_resourcefulness = (
            self.summary.defensive_recoveries / max(1, self.summary.pressure_positions)
        )
        self.summary.endgame_conversion_rate = (
            self.summary.endgame_conversions / max(1, self.summary.endgame_conversion_chances)
        )
        self.summary.blunder_rate = self.summary.blunders / positions
        self.summary.mistake_rate = self.summary.mistakes / positions
        self.summary.inaccuracy_rate = self.summary.inaccuracies / positions
        return self.summary

    def write_dashboard(
        self,
        path: str | Path,
        run_id: str,
        config: dict[str, Any],
    ) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": run_id,
            "written_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": config,
            "summary": asdict(self.finish()),
        }
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return output_path

    def _classify_error(self, step_reward: float, move: str) -> None:
        if step_reward <= self.blunder_cp:
            self.summary.blunders += 1
        elif step_reward <= self.mistake_cp:
            self.summary.mistakes += 1
        elif step_reward <= self.inaccuracy_cp:
            self.summary.inaccuracies += 1

        if len(move) >= 4 and step_reward <= self.mistake_cp:
            self.summary.structural_inaccuracies += 1

    def _classify_move_shape(self, fen: str, move: str, step_reward: float) -> None:
        if len(move) < 4:
            return
        try:
            position = UciPosition.from_fen(fen)
            to_sq = square_index(move[2:4])
            target = position.board[to_sq]
        except (ValueError, IndexError):
            return

        is_capture = target is not None
        is_promotion = len(move) == 5
        gives_check_proxy = move[2:4] in ("e8", "e1", "d8", "d1")

        if is_capture:
            self.summary.captures += 1
            if step_reward >= -30.0:
                self.summary.favorable_trades += 1
            elif step_reward <= -75.0:
                self.summary.unfavorable_trades += 1
        if is_promotion:
            self.summary.promotions += 1
        if gives_check_proxy:
            self.summary.checks_or_mate_threats += 1

    def _is_endgame(self, fen: str) -> bool:
        try:
            position = UciPosition.from_fen(fen)
        except ValueError:
            return False
        non_king_material = 0
        queens = 0
        for piece in position.board:
            if piece is None:
                continue
            value = PIECE_VALUES[piece.upper()]
            non_king_material += value
            if piece.upper() == "Q":
                queens += 1
        return queens == 0 or non_king_material <= 2600

    @staticmethod
    def _result_matches_advantage(raw_cp: float, result: float) -> bool:
        if raw_cp > 0:
            return result > 0
        if raw_cp < 0:
            return result < 0
        return result == 0
