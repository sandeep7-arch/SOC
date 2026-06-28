"""Stockfish helpers for capped self-play data and model Elo checks."""

from __future__ import annotations

import math
import os
import random
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rl.chess_terminal import (
    MATE_SCORE_CP,
    model_score_for_no_move,
    repetition_key,
    terminal_draw_reason,
    terminal_result_for_no_move as python_terminal_result_for_no_move,
)
from rl.rlgame_logger import GameLogger
from rl.rlreward import RewardEngine
from uci import STARTPOS_FEN, UciPosition


@dataclass(frozen=True)
class EngineMove:
    move: str
    score_cp: int


@dataclass(frozen=True)
class EloResult:
    opponent_elo: int
    games: int
    wins: int
    draws: int
    losses: int
    score: float
    estimated_elo: float
    score_ci_low: float
    score_ci_high: float
    estimated_elo_ci_low: float
    estimated_elo_ci_high: float
    confidence: float

    @property
    def below_or_equal_opponent(self) -> bool:
        return self.estimated_elo <= self.opponent_elo

    @property
    def confidently_at_or_above_opponent(self) -> bool:
        return self.estimated_elo_ci_low >= self.opponent_elo


class StockfishProcess:
    """Small UCI client for a Stockfish process with optional Elo limiting."""

    def __init__(
        self,
        stockfish_bin: Path | str = "stockfish",
        elo: int = 1800,
        threads: int = 1,
        hash_mb: int = 32,
    ) -> None:
        self.requested_elo = elo
        self.configured_elo = elo
        self.uci_elo_min: int | None = None
        self.uci_elo_max: int | None = None
        self.proc = subprocess.Popen(
            [str(stockfish_bin)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self.last_score_cp = 0
        self._send("uci")
        self._read_uci_options()
        self._setoption("Threads", str(max(1, threads)))
        self._setoption("Hash", str(max(1, hash_mb)))
        self._setoption("UCI_LimitStrength", "true")
        self.configured_elo = self._clamp_uci_elo(elo)
        if self.configured_elo != elo:
            print(
                f"[stockfish] requested Elo {elo} is outside this build's "
                f"UCI_Elo range; using {self.configured_elo}",
                flush=True,
            )
        self._setoption("UCI_Elo", str(self.configured_elo))
        self._send("isready")
        self._read_until("readyok")

    def bestmove(self, fen: str, depth: int | None, movetime_ms: int) -> EngineMove:
        self.last_score_cp = 0
        self._send(f"position fen {fen}")
        limits: list[str] = []
        if depth is not None and depth > 0:
            limits.extend(["depth", str(depth)])
        if movetime_ms > 0:
            limits.extend(["movetime", str(movetime_ms)])
        if not limits:
            limits = ["movetime", "50"]
        self._send("go " + " ".join(limits))

        while True:
            line = self._readline()
            if line.startswith("info "):
                self._parse_score(line)
            elif line.startswith("bestmove "):
                parts = line.split()
                move = parts[1] if len(parts) > 1 else "0000"
                if move == "(none)":
                    move = "0000"
                return EngineMove(move=move, score_cp=self.last_score_cp)

    def bestmove_clock(
        self,
        fen: str,
        wtime_ms: int,
        btime_ms: int,
        winc_ms: int,
        binc_ms: int,
        depth: int | None = None,
    ) -> EngineMove:
        self.last_score_cp = 0
        self._send(f"position fen {fen}")
        limits = [
            "wtime",
            str(max(1, wtime_ms)),
            "btime",
            str(max(1, btime_ms)),
            "winc",
            str(max(0, winc_ms)),
            "binc",
            str(max(0, binc_ms)),
        ]
        if depth is not None and depth > 0:
            limits.extend(["depth", str(depth)])
        self._send("go " + " ".join(limits))

        while True:
            line = self._readline()
            if line.startswith("info "):
                self._parse_score(line)
            elif line.startswith("bestmove "):
                parts = line.split()
                move = parts[1] if len(parts) > 1 else "0000"
                if move == "(none)":
                    move = "0000"
                return EngineMove(move=move, score_cp=self.last_score_cp)

    def side_to_move_is_in_check(self, fen: str) -> bool:
        self._send(f"position fen {fen}")
        self._send("d")
        while True:
            line = self._readline()
            if line.startswith("Checkers:"):
                return bool(line.removeprefix("Checkers:").strip())

    def close(self) -> None:
        if self.proc.poll() is None:
            self._send("quit")
            try:
                self.proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def _setoption(self, name: str, value: str) -> None:
        self._send(f"setoption name {name} value {value}")

    def _send(self, command: str) -> None:
        if self.proc.stdin is None:
            raise RuntimeError("Stockfish stdin closed")
        self.proc.stdin.write(command + "\n")
        self.proc.stdin.flush()

    def _readline(self) -> str:
        if self.proc.stdout is None:
            raise RuntimeError("Stockfish stdout closed")
        line = self.proc.stdout.readline()
        if line == "":
            raise RuntimeError("Stockfish exited unexpectedly")
        return line.strip()

    def _read_until(self, marker: str) -> None:
        while True:
            if self._readline() == marker:
                return

    def _read_uci_options(self) -> None:
        while True:
            line = self._readline()
            if line == "uciok":
                return
            self._parse_uci_option(line)

    def _parse_uci_option(self, line: str) -> None:
        if not line.startswith("option ") or " name UCI_Elo " not in line:
            return
        tokens = line.split()
        if "min" in tokens:
            index = tokens.index("min") + 1
            if index < len(tokens):
                try:
                    self.uci_elo_min = int(tokens[index])
                except ValueError:
                    pass
        if "max" in tokens:
            index = tokens.index("max") + 1
            if index < len(tokens):
                try:
                    self.uci_elo_max = int(tokens[index])
                except ValueError:
                    pass

    def _clamp_uci_elo(self, elo: int) -> int:
        configured = elo
        if self.uci_elo_min is not None:
            configured = max(configured, self.uci_elo_min)
        if self.uci_elo_max is not None:
            configured = min(configured, self.uci_elo_max)
        return configured

    def _parse_score(self, line: str) -> None:
        tokens = line.split()
        if "score" not in tokens:
            return
        index = tokens.index("score")
        if index + 2 >= len(tokens):
            return
        score_type = tokens[index + 1]
        try:
            value = int(tokens[index + 2])
        except ValueError:
            return
        if score_type == "cp":
            self.last_score_cp = max(-MATE_SCORE_CP, min(MATE_SCORE_CP, value))
        elif score_type == "mate":
            sign = 1 if value > 0 else -1
            self.last_score_cp = sign * MATE_SCORE_CP


class NativeUciProcess:
    """UCI client for this project's Python/native engine wrapper."""

    def __init__(self, root: Path, model_path: Path, engine_lib: Path) -> None:
        env = os.environ.copy()
        env["SOC_MODEL_PATH"] = str(model_path)
        env["SOC_NATIVE_ENGINE_PATH"] = str(engine_lib)
        self.proc = subprocess.Popen(
            [sys.executable, str(root / "uci.py")],
            cwd=str(root),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._send("uci")
        self._read_until("uciok")
        self._send("isready")
        self._read_until("readyok")

    def bestmove(self, fen: str, depth: int, movetime_ms: int) -> str:
        self._send(f"position fen {fen}")
        self._send(f"go depth {depth} movetime {movetime_ms}")
        while True:
            line = self._readline()
            if line.startswith("bestmove "):
                parts = line.split()
                move = parts[1] if len(parts) > 1 else "0000"
                return "0000" if move == "(none)" else move

    def bestmove_clock(
        self,
        fen: str,
        depth: int,
        wtime_ms: int,
        btime_ms: int,
        winc_ms: int,
        binc_ms: int,
    ) -> str:
        self._send(f"position fen {fen}")
        self._send(
            "go "
            f"depth {max(1, depth)} "
            f"wtime {max(1, wtime_ms)} "
            f"btime {max(1, btime_ms)} "
            f"winc {max(0, winc_ms)} "
            f"binc {max(0, binc_ms)}"
        )
        while True:
            line = self._readline()
            if line.startswith("bestmove "):
                parts = line.split()
                move = parts[1] if len(parts) > 1 else "0000"
                return "0000" if move == "(none)" else move

    def close(self) -> None:
        if self.proc.poll() is None:
            self._send("quit")
            try:
                self.proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def _send(self, command: str) -> None:
        if self.proc.stdin is None:
            raise RuntimeError("native UCI stdin closed")
        self.proc.stdin.write(command + "\n")
        self.proc.stdin.flush()

    def _readline(self) -> str:
        if self.proc.stdout is None:
            raise RuntimeError("native UCI stdout closed")
        line = self.proc.stdout.readline()
        if line == "":
            raise RuntimeError("native UCI engine exited unexpectedly")
        return line.strip()

    def _read_until(self, marker: str) -> None:
        while True:
            if self._readline() == marker:
                return


class StockfishSelfPlayEngine:
    """Generate training rows from Elo-capped Stockfish self-play."""

    def __init__(
        self,
        stockfish_bin: Path | str,
        stockfish_elo: int,
        reward_engine: RewardEngine,
        game_logger: GameLogger | None = None,
        max_ply: int = 200,
        movetime_ms: int = 50,
        threads: int = 1,
        hash_mb: int = 32,
        start_fen: str = STARTPOS_FEN,
        start_fens: list[str] | None = None,
    ) -> None:
        self.stockfish_bin = stockfish_bin
        self.stockfish_elo = stockfish_elo
        self.reward_engine = reward_engine
        self.logger = game_logger
        self.max_ply = max_ply
        self.movetime_ms = movetime_ms
        self.threads = threads
        self.hash_mb = hash_mb
        self.start_fen = start_fen
        self.start_fens = start_fens or [start_fen]

    def execute_match(self) -> tuple[list[dict[str, Any]], float, str]:
        game_id = f"stockfish_{uuid.uuid4().hex[:12]}"
        start_fen = random.choice(self.start_fens)
        position = UciPosition.from_fen(start_fen)
        history: list[dict[str, Any]] = []
        previous_white_cp: float | None = None
        stockfish = StockfishProcess(
            self.stockfish_bin,
            elo=self.stockfish_elo,
            threads=self.threads,
            hash_mb=self.hash_mb,
        )
        try:
            repetitions = {repetition_key(position): 1}
            for ply in range(self.max_ply):
                draw_reason = terminal_draw_reason(position, repetitions)
                if draw_reason is not None:
                    self._log(game_id, history, 0.0, draw_reason)
                    return history, 0.0, draw_reason

                fen = position.to_fen()
                is_white_to_move = position.turn == "w"
                selected = stockfish.bestmove(fen, None, self.movetime_ms)
                if selected.move in ("", "0000", "none", "null", "(none)"):
                    result, reason = terminal_result_for_no_move(position)
                    self._log(game_id, history, result, reason)
                    return history, result, reason

                white_cp = float(selected.score_cp if is_white_to_move else -selected.score_cp)
                step_reward, is_blunder = self.reward_engine.calculate_step_reward(
                    current_cp=white_cp,
                    previous_cp=previous_white_cp,
                    is_white_to_move=is_white_to_move,
                )
                history.append(
                    {
                        "fen": fen,
                        "move": selected.move,
                        "raw_cp": white_cp,
                        "search_cp": float(selected.score_cp),
                        "is_white_to_move": is_white_to_move,
                        "ply": ply,
                        "step_reward": step_reward,
                        "is_blunder": is_blunder,
                    }
                )
                try:
                    position.apply_uci_move(selected.move)
                except ValueError as exc:
                    reason = f"Illegal Stockfish move {selected.move}: {exc}"
                    self._log(game_id, history, 0.0, reason)
                    return history, 0.0, reason
                previous_white_cp = white_cp
                key = repetition_key(position)
                repetitions[key] = repetitions.get(key, 0) + 1
                draw_reason = terminal_draw_reason(position, repetitions)
                if draw_reason is not None:
                    self._log(game_id, history, 0.0, draw_reason)
                    return history, 0.0, draw_reason
        finally:
            stockfish.close()

        reason = "Max ply threshold reached"
        self._log(game_id, history, 0.0, reason)
        return history, 0.0, reason

    def _log(
        self,
        game_id: str,
        history: list[dict[str, Any]],
        result: float,
        reason: str,
    ) -> None:
        if self.logger is not None:
            self.logger.log_game(game_id, history, result, reason)


class EloEstimator:
    """Estimate model Elo from games against Elo-capped Stockfish."""

    def __init__(
        self,
        root: Path,
        engine_lib: Path,
        model_path: Path,
        stockfish_bin: Path | str = "stockfish",
        stockfish_elo: int = 1800,
        games: int = 20,
        native_depth: int = 4,
        native_movetime_ms: int = 50,
        stockfish_movetime_ms: int = 50,
        max_ply: int = 160,
        time_control: str = "clock",
        clock_ms: int = 30_000,
        increment_ms: int = 1_000,
    ) -> None:
        self.root = root
        self.engine_lib = engine_lib
        self.model_path = model_path
        self.stockfish_bin = stockfish_bin
        self.stockfish_elo = stockfish_elo
        self.games = games
        self.native_depth = native_depth
        self.native_movetime_ms = native_movetime_ms
        self.stockfish_movetime_ms = stockfish_movetime_ms
        self.max_ply = max_ply
        self.time_control = time_control
        self.clock_ms = clock_ms
        self.increment_ms = increment_ms

    def run(self) -> EloResult:
        native = NativeUciProcess(self.root, self.model_path, self.engine_lib)
        stockfish = StockfishProcess(self.stockfish_bin, elo=self.stockfish_elo)
        wins = draws = losses = 0
        try:
            for game_index in range(self.games):
                native_is_white = game_index % 2 == 0
                result = self._play_game(native, stockfish, native_is_white)
                if result > 0:
                    wins += 1
                elif result < 0:
                    losses += 1
                else:
                    draws += 1
                role = "white" if native_is_white else "black"
                print(
                    f"[elo] game {game_index + 1}/{self.games}: "
                    f"model as {role}, result={result:+.1f}",
                    flush=True,
                )
        finally:
            native.close()
            stockfish.close()

        total = max(1, wins + draws + losses)
        score = (wins + 0.5 * draws) / total
        estimated = estimate_elo_from_score(score, self.stockfish_elo)
        score_ci_low, score_ci_high = wilson_score_interval(score, total)
        return EloResult(
            opponent_elo=self.stockfish_elo,
            games=total,
            wins=wins,
            draws=draws,
            losses=losses,
            score=score,
            estimated_elo=estimated,
            score_ci_low=score_ci_low,
            score_ci_high=score_ci_high,
            estimated_elo_ci_low=estimate_elo_from_score(score_ci_low, self.stockfish_elo),
            estimated_elo_ci_high=estimate_elo_from_score(score_ci_high, self.stockfish_elo),
            confidence=0.95,
        )

    def _play_game(
        self,
        native: NativeUciProcess,
        stockfish: StockfishProcess,
        native_is_white: bool,
    ) -> float:
        position = UciPosition.from_fen(STARTPOS_FEN)
        repetitions = {repetition_key(position): 1}
        white_ms = self.clock_ms
        black_ms = self.clock_ms
        for _ in range(self.max_ply):
            if terminal_draw_reason(position, repetitions) is not None:
                return 0.0
            native_to_move = (position.turn == "w") == native_is_white
            fen = position.to_fen()
            started = time.perf_counter()
            if self.time_control == "clock":
                if native_to_move:
                    move = native.bestmove_clock(
                        fen,
                        self.native_depth,
                        white_ms,
                        black_ms,
                        self.increment_ms,
                        self.increment_ms,
                    )
                else:
                    move = stockfish.bestmove_clock(
                        fen,
                        white_ms,
                        black_ms,
                        self.increment_ms,
                        self.increment_ms,
                    ).move
            else:
                if native_to_move:
                    move = native.bestmove(fen, self.native_depth, self.native_movetime_ms)
                else:
                    move = stockfish.bestmove(
                        fen,
                        None,
                        self.stockfish_movetime_ms,
                    ).move
            elapsed_ms = max(1, int((time.perf_counter() - started) * 1000.0))
            if self.time_control == "clock":
                if position.turn == "w":
                    white_ms = max(1, white_ms - elapsed_ms + self.increment_ms)
                else:
                    black_ms = max(1, black_ms - elapsed_ms + self.increment_ms)
            if move in ("0000", "none", "null", "", "(none)"):
                return self._score_no_move(position, stockfish, native_to_move)
            try:
                position.apply_uci_move(move)
            except ValueError:
                return -1.0 if native_to_move else 1.0
            key = repetition_key(position)
            repetitions[key] = repetitions.get(key, 0) + 1
            if terminal_draw_reason(position, repetitions) is not None:
                return 0.0

        balance = material_balance(position)
        if not native_is_white:
            balance = -balance
        if balance > 100:
            return 1.0
        if balance < -100:
            return -1.0
        return 0.0

    def _score_no_move(
        self,
        position: UciPosition,
        stockfish: StockfishProcess,
        native_to_move: bool,
    ) -> float:
        return model_score_for_no_move(position, native_to_move)


def estimate_elo_from_score(score: float, opponent_elo: int) -> float:
    bounded_score = max(0.01, min(0.99, score))
    return opponent_elo + 400.0 * math.log10(bounded_score / (1.0 - bounded_score))


def terminal_result_for_no_move(
    position: UciPosition,
    stockfish: StockfishProcess | None = None,
) -> tuple[float, str]:
    return python_terminal_result_for_no_move(position)


def wilson_score_interval(score: float, games: int, z: float = 1.96) -> tuple[float, float]:
    if games <= 0:
        return 0.0, 1.0
    denominator = 1.0 + z * z / games
    center = (score + z * z / (2.0 * games)) / denominator
    margin = z * math.sqrt((score * (1.0 - score) / games) + (z * z / (4.0 * games * games))) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def material_balance(position: UciPosition) -> int:
    values = {"P": 100, "N": 320, "B": 330, "R": 500, "Q": 900, "K": 0}
    score = 0
    for piece in position.board:
        if piece is None:
            continue
        value = values[piece.upper()]
        score += value if piece.isupper() else -value
    return score
