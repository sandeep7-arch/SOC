"""Candidate-vs-champion gate for RL model promotion."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from uci import STARTPOS_FEN, UciPosition


PIECE_VALUES = {
    "P": 100,
    "N": 320,
    "B": 330,
    "R": 500,
    "Q": 900,
    "K": 0,
}


@dataclass(frozen=True)
class ArenaResult:
    wins: int
    draws: int
    losses: int

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def score(self) -> float:
        if self.games == 0:
            return 0.0
        return (self.wins + 0.5 * self.draws) / self.games

    @property
    def passed(self) -> bool:
        return self.score > 0.5


class UciEngineProcess:
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
                return line.split()[1]

    def close(self) -> None:
        if self.proc.poll() is None:
            self._send("quit")
            try:
                self.proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def _send(self, command: str) -> None:
        if self.proc.stdin is None:
            raise RuntimeError("UCI engine stdin closed")
        self.proc.stdin.write(command + "\n")
        self.proc.stdin.flush()

    def _readline(self) -> str:
        if self.proc.stdout is None:
            raise RuntimeError("UCI engine stdout closed")
        line = self.proc.stdout.readline()
        if line == "":
            raise RuntimeError("UCI engine exited unexpectedly")
        return line.strip()

    def _read_until(self, marker: str) -> None:
        while True:
            if self._readline() == marker:
                return


class ArenaGate:
    def __init__(
        self,
        root: Path,
        engine_lib: Path,
        champion_model: Path,
        candidate_model: Path,
        games: int = 6,
        depth: int = 4,
        movetime_ms: int = 50,
        max_ply: int = 120,
    ) -> None:
        self.root = root
        self.engine_lib = engine_lib
        self.champion_model = champion_model
        self.candidate_model = candidate_model
        self.games = games
        self.depth = depth
        self.movetime_ms = movetime_ms
        self.max_ply = max_ply

    def run(self) -> ArenaResult:
        candidate = UciEngineProcess(self.root, self.candidate_model, self.engine_lib)
        champion = UciEngineProcess(self.root, self.champion_model, self.engine_lib)
        wins = draws = losses = 0
        try:
            for game_index in range(self.games):
                candidate_is_white = game_index % 2 == 0
                result = self._play_game(candidate, champion, candidate_is_white)
                if result > 0:
                    wins += 1
                elif result < 0:
                    losses += 1
                else:
                    draws += 1
                role = "white" if candidate_is_white else "black"
                print(
                    f"[arena] game {game_index + 1}/{self.games}: "
                    f"candidate as {role}, result={result:+.1f}",
                    flush=True,
                )
        finally:
            candidate.close()
            champion.close()
        return ArenaResult(wins=wins, draws=draws, losses=losses)

    def _play_game(
        self,
        candidate: UciEngineProcess,
        champion: UciEngineProcess,
        candidate_is_white: bool,
    ) -> float:
        position = UciPosition.from_fen(STARTPOS_FEN)
        for _ in range(self.max_ply):
            candidate_to_move = (position.turn == "w") == candidate_is_white
            engine = candidate if candidate_to_move else champion
            move = engine.bestmove(position.to_fen(), self.depth, self.movetime_ms)
            if move in ("0000", "none", "null", ""):
                return 0.0
            try:
                position.apply_uci_move(move)
            except ValueError:
                return -1.0 if candidate_to_move else 1.0

        material = material_balance(position)
        if candidate_is_white:
            return _score_from_balance(material)
        return _score_from_balance(-material)


def material_balance(position: UciPosition) -> int:
    score = 0
    for piece in position.board:
        if piece is None:
            continue
        value = PIECE_VALUES[piece.upper()]
        score += value if piece.isupper() else -value
    return score


def _score_from_balance(balance: int) -> float:
    if balance > 100:
        return 1.0
    if balance < -100:
        return -1.0
    return 0.0
