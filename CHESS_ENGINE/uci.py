#!/usr/bin/env python3
"""UCI entrypoint for the SOC native chess engine."""

from __future__ import annotations

import os
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from engine.engine import ChessEngine


ROOT = Path(__file__).resolve().parent
DLL_PATH = Path(os.getenv("SOC_NATIVE_ENGINE_PATH", ROOT / "native_engine.so"))
DEFAULT_MODEL_CANDIDATES = (
    ROOT / "exports" / "nnue_inference.bin",
    ROOT / "exports" / "nnue_inference_m.bin",
    ROOT / "exports" / "nnue_inference_m2.bin",
)
MODEL_PATH = Path(
    os.getenv(
        "SOC_MODEL_PATH",
        next((str(path) for path in DEFAULT_MODEL_CANDIDATES if path.exists()), str(DEFAULT_MODEL_CANDIDATES[0])),
    )
)
STARTPOS_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

ENGINE_NAME = "FlowMammal v1.5"
ENGINE_AUTHOR = "Ooppsie"

FILES = "abcdefgh"
RANKS = "12345678"


def write(line: str) -> None:
    print(line, flush=True)


def log_error(message: str) -> None:
    print(f"info string {message}", flush=True)


def square_index(square: str) -> int:
    if len(square) != 2 or square[0] not in FILES or square[1] not in RANKS:
        raise ValueError(f"invalid square: {square}")
    return FILES.index(square[0]) + (int(square[1]) - 1) * 8


def square_name(index: int) -> str:
    return FILES[index % 8] + str((index // 8) + 1)


def normalize_fen(fen: str) -> str:
    parts = fen.split()
    if len(parts) == 4:
        parts.extend(["0", "1"])
    if len(parts) != 6:
        raise ValueError("FEN must contain 4 or 6 fields")
    return " ".join(parts)


@dataclass
class UciPosition:
    board: list[str | None]
    turn: str
    castling: str
    ep_square: str
    halfmove_clock: int
    fullmove_number: int

    @classmethod
    def from_fen(cls, fen: str) -> "UciPosition":
        parts = normalize_fen(fen).split()
        placement, turn, castling, ep_square, halfmove, fullmove = parts
        board: list[str | None] = [None] * 64
        ranks = placement.split("/")
        if len(ranks) != 8:
            raise ValueError("FEN placement must contain 8 ranks")

        for fen_rank, rank_text in enumerate(ranks):
            file_index = 0
            board_rank = 7 - fen_rank
            for char in rank_text:
                if char.isdigit():
                    file_index += int(char)
                    continue
                if char not in "PNBRQKpnbrqk":
                    raise ValueError(f"invalid FEN piece: {char}")
                if file_index >= 8:
                    raise ValueError("too many files in FEN rank")
                board[board_rank * 8 + file_index] = char
                file_index += 1
            if file_index != 8:
                raise ValueError("FEN rank does not contain 8 files")

        if turn not in ("w", "b"):
            raise ValueError("FEN side to move must be w or b")
        if castling != "-" and any(c not in "KQkq" for c in castling):
            raise ValueError("invalid FEN castling rights")
        if ep_square != "-":
            square_index(ep_square)

        return cls(board, turn, castling, ep_square, int(halfmove), int(fullmove))

    def to_fen(self) -> str:
        ranks: list[str] = []
        for rank in range(7, -1, -1):
            empty = 0
            text = ""
            for file_index in range(8):
                piece = self.board[rank * 8 + file_index]
                if piece is None:
                    empty += 1
                else:
                    if empty:
                        text += str(empty)
                        empty = 0
                    text += piece
            if empty:
                text += str(empty)
            ranks.append(text)

        return (
            f"{'/'.join(ranks)} {self.turn} {self.castling or '-'} "
            f"{self.ep_square} {self.halfmove_clock} {self.fullmove_number}"
        )

    def apply_moves(self, moves: Iterable[str]) -> None:
        for move in moves:
            self.apply_uci_move(move)

    def apply_uci_move(self, move: str) -> None:
        if move in ("0000", "null"):
            self._advance_turn()
            return
        if len(move) not in (4, 5):
            raise ValueError(f"invalid UCI move: {move}")

        from_sq = square_index(move[:2])
        to_sq = square_index(move[2:4])
        promotion = move[4].lower() if len(move) == 5 else ""
        if promotion and promotion not in "nbrq":
            raise ValueError(f"invalid promotion piece: {move}")

        piece = self.board[from_sq]
        if piece is None:
            raise ValueError(f"no piece on source square for {move}")

        is_white = piece.isupper()
        if (self.turn == "w") != is_white:
            raise ValueError(f"move {move} does not match side to move")

        target = self.board[to_sq]
        is_pawn = piece.lower() == "p"
        is_capture = target is not None

        ep_capture_sq: int | None = None
        if is_pawn and target is None and self.ep_square != "-":
            if to_sq == square_index(self.ep_square) and (from_sq % 8) != (to_sq % 8):
                ep_capture_sq = to_sq - 8 if is_white else to_sq + 8
                is_capture = True

        self.board[from_sq] = None
        if ep_capture_sq is not None:
            self.board[ep_capture_sq] = None

        placed_piece = piece
        if promotion:
            placed_piece = promotion.upper() if is_white else promotion
        self.board[to_sq] = placed_piece

        if piece.lower() == "k" and abs((to_sq % 8) - (from_sq % 8)) == 2:
            self._apply_castling_rook_move(is_white, to_sq)

        self._update_castling_rights(piece, from_sq, to_sq, target)
        self.ep_square = self._next_ep_square(piece, from_sq, to_sq)
        self.halfmove_clock = 0 if is_pawn or is_capture else self.halfmove_clock + 1
        self._advance_turn()

    def _advance_turn(self) -> None:
        if self.turn == "b":
            self.fullmove_number += 1
        self.turn = "b" if self.turn == "w" else "w"

    def _apply_castling_rook_move(self, is_white: bool, king_to: int) -> None:
        if is_white and king_to == square_index("g1"):
            self._move_rook("h1", "f1")
        elif is_white and king_to == square_index("c1"):
            self._move_rook("a1", "d1")
        elif not is_white and king_to == square_index("g8"):
            self._move_rook("h8", "f8")
        elif not is_white and king_to == square_index("c8"):
            self._move_rook("a8", "d8")

    def _move_rook(self, src: str, dst: str) -> None:
        src_i = square_index(src)
        dst_i = square_index(dst)
        self.board[dst_i] = self.board[src_i]
        self.board[src_i] = None

    def _update_castling_rights(
        self, piece: str, from_sq: int, to_sq: int, captured_piece: str | None
    ) -> None:
        rights = set("" if self.castling == "-" else self.castling)

        if piece == "K":
            rights.discard("K")
            rights.discard("Q")
        elif piece == "k":
            rights.discard("k")
            rights.discard("q")

        rook_rights = {
            square_index("h1"): "K",
            square_index("a1"): "Q",
            square_index("h8"): "k",
            square_index("a8"): "q",
        }
        if piece.lower() == "r" and from_sq in rook_rights:
            rights.discard(rook_rights[from_sq])
        if captured_piece and captured_piece.lower() == "r" and to_sq in rook_rights:
            rights.discard(rook_rights[to_sq])

        self.castling = "".join(c for c in "KQkq" if c in rights)

    @staticmethod
    def _next_ep_square(piece: str, from_sq: int, to_sq: int) -> str:
        if piece.lower() == "p" and abs(to_sq - from_sq) == 16:
            return square_name((from_sq + to_sq) // 2)
        return "-"


class UciDriver:
    def __init__(self) -> None:
        self.position = UciPosition.from_fen(STARTPOS_FEN)
        self.engine: ChessEngine | None = None
        self.hash_size = 2_000_000
        self.eval_cache_size = 524_288
        self.analysis_log_enabled = os.getenv("SOC_ANALYSIS_LOG", "0") == "1"
        self.quantized_eval_enabled = os.getenv("SOC_QUANTIZED_EVAL", "1") != "0"
        self.debug = False

    def loop(self) -> None:
        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line:
                continue
            try:
                if self.handle_command(line):
                    break
            except Exception as exc:
                log_error(f"command failed: {exc}")

    def handle_command(self, line: str) -> bool:
        tokens = line.split()
        command = tokens[0]

        if command == "uci":
            write(f"id name {ENGINE_NAME}")
            write(f"id author {ENGINE_AUTHOR}")
            write("option name Hash type spin default 2000000 min 1024 max 134217728")
            write("option name EvalCache type spin default 524288 min 1024 max 16777216")
            write("option name AnalysisLog type check default false")
            write("option name QuantizedEval type check default true")
            write("uciok")
        elif command == "debug":
            self.debug = len(tokens) > 1 and tokens[1] == "on"
        elif command == "isready":
            self.ensure_engine()
            write("readyok")
        elif command == "setoption":
            self.handle_setoption(tokens)
        elif command == "ucinewgame":
            self.position = UciPosition.from_fen(STARTPOS_FEN)
        elif command == "position":
            self.handle_position(tokens)
        elif command == "go":
            self.handle_go(tokens)
        elif command in ("stop", "ponderhit"):
            log_error("stop is honored after the current native search call returns")
        elif command == "quit":
            return True
        else:
            if self.debug:
                log_error(f"ignored unknown command: {line}")
        return False

    def ensure_engine(self) -> ChessEngine:
        if self.engine is None:
            if not DLL_PATH.exists():
                raise FileNotFoundError(f"missing native library: {DLL_PATH}")
            if not MODEL_PATH.exists():
                raise FileNotFoundError(f"missing NNUE model: {MODEL_PATH}")
            with redirect_stdout_to_stderr():
                self.engine = ChessEngine(
                    str(DLL_PATH),
                    str(MODEL_PATH),
                    tt_size=self.hash_size,
                    eval_cache_size=self.eval_cache_size,
                    emit_search_info=True,
                )
                self.engine.set_quantized_inference(self.quantized_eval_enabled)
        return self.engine

    def handle_setoption(self, tokens: list[str]) -> None:
        name, value = parse_setoption(tokens)
        normalized_name = name.lower()
        if normalized_name == "analysislog":
            self.analysis_log_enabled = value.lower() in ("true", "1", "yes", "on")
        elif normalized_name == "quantizedeval":
            self.quantized_eval_enabled = value.lower() in ("true", "1", "yes", "on")
            if self.engine is not None:
                self.engine.set_quantized_inference(self.quantized_eval_enabled)
        elif normalized_name == "hash":
            self.hash_size = bounded_int(value, 1024, 134_217_728, self.hash_size)
            self._warn_if_engine_already_loaded(name)
        elif normalized_name == "evalcache":
            self.eval_cache_size = bounded_int(value, 1024, 16_777_216, self.eval_cache_size)
            self._warn_if_engine_already_loaded(name)
        elif self.debug:
            log_error(f"option {name} is accepted for GUI compatibility")

    def _warn_if_engine_already_loaded(self, name: str) -> None:
        if self.engine is not None:
            log_error(f"{name} will apply after restarting the engine process")

    def handle_position(self, tokens: list[str]) -> None:
        if len(tokens) < 2:
            raise ValueError("position command is missing a position type")

        moves_index = tokens.index("moves") if "moves" in tokens else len(tokens)
        if tokens[1] == "startpos":
            position = UciPosition.from_fen(STARTPOS_FEN)
        elif tokens[1] == "fen":
            fen = " ".join(tokens[2:moves_index])
            position = UciPosition.from_fen(fen)
        else:
            raise ValueError(f"unsupported position type: {tokens[1]}")

        if moves_index < len(tokens):
            position.apply_moves(tokens[moves_index + 1 :])
        self.position = position

    def handle_go(self, tokens: list[str]) -> None:
        engine = self.ensure_engine()
        depth = parse_int_after(tokens, "depth", 64)
        move_time = parse_int_after(tokens, "movetime", None)
        infinite = "infinite" in tokens
        has_clock = "wtime" in tokens or "btime" in tokens

        if has_clock and move_time is None:
            wtime = parse_int_after(tokens, "wtime", -1) or -1
            btime = parse_int_after(tokens, "btime", -1) or -1
            winc = parse_int_after(tokens, "winc", 0) or 0
            binc = parse_int_after(tokens, "binc", 0) or 0
            movestogo = parse_int_after(tokens, "movestogo", 0) or 0
            fen = self.position.to_fen()
            best_move = engine.get_best_move_with_clock(
                fen,
                max(1, depth),
                max(-1, wtime),
                max(-1, btime),
                max(0, winc),
                max(0, binc),
                max(0, movestogo),
            )
        else:
            if move_time is None:
                move_time = 86_400_000 if "depth" in tokens else allocate_time_ms(tokens, self.position.turn)
            if infinite and "depth" not in tokens and "movetime" not in tokens:
                depth = 64
                move_time = 60_000

            fen = self.position.to_fen()
            best_move = engine.get_best_move(fen, max(1, depth), float(max(1, move_time)))

        best_move = best_move or "0000"
        write(f"bestmove {best_move}")

        if self.analysis_log_enabled and best_move not in ("0000", "none"):
            threading.Thread(
                target=write_analysis_log,
                args=(fen, best_move),
                daemon=True,
            ).start()


def parse_int_after(tokens: list[str], key: str, default: int | None) -> int | None:
    if key not in tokens:
        return default
    index = tokens.index(key) + 1
    if index >= len(tokens):
        return default
    try:
        return int(tokens[index])
    except ValueError:
        return default


def allocate_time_ms(tokens: list[str], turn: str) -> int:
    remaining = parse_int_after(tokens, "wtime" if turn == "w" else "btime", None)
    increment = parse_int_after(tokens, "winc" if turn == "w" else "binc", 0) or 0
    moves_to_go = parse_int_after(tokens, "movestogo", 30) or 30

    if remaining is None:
        return 1000

    reserve = min(1000, max(0, remaining // 20))
    usable = max(1, remaining - reserve)
    base = usable // max(1, moves_to_go)
    return max(1, min(usable, base + int(increment * 0.75)))


def parse_setoption(tokens: list[str]) -> tuple[str, str]:
    if "name" not in tokens:
        return "", ""
    name_start = tokens.index("name") + 1
    value_start = tokens.index("value") if "value" in tokens else len(tokens)
    name = " ".join(tokens[name_start:value_start])
    value = " ".join(tokens[value_start + 1 :]) if value_start < len(tokens) else ""
    return name, value


def bounded_int(value: str, minimum: int, maximum: int, fallback: int) -> int:
    try:
        parsed = int(value)
    except ValueError:
        return fallback
    return max(minimum, min(maximum, parsed))


@contextmanager
def redirect_stdout_to_stderr():
    sys.stdout.flush()
    saved_stdout = os.dup(1)
    try:
        os.dup2(2, 1)
        yield
        sys.stdout.flush()
    finally:
        os.dup2(saved_stdout, 1)
        os.close(saved_stdout)


def write_analysis_log(fen: str, move: str) -> None:
    try:
        with (ROOT / "llm_analysis_stream.log").open("a", encoding="utf-8") as handle:
            handle.write("--- New Analysis ---\n")
            handle.write(f"Position FEN: {fen}\n")
            handle.write(f"Engine Move: {move}\n\n")
    except OSError:
        pass


if __name__ == "__main__":
    UciDriver().loop()
