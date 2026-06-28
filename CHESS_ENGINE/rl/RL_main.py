#!/usr/bin/env python3
"""Native self-play reinforcement-learning pipeline.

Python generates fresh self-play FEN/eval rows, then the C++ NNUE trainer
fine-tunes the current engine model and exports a candidate inference binary.
"""

from __future__ import annotations

import argparse
import os
import json
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.engine import ChessEngine
from rl.rlarena import ArenaGate
from rl.rlgame_logger import GameLogger
from rl.rlreplay_buffer import ReplayBuffer
from rl.rlreward import RewardEngine
from rl.rlself_play import SelfPlayEngine
from rl.stockfish_tools import StockfishSelfPlayEngine
from uci import STARTPOS_FEN, UciPosition


DEFAULT_ENGINE_LIB = ROOT / "native_engine.so"
DEFAULT_MODEL = ROOT / "exports" / "nnue_inference.bin"
DEFAULT_FEN_DATASET = ROOT / "data" / "fen_files" / "chessData.fen"
DEFAULT_GAME_LOG_DIR = ROOT / "data" / "self_play_logs"
DEFAULT_TRAINER = ROOT / "nnue_trainer"
DEFAULT_RL_DIR = ROOT / "exports" / "rl"
DEFAULT_RL_ARCHIVE_DIR = DEFAULT_RL_DIR
DEFAULT_RESUME_DIR = ROOT / "checkpoints" / "resume" / "rl"
DEFAULT_CHAMPION = DEFAULT_RL_DIR / "champion.bin"
DEFAULT_CANDIDATE = DEFAULT_RL_DIR / "candidate.bin"
DEFAULT_STOCKFISH = Path("stockfish")
DEFAULT_AUDIT_LOG = DEFAULT_RL_DIR / "training_audit.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate native self-play RL data.")
    parser.add_argument("--games", type=int, default=20, help="Number of self-play games to generate.")
    parser.add_argument("--max-ply", type=int, default=160, help="Maximum plies per self-play game.")
    parser.add_argument("--depth", type=int, default=6, help="Native search depth per move.")
    parser.add_argument("--movetime-ms", type=int, default=40, help="Native search budget per move.")
    parser.add_argument(
        "--generator",
        choices=("native", "stockfish"),
        default="native",
        help=(
            "Data source: native uses your current weights, stockfish uses capped Stockfish."
        ),
    )
    parser.add_argument("--engine-lib", type=Path, default=DEFAULT_ENGINE_LIB)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--fen-output", type=Path, default=DEFAULT_FEN_DATASET)
    parser.add_argument("--start-fen", default=None, help="Single starting FEN for generated games.")
    parser.add_argument("--start-fen-file", type=Path, default=None, help="Optional file of starting FENs, one per line.")
    parser.add_argument(
        "--val-fen",
        type=Path,
        default=None,
        help="Optional held-out validation FEN passed to the native trainer.",
    )
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_GAME_LOG_DIR)
    parser.add_argument("--replace", action="store_true", help="Replace the FEN output instead of appending.")
    parser.add_argument("--stockfish-bin", type=Path, default=DEFAULT_STOCKFISH)
    parser.add_argument("--stockfish-elo", type=int, default=1800)
    parser.add_argument("--stockfish-movetime-ms", type=int, default=40)
    parser.add_argument("--stockfish-threads", type=int, default=1)
    parser.add_argument("--stockfish-hash-mb", type=int, default=32)
    parser.add_argument(
        "--audit-log",
        type=Path,
        default=DEFAULT_AUDIT_LOG,
        help="JSONL proof log written for every --train run.",
    )
    parser.add_argument("--train", action="store_true", help="Run the native NNUE trainer after generation.")
    parser.add_argument("--trainer-bin", type=Path, default=DEFAULT_TRAINER)
    parser.add_argument("--trainer-checkpoint-dir", type=Path, default=DEFAULT_RESUME_DIR)
    parser.add_argument(
        "--trainer-mode",
        choices=("continue", "finetune", "scratch"),
        default="finetune",
        help="Mode passed to the native trainer.",
    )
    parser.add_argument("--trainer-lr", type=float, default=1e-5, help="Fine-tune learning rate.")
    parser.add_argument("--trainer-epochs", type=int, default=3, help="Fine-tune epochs.")
    parser.add_argument("--trainer-batch-size", type=int, default=8192, help="Native trainer batch size.")
    parser.add_argument("--trainer-limit", type=int, default=13_000_000, help="Maximum FEN rows loaded by the trainer.")
    parser.add_argument("--trainer-val-limit", type=int, default=1_250_000, help="Maximum validation FEN rows loaded by the trainer.")
    parser.add_argument(
        "--trainer-eval-perspective",
        choices=("white", "stm"),
        default="white",
        help="Perspective of FEN targets written by the replay buffer.",
    )
    parser.add_argument(
        "--terminal-blend",
        type=float,
        default=0.0,
        help="Blend terminal game result into per-position search targets. Keep small, e.g. 0.05.",
    )
    parser.add_argument("--copy-retries", type=int, default=3, help="Retries for model copy/promotion I/O.")
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_RL_ARCHIVE_DIR, help="Directory for timestamped RL candidates and gate outcomes.")
    parser.add_argument("--gate", action="store_true", help="Promote trained weights only if they beat the champion.")
    parser.add_argument("--champion-model", type=Path, default=DEFAULT_CHAMPION)
    parser.add_argument("--candidate-model", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--gate-games", type=int, default=6)
    parser.add_argument("--gate-depth", type=int, default=4)
    parser.add_argument("--gate-movetime-ms", type=int, default=50)
    parser.add_argument("--gate-max-ply", type=int, default=120)
    parser.add_argument("--gate-min-score", type=float, default=0.55, help="Minimum candidate score required for promotion.")
    parser.add_argument("--tt-size", type=int, default=2_000_000)
    parser.add_argument("--eval-cache-size", type=int, default=524_288)
    parser.add_argument(
        "--rl-framework",
        choices=("value-guided", "policy-gradient", "actor-critic", "ppo", "alphazero"),
        default="value-guided",
        help="Optimization family recorded in training audits.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.games < 0:
        raise ValueError("--games must be non-negative")
    if args.trainer_epochs <= 0:
        raise ValueError("--trainer-epochs must be positive")
    if args.trainer_batch_size <= 0:
        raise ValueError("--trainer-batch-size must be positive")
    if args.trainer_limit <= 0:
        raise ValueError("--trainer-limit must be positive")
    if args.trainer_val_limit <= 0:
        raise ValueError("--trainer-val-limit must be positive")
    if args.gate and args.gate_games <= 0:
        raise ValueError("--gate-games must be positive when --gate is enabled")
    if args.gate and args.gate_max_ply <= 0:
        raise ValueError("--gate-max-ply must be positive when --gate is enabled")
    if not 0.0 <= args.gate_min_score <= 1.0:
        raise ValueError("--gate-min-score must be between 0.0 and 1.0")
    if not 0.0 <= args.terminal_blend <= 1.0:
        raise ValueError("--terminal-blend must be between 0.0 and 1.0")
    needs_native_engine = (
        args.generator == "native"
        or args.train
        or args.gate
    )
    if needs_native_engine and not args.engine_lib.exists():
        raise FileNotFoundError(f"missing native engine library: {args.engine_lib}")
    if needs_native_engine and not args.model.exists():
        raise FileNotFoundError(f"missing NNUE model: {args.model}")
    start_fens = load_start_fens(args.start_fen, args.start_fen_file)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    training_audit: dict[str, object] = {
        "run_id": run_id,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "model": str(args.model),
        "fen_output": str(args.fen_output),
        "val_fen": str(args.val_fen) if args.val_fen is not None else None,
        "start_fens": {
            "count": len(start_fens),
            "file": str(args.start_fen_file) if args.start_fen_file is not None else None,
        },
        "requested_generator": args.generator,
        "stockfish_data_elo": args.stockfish_elo if args.generator == "stockfish" else None,
        "train": bool(args.train),
        "gate": bool(args.gate),
        "trainer": {
            "rl_framework": args.rl_framework,
            "mode": args.trainer_mode,
            "lr": args.trainer_lr,
            "epochs": args.trainer_epochs,
            "batch_size": args.trainer_batch_size,
            "limit": args.trainer_limit,
            "val_limit": args.trainer_val_limit,
            "eval_perspective": args.trainer_eval_perspective,
            "terminal_blend": args.terminal_blend,
        },
    }

    generator = args.generator
    training_audit["selected_generator"] = generator

    reward_engine = RewardEngine()
    logger = GameLogger(output_dir=args.log_dir)
    replay = ReplayBuffer(capacity=max(1, args.games * args.max_ply))
    exported_positions = 0
    termination_counts: Counter[str] = Counter()
    result_counts: Counter[str] = Counter()
    if generator == "stockfish":
        self_play = StockfishSelfPlayEngine(
            stockfish_bin=args.stockfish_bin,
            stockfish_elo=args.stockfish_elo,
            reward_engine=reward_engine,
            game_logger=logger,
            max_ply=args.max_ply,
            movetime_ms=args.stockfish_movetime_ms,
            threads=args.stockfish_threads,
            hash_mb=args.stockfish_hash_mb,
            start_fens=start_fens,
        )
        print(f"[self-play] using Stockfish capped at Elo {args.stockfish_elo}", flush=True)
    else:
        engine = ChessEngine(
            str(args.engine_lib),
            str(args.model),
            tt_size=args.tt_size,
            eval_cache_size=args.eval_cache_size,
        )
        self_play = SelfPlayEngine(
            engine=engine,
            reward_engine=reward_engine,
            game_logger=logger,
            max_ply=args.max_ply,
            time_limit_per_move_ms=args.movetime_ms,
            max_search_depth=args.depth,
            start_fens=start_fens,
        )
        print("[self-play] using native model weights", flush=True)

    try:
        if args.games > 0:
            for game_index in range(1, args.games + 1):
                history, result, reason = self_play.execute_match()
                replay.add_rl_game(history, result, blend_terminal=args.terminal_blend)
                termination_counts[reason] += 1
                result_counts[f"{result:.1f}"] += 1
                print(
                    f"[self-play] game {game_index}/{args.games}: "
                    f"{len(history)} positions, result={result:.1f}, reason={reason}",
                    flush=True,
                )

            output_path = replay.export_fen_file(args.fen_output, append=not args.replace)
            exported_positions = replay.size()
            print(f"[self-play] exported {replay.size()} positions -> {output_path}")
            print(
                f"[self-play] terminations={dict(termination_counts)} "
                f"results={dict(result_counts)}",
                flush=True,
            )
    finally:
        logger.shutdown()

    training_audit["generated_games"] = args.games
    training_audit["exported_positions"] = exported_positions
    training_audit["termination_counts"] = dict(termination_counts)
    training_audit["result_counts"] = dict(result_counts)

    if args.train:
        if args.games > 0 and exported_positions == 0:
            raise RuntimeError("no self-play positions were exported; refusing to train")
        training_audit["training_started_at_utc"] = datetime.now(timezone.utc).isoformat()
        if not args.trainer_bin.exists():
            raise FileNotFoundError(f"missing native trainer binary: {args.trainer_bin}")
        args.champion_model.parent.mkdir(parents=True, exist_ok=True)
        args.candidate_model.parent.mkdir(parents=True, exist_ok=True)
        args.trainer_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if not args.champion_model.exists():
            safe_copy_model(args.model, args.champion_model, attempts=args.copy_retries)
            print(f"[rl] seeded champion model -> {args.champion_model}")

        trainer_command = [
                str(args.trainer_bin),
                "--mode",
                args.trainer_mode,
                "--base-model",
                str(args.model),
                "--fen",
                str(args.fen_output),
                "--output-model",
                str(args.candidate_model),
                "--checkpoint-dir",
                str(args.trainer_checkpoint_dir),
                "--lr",
                str(args.trainer_lr),
                "--epochs",
                str(args.trainer_epochs),
                "--batch-size",
                str(args.trainer_batch_size),
                "--limit",
                str(args.trainer_limit),
                "--eval-perspective",
                args.trainer_eval_perspective,
            ]
        if args.val_fen is not None:
            if not args.val_fen.exists():
                raise FileNotFoundError(f"missing validation FEN file: {args.val_fen}")
            trainer_command.extend(
                [
                    "--val-fen",
                    str(args.val_fen),
                    "--val-limit",
                    str(args.trainer_val_limit),
                ]
            )
        print(
            "[rl] trainer settings: "
            f"lr={args.trainer_lr} epochs={args.trainer_epochs} "
            f"batch_size={args.trainer_batch_size} limit={args.trainer_limit} "
            f"val_fen={args.val_fen if args.val_fen is not None else 'auto-split'}",
            flush=True,
        )
        subprocess.run(
            trainer_command,
            cwd=str(ROOT),
            check=True,
        )
        candidate_archive = archive_model(
            args.candidate_model,
            args.archive_dir / "candidates",
            f"{run_id}.bin",
            attempts=args.copy_retries,
        )
        print(f"[rl] archived trained candidate -> {candidate_archive}")

        gate_result = None
        candidate_promoted = False
        if args.gate:
            gate = ArenaGate(
                root=ROOT,
                engine_lib=args.engine_lib,
                champion_model=args.champion_model,
                candidate_model=args.candidate_model,
                games=args.gate_games,
                depth=args.gate_depth,
                movetime_ms=args.gate_movetime_ms,
                max_ply=args.gate_max_ply,
                start_fens=start_fens,
                min_score=args.gate_min_score,
            )
            result = gate.run()
            gate_result = result
            print(
                f"[arena] candidate score={result.score:.3f} "
                f"({result.wins}W/{result.draws}D/{result.losses}L)"
            )
            if result.passed:
                promoted_archive = archive_model(
                    args.candidate_model,
                    args.archive_dir / "promoted",
                    f"{run_id}_score_{result.score:.3f}.bin",
                    attempts=args.copy_retries,
                )
                safe_copy_model(args.candidate_model, args.champion_model, attempts=args.copy_retries)
                safe_copy_model(args.candidate_model, args.model, attempts=args.copy_retries)
                candidate_promoted = True
                print(f"[rl] candidate promoted -> {args.model} (archive: {promoted_archive})")
            else:
                rejected_archive = archive_model(
                    args.candidate_model,
                    args.archive_dir / "rejected",
                    f"{run_id}_score_{result.score:.3f}.bin",
                    attempts=args.copy_retries,
                )
                safe_copy_model(args.champion_model, args.model, attempts=args.copy_retries)
                print(f"[rl] candidate rejected; champion restored -> {args.model} (archive: {rejected_archive})")
        else:
            promoted_archive = archive_model(
                args.candidate_model,
                args.archive_dir / "promoted",
                f"{run_id}_ungated.bin",
                attempts=args.copy_retries,
            )
            safe_copy_model(args.candidate_model, args.model, attempts=args.copy_retries)
            candidate_promoted = True
            print(f"[rl] candidate accepted without gate -> {args.model} (archive: {promoted_archive})")

        training_audit.update(
            {
                "training_finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "candidate_model": str(args.candidate_model),
                "candidate_archive": str(candidate_archive),
                "champion_model": str(args.champion_model),
                "candidate_promoted": candidate_promoted,
            }
        )
        if gate_result is not None:
            training_audit["arena_gate"] = {
                "wins": gate_result.wins,
                "draws": gate_result.draws,
                "losses": gate_result.losses,
                "score": gate_result.score,
                "min_score": gate_result.min_score,
                "passed": gate_result.passed,
            }
        print(f"[audit] source={generator} promoted={candidate_promoted}", flush=True)
        write_training_audit(args.audit_log, training_audit)
        print(f"[audit] wrote training proof -> {args.audit_log}", flush=True)

    return 0


def write_training_audit(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    complete_row = dict(row)
    complete_row["audit_written_at_utc"] = datetime.now(timezone.utc).isoformat()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(complete_row, sort_keys=True) + "\n")


def load_start_fens(start_fen: str | None, start_fen_file: Path | None) -> list[str]:
    fens: list[str] = []
    if start_fen is not None:
        fens.append(start_fen.strip())
    if start_fen_file is not None:
        if not start_fen_file.exists():
            raise FileNotFoundError(f"missing start FEN file: {start_fen_file}")
        with start_fen_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    fens.append(stripped)
    if not fens:
        fens.append(STARTPOS_FEN)

    valid_fens: list[str] = []
    for fen in fens:
        try:
            UciPosition.from_fen(fen)
        except ValueError as exc:
            raise ValueError(f"invalid start FEN: {fen}") from exc
        valid_fens.append(fen)
    return valid_fens


def archive_model(src: Path, directory: Path, filename: str, attempts: int = 3) -> Path:
    dst = directory / filename
    safe_copy_model(src, dst, attempts=attempts)
    return dst


def safe_copy_model(src: Path, dst: Path, attempts: int = 3) -> None:
    if not src.exists():
        raise FileNotFoundError(f"missing model to copy: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    last_error: OSError | None = None
    total_attempts = max(1, attempts)
    for attempt in range(1, total_attempts + 1):
        tmp = dst.with_name(f".{dst.name}.tmp.{os.getpid()}.{attempt}")
        try:
            shutil.copy2(src, tmp)
            if tmp.stat().st_size != src.stat().st_size:
                raise OSError(f"short model copy: {src} -> {tmp}")
            os.replace(tmp, dst)
            return
        except OSError as exc:
            last_error = exc
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass
            if attempt < total_attempts:
                time.sleep(0.25 * attempt)
    raise OSError(f"failed to copy model after {total_attempts} attempts: {src} -> {dst}") from last_error


if __name__ == "__main__":
    raise SystemExit(main())
