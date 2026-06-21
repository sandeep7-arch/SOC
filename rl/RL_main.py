#!/usr/bin/env python3
"""Native self-play reinforcement-learning pipeline.

Python generates fresh self-play FEN/eval rows. The existing C++ NNUE trainer
then consumes `data/fen_files/chessData.fen` and exports `exports/nnue_inference.bin`.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
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


DEFAULT_ENGINE_LIB = ROOT / "native_engine.so"
DEFAULT_MODEL = ROOT / "exports" / "nnue_inference.bin"
DEFAULT_FEN_DATASET = ROOT / "data" / "fen_files" / "chessData.fen"
DEFAULT_GAME_LOG_DIR = ROOT / "data" / "self_play_logs"
DEFAULT_TRAINER = ROOT / "nnue_trainer"
DEFAULT_RL_DIR = ROOT / "checkpoints" / "rl"
DEFAULT_CHAMPION = DEFAULT_RL_DIR / "champion.bin"
DEFAULT_CANDIDATE = DEFAULT_RL_DIR / "candidate.bin"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate native self-play RL data.")
    parser.add_argument("--games", type=int, default=20, help="Number of self-play games to generate.")
    parser.add_argument("--max-ply", type=int, default=160, help="Maximum plies per self-play game.")
    parser.add_argument("--depth", type=int, default=6, help="Native search depth per move.")
    parser.add_argument("--movetime-ms", type=int, default=40, help="Native search budget per move.")
    parser.add_argument("--engine-lib", type=Path, default=DEFAULT_ENGINE_LIB)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--fen-output", type=Path, default=DEFAULT_FEN_DATASET)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_GAME_LOG_DIR)
    parser.add_argument("--replace", action="store_true", help="Replace the FEN output instead of appending.")
    parser.add_argument("--train", action="store_true", help="Run the native NNUE trainer after generation.")
    parser.add_argument("--trainer-bin", type=Path, default=DEFAULT_TRAINER)
    parser.add_argument("--gate", action="store_true", help="Promote trained weights only if they beat the champion.")
    parser.add_argument("--champion-model", type=Path, default=DEFAULT_CHAMPION)
    parser.add_argument("--candidate-model", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--gate-games", type=int, default=6)
    parser.add_argument("--gate-depth", type=int, default=4)
    parser.add_argument("--gate-movetime-ms", type=int, default=50)
    parser.add_argument("--gate-max-ply", type=int, default=120)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.engine_lib.exists():
        raise FileNotFoundError(f"missing native engine library: {args.engine_lib}")
    if not args.model.exists():
        raise FileNotFoundError(f"missing NNUE model: {args.model}")

    engine = ChessEngine(str(args.engine_lib), str(args.model))
    reward_engine = RewardEngine()
    logger = GameLogger(output_dir=args.log_dir)
    replay = ReplayBuffer(capacity=max(1, args.games * args.max_ply))
    self_play = SelfPlayEngine(
        engine=engine,
        reward_engine=reward_engine,
        game_logger=logger,
        max_ply=args.max_ply,
        time_limit_per_move_ms=args.movetime_ms,
        max_search_depth=args.depth,
    )

    try:
        for game_index in range(1, args.games + 1):
            history, result, reason = self_play.execute_match()
            replay.add_rl_game(history, result)
            print(
                f"[self-play] game {game_index}/{args.games}: "
                f"{len(history)} positions, result={result:.1f}, reason={reason}",
                flush=True,
            )

        output_path = replay.export_fen_file(args.fen_output, append=not args.replace)
        print(f"[self-play] exported {replay.size()} positions -> {output_path}")
    finally:
        logger.shutdown()

    if args.train:
        if not args.trainer_bin.exists():
            raise FileNotFoundError(f"missing native trainer binary: {args.trainer_bin}")
        args.champion_model.parent.mkdir(parents=True, exist_ok=True)
        args.candidate_model.parent.mkdir(parents=True, exist_ok=True)
        if not args.champion_model.exists():
            shutil.copy2(args.model, args.champion_model)
            print(f"[rl] seeded champion model -> {args.champion_model}")

        subprocess.run([str(args.trainer_bin)], cwd=str(ROOT), check=True)
        shutil.copy2(args.model, args.candidate_model)
        print(f"[rl] archived trained candidate -> {args.candidate_model}")

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
            )
            result = gate.run()
            print(
                f"[arena] candidate score={result.score:.3f} "
                f"({result.wins}W/{result.draws}D/{result.losses}L)"
            )
            if result.passed:
                shutil.copy2(args.candidate_model, args.champion_model)
                shutil.copy2(args.candidate_model, args.model)
                print(f"[rl] candidate promoted -> {args.model}")
            else:
                shutil.copy2(args.champion_model, args.model)
                print(f"[rl] candidate rejected; champion restored -> {args.model}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
