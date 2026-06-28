#!/usr/bin/env python3
"""Standalone analysis and Elo tracker for the current SOC engine.

This script is intentionally separate from RL training. It plays the current
model through a short native analysis sample, then against several Elo-capped
Stockfish settings, and records both reports together.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.engine import ChessEngine
from rl.analysis_tracker import AnalysisTracker
from rl.rlreward import RewardEngine
from rl.rlself_play import SelfPlayEngine
from rl.stockfish_tools import EloEstimator, EloResult


DEFAULT_ENGINE_LIB = ROOT / "native_engine.so"
DEFAULT_MODEL = ROOT / "exports" / "nnue_inference.bin"
DEFAULT_STOCKFISH = Path("stockfish")
DEFAULT_REPORT = ROOT / "exports" / "rl" / "elo_reports.jsonl"
DEFAULT_ANALYSIS_DASHBOARD = ROOT / "exports" / "rl" / "analysis_dashboard.json"


@dataclass(frozen=True)
class LadderFit:
    estimated_elo: int
    ci_low: int
    ci_high: int
    confidence: float
    total_games: int
    total_wins: int
    total_draws: int
    total_losses: int
    total_score: float
    note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze SOC engine behavior and measure Elo with a Stockfish ladder.")
    parser.add_argument("--engine-lib", type=Path, default=DEFAULT_ENGINE_LIB)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--stockfish-bin", type=Path, default=DEFAULT_STOCKFISH)
    parser.add_argument(
        "--stockfish-elos",
        default="1320,1400,1600,1800,2000",
        help="Comma-separated capped Stockfish Elo ladder.",
    )
    parser.add_argument("--games-per-level", type=int, default=40)
    parser.add_argument("--native-depth", type=int, default=12)
    parser.add_argument(
        "--time-control",
        choices=("clock", "movetime"),
        default="clock",
        help="Use clock for uniform 30s+1s benchmarking, or movetime for quick smoke tests.",
    )
    parser.add_argument("--clock-ms", type=int, default=30_000, help="Initial clock per side for clock mode.")
    parser.add_argument("--increment-ms", type=int, default=1_000, help="Increment per move for clock mode.")
    parser.add_argument("--native-movetime-ms", type=int, default=100)
    parser.add_argument("--stockfish-movetime-ms", type=int, default=100)
    parser.add_argument("--max-ply", type=int, default=160)
    parser.add_argument("--report-log", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--no-report", action="store_true", help="Print only; do not append JSONL report.")
    parser.add_argument("--analysis-games", type=int, default=40, help="Native self-play games for analysis metrics.")
    parser.add_argument("--analysis-depth", type=int, default=10)
    parser.add_argument("--analysis-movetime-ms", type=int, default=150)
    parser.add_argument("--analysis-dashboard", type=Path, default=DEFAULT_ANALYSIS_DASHBOARD)
    parser.add_argument("--no-analysis-dashboard", action="store_true", help="Skip the analysis dashboard.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.engine_lib.exists():
        raise FileNotFoundError(f"missing native engine library: {args.engine_lib}")
    if not args.model.exists():
        raise FileNotFoundError(f"missing NNUE model: {args.model}")

    ladder = parse_ladder(args.stockfish_elos)
    if not ladder:
        raise ValueError("empty --stockfish-elos ladder")
    if args.games_per_level <= 0:
        raise ValueError("--games-per-level must be positive")
    if args.native_depth <= 0:
        raise ValueError("--native-depth must be positive")
    if args.max_ply <= 0:
        raise ValueError("--max-ply must be positive")
    if args.analysis_games < 0:
        raise ValueError("--analysis-games must be non-negative")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    analysis_summary = None
    analysis_counts = None
    if not args.no_analysis_dashboard:
        analysis_summary, analysis_counts = run_analysis(args, run_id)

    results: list[EloResult] = []
    print(f"[elo-tracker] run={run_id} model={args.model}", flush=True)
    print(f"[elo-tracker] ladder={','.join(str(level) for level in ladder)}", flush=True)

    for level in ladder:
        estimator = EloEstimator(
            root=ROOT,
            engine_lib=args.engine_lib,
            model_path=args.model,
            stockfish_bin=args.stockfish_bin,
            stockfish_elo=level,
            games=args.games_per_level,
            native_depth=args.native_depth,
            native_movetime_ms=args.native_movetime_ms,
            stockfish_movetime_ms=args.stockfish_movetime_ms,
            max_ply=args.max_ply,
            time_control=args.time_control,
            clock_ms=args.clock_ms,
            increment_ms=args.increment_ms,
        )
        result = estimator.run()
        results.append(result)
        print(
            f"[elo-tracker] vs {level}: score={result.score:.3f} "
            f"single-level-est={result.estimated_elo:.0f} "
            f"95% CI={result.estimated_elo_ci_low:.0f}..{result.estimated_elo_ci_high:.0f} "
            f"({result.wins}W/{result.draws}D/{result.losses}L)",
            flush=True,
        )

    fit = fit_ladder(results)
    print(
        f"[elo-tracker] fitted Elo={fit.estimated_elo} "
        f"95% CI={fit.ci_low}..{fit.ci_high} "
        f"total={fit.total_wins}W/{fit.total_draws}D/{fit.total_losses}L "
        f"score={fit.total_score:.3f}",
        flush=True,
    )
    print(f"[elo-tracker] note: {fit.note}", flush=True)

    if not args.no_report:
        write_report(args.report_log, args, run_id, results, fit, analysis_summary, analysis_counts)
        print(f"[elo-tracker] wrote report -> {args.report_log}", flush=True)

    return 0


def run_analysis(args: argparse.Namespace, run_id: str) -> tuple[dict[str, object], dict[str, dict[str, int]]]:
    tracker = AnalysisTracker()
    termination_counts: Counter[str] = Counter()
    result_counts: Counter[str] = Counter()
    if args.analysis_games > 0:
        engine = ChessEngine(str(args.engine_lib), str(args.model))
        self_play = SelfPlayEngine(
            engine=engine,
            reward_engine=RewardEngine(),
            game_logger=None,
            max_ply=args.max_ply,
            time_limit_per_move_ms=args.analysis_movetime_ms,
            max_search_depth=args.analysis_depth,
        )
        print(
            f"[analysis] games={args.analysis_games} depth={args.analysis_depth} "
            f"movetime_ms={args.analysis_movetime_ms}",
            flush=True,
        )
        for game_index in range(1, args.analysis_games + 1):
            history, result, reason = self_play.execute_match()
            tracker.add_game(history, result)
            termination_counts[reason] += 1
            result_counts[f"{result:.1f}"] += 1
            print(
                f"[analysis] game {game_index}/{args.analysis_games}: "
                f"{len(history)} positions, result={result:.1f}, reason={reason}",
                flush=True,
            )

    summary = asdict(tracker.finish())
    dashboard_path = tracker.write_dashboard(
        args.analysis_dashboard,
        run_id=run_id,
        config={
            "model": str(args.model),
            "engine_lib": str(args.engine_lib),
            "games": args.analysis_games,
            "max_ply": args.max_ply,
            "depth": args.analysis_depth,
            "movetime_ms": args.analysis_movetime_ms,
        },
    )
    print(f"[analysis] wrote dashboard metrics -> {dashboard_path}", flush=True)
    counts = {
        "termination_counts": dict(termination_counts),
        "result_counts": dict(result_counts),
    }
    return summary, counts


def parse_ladder(raw: str) -> list[int]:
    return sorted({int(part.strip()) for part in raw.split(",") if part.strip()})


def fit_ladder(results: list[EloResult]) -> LadderFit:
    total_games = sum(result.games for result in results)
    total_wins = sum(result.wins for result in results)
    total_draws = sum(result.draws for result in results)
    total_losses = sum(result.losses for result in results)
    total_score = (total_wins + 0.5 * total_draws) / max(1, total_games)

    min_elo = min(result.opponent_elo for result in results) - 800
    max_elo = max(result.opponent_elo for result in results) + 800
    scored = [(elo, ladder_log_likelihood(elo, results)) for elo in range(min_elo, max_elo + 1)]
    best_elo, best_ll = max(scored, key=lambda item: item[1])
    cutoff = best_ll - 1.920729
    inside = [elo for elo, ll in scored if ll >= cutoff]
    ci_low = min(inside) if inside else min_elo
    ci_high = max(inside) if inside else max_elo
    return LadderFit(
        estimated_elo=best_elo,
        ci_low=ci_low,
        ci_high=ci_high,
        confidence=0.95,
        total_games=total_games,
        total_wins=total_wins,
        total_draws=total_draws,
        total_losses=total_losses,
        total_score=total_score,
        note=(
            "Approximate local Elo under these exact engine, Stockfish cap, "
            "native depth, movetime, max-ply, and adjudication settings. Use the same "
            "settings before/after training for the most meaningful comparison."
        ),
    )


def ladder_log_likelihood(engine_elo: int, results: list[EloResult]) -> float:
    ll = 0.0
    for result in results:
        expected = expected_score(engine_elo, result.opponent_elo)
        win_p = max(1e-9, min(1.0 - 1e-9, expected))
        loss_p = max(1e-9, min(1.0 - 1e-9, 1.0 - expected))
        ll += result.wins * math.log(win_p)
        ll += result.losses * math.log(loss_p)
        ll += result.draws * 0.5 * (math.log(win_p) + math.log(loss_p))
    return ll


def expected_score(engine_elo: float, opponent_elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((opponent_elo - engine_elo) / 400.0))


def write_report(
    path: Path,
    args: argparse.Namespace,
    run_id: str,
    results: list[EloResult],
    fit: LadderFit,
    analysis_summary: dict[str, object] | None,
    analysis_counts: dict[str, dict[str, int]] | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "run_id": run_id,
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "model": str(args.model),
        "engine_lib": str(args.engine_lib),
        "stockfish_bin": str(args.stockfish_bin),
        "settings": {
            "stockfish_elos": [result.opponent_elo for result in results],
            "games_per_level": args.games_per_level,
            "native_depth": args.native_depth,
            "native_movetime_ms": args.native_movetime_ms,
            "stockfish_movetime_ms": args.stockfish_movetime_ms,
            "time_control": args.time_control,
            "clock_ms": args.clock_ms,
            "increment_ms": args.increment_ms,
            "max_ply": args.max_ply,
        },
        "levels": [asdict(result) for result in results],
        "fit": asdict(fit),
    }
    if analysis_summary is not None:
        row["analysis"] = {
            "settings": {
                "games": args.analysis_games,
                "depth": args.analysis_depth,
                "movetime_ms": args.analysis_movetime_ms,
                "max_ply": args.max_ply,
                "dashboard": str(args.analysis_dashboard),
            },
            "summary": analysis_summary,
            **(analysis_counts or {}),
        }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
