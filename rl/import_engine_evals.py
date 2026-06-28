#!/usr/bin/env python3
"""Convert external engine-eval JSON/JSONL dumps into SOC NNUE FEN rows.

The native trainer loads rows shaped as:

    <fen>,<centipawn evaluation from White's perspective>

This importer accepts records like:

    {
      "fen": "8/8/8/8/8/8/8/8 w - -",
      "evals": [
        {
          "knodes": 1234,
          "depth": 20,
          "pvs": [{"cp": 34, "line": "e2e4 e7e5"}]
        }
      ]
    }

JSONL, .jsonl.zst, a JSON array, or a JSON object containing a top-level
"positions" or "data" list are supported. For huge Lichess dumps, keep the
default streaming mode and use --max-rows to select a subset.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, TextIO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uci import normalize_fen


DEFAULT_OUTPUT = ROOT / "data" / "fen_files" / "external_engine_evals.fen"


@dataclass(frozen=True)
class ConvertedRow:
    fen: str
    white_cp: float
    depth: int
    knodes: int


@dataclass
class ImportStats:
    seen_records: int = 0
    converted_rows: int = 0
    written_rows: int = 0
    train_rows: int = 0
    val_rows: int = 0
    skipped_missing_fen: int = 0
    skipped_bad_fen: int = 0
    skipped_no_eval: int = 0
    skipped_no_score: int = 0
    skipped_abs_cp_low: int = 0
    skipped_abs_cp_high: int = 0
    skipped_mate: int = 0
    mate_rows: int = 0
    cp_rows: int = 0
    clamped_rows: int = 0
    min_depth: int = 0
    max_depth: int = 0
    max_knodes: int = 0
    target_sum: float = 0.0
    target_abs_sum: float = 0.0

    def observe_row(self, row: ConvertedRow, *, was_mate: bool, was_clamped: bool) -> None:
        self.converted_rows += 1
        if was_mate:
            self.mate_rows += 1
        else:
            self.cp_rows += 1
        if was_clamped:
            self.clamped_rows += 1
        if self.converted_rows == 1:
            self.min_depth = row.depth
            self.max_depth = row.depth
        else:
            self.min_depth = min(self.min_depth, row.depth)
            self.max_depth = max(self.max_depth, row.depth)
        self.max_knodes = max(self.max_knodes, row.knodes)
        self.target_sum += row.white_cp
        self.target_abs_sum += abs(row.white_cp)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert engine-eval JSON/JSONL records into SOC supervised FEN rows."
    )
    parser.add_argument("input", type=Path, help="Input JSON, JSONL, or JSONL.ZST eval dump.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--append", action="store_true", help="Append instead of replacing the output file.")
    parser.add_argument("--max-rows", type=int, default=0, help="Stop after this many converted rows; 0 means all.")
    parser.add_argument(
        "--selection",
        choices=("first", "reservoir"),
        default="reservoir",
        help="How to choose --max-rows from a stream. Reservoir samples across the whole file.",
    )
    parser.add_argument("--seed", type=int, default=20240627, help="Seed used by reservoir/split sampling.")
    parser.add_argument("--min-depth", type=int, default=0, help="Skip evals below this depth.")
    parser.add_argument(
        "--pv-index",
        type=int,
        default=0,
        help="PV to use from each selected eval. 0 means the main PV.",
    )
    parser.add_argument(
        "--score-pov",
        choices=("stm", "white"),
        default="stm",
        help="Whether cp/mate scores are from side-to-move or White perspective.",
    )
    parser.add_argument(
        "--mate-cp",
        type=float,
        default=4000.0,
        help="Centipawn target used for mate scores before clamping.",
    )
    parser.add_argument(
        "--min-abs-cp",
        type=float,
        default=0.0,
        help="Keep only rows with abs(target cp) at least this value after POV conversion.",
    )
    parser.add_argument(
        "--max-abs-cp",
        type=float,
        default=4000.0,
        help="Keep only rows with abs(target cp) at most this value after POV conversion.",
    )
    parser.add_argument(
        "--exclude-mates",
        action="store_true",
        help="Skip rows where the selected PV has a mate score.",
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Keep only the deepest eval for each FEN. Uses memory proportional to unique FENs.",
    )
    parser.add_argument(
        "--val-output",
        type=Path,
        default=None,
        help="Optional validation output. When set, rows are split between --output and this file.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.0,
        help="Fraction of rows written to --val-output. Example: 0.01 for 1%% validation.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1_000_000,
        help="Print progress every N converted rows. 0 disables progress logs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_rows < 0:
        raise ValueError("--max-rows must be non-negative")
    if args.min_depth < 0:
        raise ValueError("--min-depth must be non-negative")
    if args.pv_index < 0:
        raise ValueError("--pv-index must be non-negative")
    if args.mate_cp <= 0:
        raise ValueError("--mate-cp must be positive")
    if args.min_abs_cp < 0:
        raise ValueError("--min-abs-cp must be non-negative")
    if args.max_abs_cp < args.min_abs_cp:
        raise ValueError("--max-abs-cp must be greater than or equal to --min-abs-cp")
    if not 0.0 <= args.val_fraction < 1.0:
        raise ValueError("--val-fraction must be in [0.0, 1.0)")
    if args.val_fraction > 0.0 and args.val_output is None:
        raise ValueError("--val-fraction requires --val-output")
    if args.progress_every < 0:
        raise ValueError("--progress-every must be non-negative")

    rng = random.Random(args.seed)
    stats = ImportStats()

    converted = iter_converted_rows(
        load_records(args.input),
        min_depth=args.min_depth,
        pv_index=args.pv_index,
        score_pov=args.score_pov,
        mate_cp=args.mate_cp,
        min_abs_cp=args.min_abs_cp,
        max_abs_cp=args.max_abs_cp,
        exclude_mates=args.exclude_mates,
        stats=stats,
    )
    if args.dedupe:
        rows = list(best_rows_by_fen(converted, max_rows=args.max_rows).values())
        rows.sort(key=lambda row: (row.fen, -row.depth))
        write_result = write_rows(
            args.output,
            rows,
            append=args.append,
            val_output=args.val_output,
            val_fraction=args.val_fraction,
            rng=rng,
            progress_every=args.progress_every,
        )
    elif args.selection == "reservoir" and args.max_rows:
        rows = reservoir_sample(converted, args.max_rows, rng=rng, progress_every=args.progress_every)
        write_result = write_rows(
            args.output,
            rows,
            append=args.append,
            val_output=args.val_output,
            val_fraction=args.val_fraction,
            rng=rng,
            progress_every=args.progress_every,
        )
    else:
        write_result = write_rows(
            args.output,
            limit_rows(converted, args.max_rows),
            append=args.append,
            val_output=args.val_output,
            val_fraction=args.val_fraction,
            rng=rng,
            progress_every=args.progress_every,
        )
    stats.written_rows = write_result["written"]
    stats.train_rows = write_result["train"]
    stats.val_rows = write_result["val"]
    print(
        f"[import-evals] wrote {stats.train_rows} train rows -> {args.output} "
        f"and {stats.val_rows} val rows"
        f"{f' -> {args.val_output}' if args.val_output else ''} "
        f"(mode={'append' if args.append else 'replace'})",
        flush=True,
    )
    print_summary(stats)
    return 0


def load_records(path: Path) -> Iterable[dict[str, Any]]:
    with open_text(path) as handle:
        if path.suffix == ".zst":
            yield from load_jsonl_records(handle)
            return

        first = handle.read(1)
        handle.seek(0)
        if first == "[":
            data = json.load(handle)
            if not isinstance(data, list):
                raise ValueError("top-level JSON array did not decode to a list")
            yield from _dict_records(data)
            return
        if first == "{":
            first_line = handle.readline()
            handle.seek(0)
            if _looks_like_jsonl_object(first_line):
                yield from load_jsonl_records(handle)
                return

            data = json.load(handle)
            if isinstance(data, dict):
                for key in ("positions", "data"):
                    records = data.get(key)
                    if isinstance(records, list):
                        yield from _dict_records(records)
                        return
                yield data
                return
            raise ValueError("unsupported JSON top-level value")

        yield from load_jsonl_records(handle)


def open_text(path: Path) -> TextIO:
    if path.suffix == ".zst":
        try:
            import zstandard
        except ImportError as exc:
            raise RuntimeError(
                "reading .zst files needs the Python package 'zstandard'. "
                "Install it with: python -m pip install zstandard"
            ) from exc
        compressed = path.open("rb")
        reader = zstandard.ZstdDecompressor().stream_reader(compressed)
        return _ZstdTextWrapper(reader, compressed)
    if path.suffix == ".zip":
        archive = zipfile.ZipFile(path)
        members = [info for info in archive.infolist() if not info.is_dir()]
        if not members:
            archive.close()
            raise ValueError(f"zip archive has no files: {path}")
        raw = archive.open(members[0], "r")
        return _ZipTextWrapper(raw, archive)
    return path.open("r", encoding="utf-8")


def load_jsonl_records(handle: TextIO) -> Iterable[dict[str, Any]]:
    for line_number, line in enumerate(handle, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at line {line_number}: {exc}") from exc
        if isinstance(record, dict):
            yield record


def iter_converted_rows(
    records: Iterable[dict[str, Any]],
    *,
    min_depth: int,
    pv_index: int,
    score_pov: str,
    mate_cp: float,
    min_abs_cp: float,
    max_abs_cp: float,
    exclude_mates: bool,
    stats: ImportStats,
) -> Iterable[ConvertedRow]:
    for record in records:
        stats.seen_records += 1
        raw_fen = record.get("fen")
        if not isinstance(raw_fen, str):
            stats.skipped_missing_fen += 1
            continue
        try:
            fen = normalize_fen(raw_fen)
        except ValueError:
            stats.skipped_bad_fen += 1
            continue

        selected = select_eval(record.get("evals"), min_depth=min_depth, pv_index=pv_index)
        if selected is None:
            stats.skipped_no_eval += 1
            continue
        eval_row, pv = selected
        score, was_mate = score_from_pv(pv, mate_cp=mate_cp)
        if score is None:
            stats.skipped_no_score += 1
            continue
        if was_mate and exclude_mates:
            stats.skipped_mate += 1
            continue

        parts = fen.split()
        if score_pov == "stm" and parts[1] == "b":
            score = -score
        clamped_score = max(-4000.0, min(4000.0, float(score)))
        abs_cp = abs(clamped_score)
        if abs_cp < min_abs_cp:
            stats.skipped_abs_cp_low += 1
            continue
        if abs_cp > max_abs_cp:
            stats.skipped_abs_cp_high += 1
            continue

        row = ConvertedRow(
            fen=fen,
            white_cp=clamped_score,
            depth=int(eval_row.get("depth", 0) or 0),
            knodes=int(eval_row.get("knodes", 0) or 0),
        )
        stats.observe_row(row, was_mate=was_mate, was_clamped=clamped_score != float(score))
        yield row


def select_eval(
    evals: Any,
    *,
    min_depth: int,
    pv_index: int,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    if not isinstance(evals, list):
        return None

    candidates: list[tuple[int, int, dict[str, Any], dict[str, Any]]] = []
    for eval_row in evals:
        if not isinstance(eval_row, dict):
            continue
        depth = int(eval_row.get("depth", 0) or 0)
        if depth < min_depth:
            continue
        pvs = eval_row.get("pvs")
        if not isinstance(pvs, list) or pv_index >= len(pvs):
            continue
        pv = pvs[pv_index]
        if isinstance(pv, dict):
            candidates.append((depth, int(eval_row.get("knodes", 0) or 0), eval_row, pv))

    if not candidates:
        return None
    _, _, eval_row, pv = max(candidates, key=lambda item: (item[0], item[1]))
    return eval_row, pv


def score_from_pv(pv: dict[str, Any], *, mate_cp: float) -> tuple[float | None, bool]:
    if "cp" in pv:
        try:
            return float(pv["cp"]), False
        except (TypeError, ValueError):
            return None, False
    if "mate" in pv:
        try:
            mate = int(pv["mate"])
        except (TypeError, ValueError):
            return None, True
        return (mate_cp if mate > 0 else -mate_cp), True
    return None, False


def best_rows_by_fen(rows: Iterable[ConvertedRow], *, max_rows: int = 0) -> dict[str, ConvertedRow]:
    best: dict[str, ConvertedRow] = {}
    for row in rows:
        previous = best.get(row.fen)
        if previous is None or (row.depth, row.knodes) > (previous.depth, previous.knodes):
            best[row.fen] = row
        if max_rows and len(best) >= max_rows:
            break
    return best


def limit_rows(rows: Iterable[ConvertedRow], max_rows: int) -> Iterable[ConvertedRow]:
    for count, row in enumerate(rows, start=1):
        if max_rows and count > max_rows:
            break
        yield row


def reservoir_sample(
    rows: Iterable[ConvertedRow],
    max_rows: int,
    *,
    rng: random.Random,
    progress_every: int,
) -> list[ConvertedRow]:
    sample: list[ConvertedRow] = []
    for count, row in enumerate(rows, start=1):
        if len(sample) < max_rows:
            sample.append(row)
        else:
            index = rng.randrange(count)
            if index < max_rows:
                sample[index] = row
        if progress_every and count % progress_every == 0:
            print(
                f"[import-evals] scanned {count} convertible rows; "
                f"reservoir={len(sample)}",
                flush=True,
            )
    rng.shuffle(sample)
    return sample


def write_rows(
    path: Path,
    rows: Iterable[ConvertedRow],
    *,
    append: bool,
    val_output: Path | None,
    val_fraction: float,
    rng: random.Random,
    progress_every: int,
) -> dict[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if val_output is not None:
        val_output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    train_count = 0
    val_count = 0
    with path.open(mode, encoding="utf-8") as train_handle:
        val_handle = val_output.open(mode, encoding="utf-8") if val_output is not None else None
        try:
            for count, row in enumerate(rows, start=1):
                handle = train_handle
                if val_handle is not None and rng.random() < val_fraction:
                    handle = val_handle
                    val_count += 1
                else:
                    train_count += 1
                handle.write(f"{row.fen},{row.white_cp:.2f}\n")
                if progress_every and count % progress_every == 0:
                    print(f"[import-evals] wrote {count} rows...", flush=True)
        finally:
            if val_handle is not None:
                val_handle.close()
    return {"written": train_count + val_count, "train": train_count, "val": val_count}


def print_summary(stats: ImportStats) -> None:
    mean = stats.target_sum / stats.converted_rows if stats.converted_rows else 0.0
    mean_abs = stats.target_abs_sum / stats.converted_rows if stats.converted_rows else 0.0
    print(
        "[import-evals] summary: "
        f"seen={stats.seen_records} converted={stats.converted_rows} "
        f"written={stats.written_rows} train={stats.train_rows} val={stats.val_rows}",
        flush=True,
    )
    print(
        "[import-evals] labels: "
        f"cp={stats.cp_rows} mate={stats.mate_rows} clamped={stats.clamped_rows} "
        f"mean={mean:.2f} mean_abs={mean_abs:.2f} "
        f"depth_min={stats.min_depth} depth_max={stats.max_depth} "
        f"max_knodes={stats.max_knodes}",
        flush=True,
    )
    print(
        "[import-evals] skipped: "
        f"missing_fen={stats.skipped_missing_fen} bad_fen={stats.skipped_bad_fen} "
        f"no_eval={stats.skipped_no_eval} no_score={stats.skipped_no_score} "
        f"abs_cp_low={stats.skipped_abs_cp_low} abs_cp_high={stats.skipped_abs_cp_high} "
        f"mate={stats.skipped_mate}",
        flush=True,
    )


def _dict_records(records: Iterable[Any]) -> Iterable[dict[str, Any]]:
    for record in records:
        if isinstance(record, dict):
            yield record


def _looks_like_jsonl_object(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("{") and stripped.endswith("}") and '"fen"' in stripped


class _ZstdTextWrapper:
    def __init__(self, reader: Any, compressed: Any) -> None:
        self.compressed = compressed
        self.buffer = reader
        self.text = io.TextIOWrapper(reader, encoding="utf-8")

    def __enter__(self) -> TextIO:
        return self.text

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.text.close()
        self.buffer.close()
        self.compressed.close()


class _ZipTextWrapper:
    def __init__(self, raw: Any, archive: zipfile.ZipFile) -> None:
        self.raw = raw
        self.archive = archive
        self.text = io.TextIOWrapper(raw, encoding="utf-8")

    def __enter__(self) -> TextIO:
        return self.text

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.text.close()
        self.raw.close()
        self.archive.close()


if __name__ == "__main__":
    raise SystemExit(main())
