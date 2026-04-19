#!/usr/bin/env python3
"""
check_histogram_gate.py — CI gate: assert histogram_ms does not regress between
two stage-profile JSON snapshots produced by csv_train (built with
-DCATBOOST_MLX_STAGE_PROFILE=ON).

Gate config (Sprint 19, updated from Sprint 17 S16-06): N=50000, RMSE, depth=6,
128 bins, 50 iterations.  Reference baseline:
  .cache/profiling/sprint19/baseline/50000_rmse_d6_128bins.json
  (Sprint 18 after-JSON; mean histogram_ms ≈ 15.46 ms; Sprint 18 simdHist kernel)
Regression threshold: >5% wall-clock increase fails the gate (S16-06 standing rule).

Single-config mode
------------------
    python benchmarks/check_histogram_gate.py \
        --before .cache/profiling/sprint19/baseline/50000_rmse_d6_128bins.json \
        --after  .cache/profiling/sprint19/after/50000_rmse_d6_128bins.json \
        --max-regression 0.05

    Exit 0  if:
        (after_mean - before_mean) / before_mean <= --max-regression
        AND (if --min-reduction > 0)
            (before_mean - after_mean) / before_mean >= --min-reduction

    When --min-reduction is 0 (the default), the reduction gate is skipped
    and only the max-regression ceiling applies.

18-config mode
--------------
    python benchmarks/check_histogram_gate.py \
        --18config \
        --before-dir .cache/profiling/sprint19/baseline/ \
        --after-dir  .cache/profiling/sprint19/after/ \
        --max-regression 0.05

    Matches JSONs by filename across --before-dir and --after-dir.
    Reports a per-config delta table to stdout.
    Exit 0 only if NO config regresses histogram_ms more than --max-regression.

JSON input formats (per-file)
-----------------------------
Two formats are accepted; the gate auto-detects by key presence.

Format A — raw profiler (TStageProfiler::WriteJson, csv_train binary):
{
  "meta": { ... },
  "stage_names": [ ... ],
  "iterations": [
    { "iter": 0, "histogram_ms": 308.2, ... },
    ...
  ]
}
histogram_ms values are in milliseconds.

Format B — bench driver (bench_mlx_vs_cpu.py --mlx-stage-profile --output):
{
  "meta": { ... },
  "runs": [
    {
      "task": "RMSE", "scale": 10000, "bins": 128,
      "stage_timings": {
        "histogram_ms": { "mean_s": 0.308, "total_s": ..., "per_iter_s": [...] },
        ...
      }
    }
  ]
}
The per_iter_s values are in seconds; this tool converts to ms.
When a file has multiple runs, all histogram_ms per-iter values are pooled
into a single mean (i.e. the file is treated as one aggregate measurement).

The gate summarises each file as mean(histogram_ms) across all recorded
iterations (iter 0 included — warmup iteration is present in all Sprint 17
baselines and is kept for consistency with Sprint 16 summary methodology).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

def _load_mean_histogram_ms(path: Path) -> float:
    """Return mean histogram_ms (in ms) across all recorded iterations in a
    stage-profile JSON file.

    Accepts two formats — see module docstring for schema details:
      - Format A: raw profiler output (TStageProfiler::WriteJson) with an
        "iterations" array, histogram_ms values in ms.
      - Format B: bench driver output (bench_mlx_vs_cpu.py) with a "runs"
        array whose entries carry stage_timings.histogram_ms.per_iter_s
        values in seconds (converted to ms here).

    Raises FileNotFoundError or ValueError with a descriptive message if the
    file is missing or malformed.
    """
    try:
        with open(path) as fh:
            data = json.load(fh)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON parse error in {path}: {exc}")

    # ---- Format A: raw profiler (has "iterations" key) ----
    if "iterations" in data:
        iterations = data["iterations"]
        if not isinstance(iterations, list) or len(iterations) == 0:
            raise ValueError(f"'iterations' is empty in {path}")
        values = []
        for i, rec in enumerate(iterations):
            if "histogram_ms" not in rec:
                raise ValueError(
                    f"'histogram_ms' key missing from iteration {i} in {path}"
                )
            values.append(float(rec["histogram_ms"]))
        return sum(values) / len(values)

    # ---- Format B: bench driver output (has "runs" key) ----
    if "runs" in data:
        runs = data["runs"]
        if not isinstance(runs, list) or len(runs) == 0:
            raise ValueError(f"'runs' is empty in {path}")
        values = []
        for r_idx, run in enumerate(runs):
            st = run.get("stage_timings")
            if st is None:
                raise ValueError(
                    f"run {r_idx} in {path} has no 'stage_timings' — "
                    "was --mlx-stage-profile passed to bench_mlx_vs_cpu.py?"
                )
            hist = st.get("histogram_ms")
            if hist is None:
                raise ValueError(
                    f"'stage_timings.histogram_ms' missing from run {r_idx} in {path}"
                )
            per_iter = hist.get("per_iter_s")
            if not isinstance(per_iter, list) or len(per_iter) == 0:
                raise ValueError(
                    f"'stage_timings.histogram_ms.per_iter_s' is missing or empty "
                    f"in run {r_idx} of {path}"
                )
            # per_iter_s is in seconds; convert to ms.
            values.extend(float(v) * 1000.0 for v in per_iter)
        return sum(values) / len(values)

    raise ValueError(
        f"Unrecognised JSON format in {path}: expected either 'iterations' "
        "(raw profiler) or 'runs' (bench driver) as top-level key."
    )


# ---------------------------------------------------------------------------
# Gate logic
# ---------------------------------------------------------------------------

def _check_pair(
    before_ms: float,
    after_ms: float,
    min_reduction: float,
    max_regression: float,
    label: str,
) -> Optional[str]:
    """Evaluate the gate for one (before, after) pair.

    Returns None on pass; returns a human-readable failure message on fail.
    """
    if before_ms <= 0.0:
        return f"{label}: before histogram_ms is zero or negative ({before_ms:.3f}) — baseline is corrupt"

    delta_rel = (after_ms - before_ms) / before_ms   # positive = regression
    reduction = -delta_rel                             # positive = improvement

    if delta_rel > max_regression:
        return (
            f"{label}: HISTOGRAM REGRESSION\n"
            f"  Before : {before_ms:.3f} ms\n"
            f"  After  : {after_ms:.3f} ms\n"
            f"  Delta  : {delta_rel:+.1%}  (limit: +{max_regression:.0%})\n"
            f"  Action : Investigate histogram kernel regression before merging.\n"
            f"           Re-run with --mlx-stage-profile to confirm, then check\n"
            f"           catboost/mlx/kernels/kernel_sources.h for recent changes."
        )

    if min_reduction > 0.0 and reduction < min_reduction:
        return (
            f"{label}: INSUFFICIENT REDUCTION\n"
            f"  Before : {before_ms:.3f} ms\n"
            f"  After  : {after_ms:.3f} ms\n"
            f"  Delta  : {delta_rel:+.1%}\n"
            f"  Required reduction: {min_reduction:.0%}  (actual: {reduction:.1%})\n"
            f"  Action : Sprint acceptance criterion not met (see sprint HANDOFF.md for G1)."
        )

    return None


# ---------------------------------------------------------------------------
# Single-config mode
# ---------------------------------------------------------------------------

def _run_single(args: argparse.Namespace) -> int:
    before_path = Path(args.before)
    after_path = Path(args.after)

    try:
        before_ms = _load_mean_histogram_ms(before_path)
        after_ms = _load_mean_histogram_ms(after_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    delta_rel = (after_ms - before_ms) / before_ms
    print(
        f"Before : {before_ms:.3f} ms  ({before_path.name})\n"
        f"After  : {after_ms:.3f} ms  ({after_path.name})\n"
        f"Delta  : {delta_rel:+.1%}"
    )

    failure = _check_pair(
        before_ms=before_ms,
        after_ms=after_ms,
        min_reduction=args.min_reduction,
        max_regression=args.max_regression,
        label=f"{before_path.name} vs {after_path.name}",
    )

    if failure:
        print(f"\n{failure}", file=sys.stderr)
        return 1

    print(
        f"\nOK: histogram_ms delta {delta_rel:+.1%} is within limits "
        f"(max_regression=+{args.max_regression:.0%}, "
        f"min_reduction={args.min_reduction:.0%})."
    )
    return 0


# ---------------------------------------------------------------------------
# 18-config mode
# ---------------------------------------------------------------------------

def _run_18config(args: argparse.Namespace) -> int:
    before_dir = Path(args.before_dir)
    after_dir = Path(args.after_dir)

    if not before_dir.is_dir():
        print(f"ERROR: --before-dir does not exist: {before_dir}", file=sys.stderr)
        return 1
    if not after_dir.is_dir():
        print(f"ERROR: --after-dir does not exist: {after_dir}", file=sys.stderr)
        return 1

    before_files = sorted(before_dir.glob("*.json"))
    if not before_files:
        print(f"ERROR: No JSON files found in --before-dir: {before_dir}", file=sys.stderr)
        return 1

    # Match after-files by filename.
    rows = []
    load_errors = []
    for bf in before_files:
        af = after_dir / bf.name
        if not af.exists():
            load_errors.append(f"After-file missing for {bf.name}: expected {af}")
            continue
        try:
            before_ms = _load_mean_histogram_ms(bf)
            after_ms = _load_mean_histogram_ms(af)
        except (FileNotFoundError, ValueError) as exc:
            load_errors.append(str(exc))
            continue
        delta_rel = (after_ms - before_ms) / before_ms
        rows.append((bf.name, before_ms, after_ms, delta_rel))

    if load_errors:
        for err in load_errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1

    if not rows:
        print("ERROR: No configs could be compared.", file=sys.stderr)
        return 1

    # Print delta table.
    col_w = max(len(r[0]) for r in rows)
    header = (
        f"{'Config':<{col_w}}  {'Before (ms)':>12}  {'After (ms)':>10}  {'Delta':>8}  Status"
    )
    print(header)
    print("-" * len(header))

    failures = []
    for name, before_ms, after_ms, delta_rel in rows:
        failure = _check_pair(
            before_ms=before_ms,
            after_ms=after_ms,
            min_reduction=args.min_reduction,
            max_regression=args.max_regression,
            label=name,
        )
        status = "FAIL" if failure else "OK"
        delta_str = f"{delta_rel:+.1%}"
        print(
            f"{name:<{col_w}}  {before_ms:>12.3f}  {after_ms:>10.3f}  {delta_str:>8}  {status}"
        )
        if failure:
            failures.append(failure)

    print()
    if failures:
        print(f"{len(failures)} config(s) failed the histogram gate:\n", file=sys.stderr)
        for msg in failures:
            print(msg, file=sys.stderr)
            print(file=sys.stderr)
        return 1

    print(
        f"OK: all {len(rows)} configs within max_regression=+{args.max_regression:.0%}."
    )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "CI gate: assert histogram_ms does not regress (and optionally meets "
            "a minimum reduction target) between two stage-profile JSON snapshots."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--18config",
        dest="mode_18config",
        action="store_true",
        default=False,
        help=(
            "Compare a directory of before JSONs against a directory of after JSONs "
            "(matched by filename). Reports a per-config delta table."
        ),
    )

    # Single-config arguments.
    parser.add_argument(
        "--before",
        default=None,
        help="Path to stage-profile JSON captured before the change (single-config mode).",
    )
    parser.add_argument(
        "--after",
        default=None,
        help="Path to stage-profile JSON captured after the change (single-config mode).",
    )

    # 18-config arguments.
    parser.add_argument(
        "--before-dir",
        default=None,
        help="Directory containing before-snapshot JSONs (--18config mode).",
    )
    parser.add_argument(
        "--after-dir",
        default=None,
        help="Directory containing after-snapshot JSONs (--18config mode).",
    )

    # Shared thresholds.
    parser.add_argument(
        "--min-reduction",
        type=float,
        default=0.00,
        metavar="FRAC",
        help=(
            "Minimum required fractional reduction in histogram_ms "
            "((before - after) / before). Default: 0.00 (no minimum). "
            "Use 0.35 to enforce the Sprint 18 S18-G1 acceptance criterion."
        ),
    )
    parser.add_argument(
        "--max-regression",
        type=float,
        default=0.05,
        metavar="FRAC",
        help=(
            "Maximum allowed fractional increase in histogram_ms "
            "((after - before) / before). Default: 0.05 (5%% regression limit)."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.mode_18config:
        if args.before_dir is None or args.after_dir is None:
            print(
                "ERROR: --18config requires --before-dir and --after-dir.",
                file=sys.stderr,
            )
            sys.exit(1)
        sys.exit(_run_18config(args))
    else:
        if args.before is None or args.after is None:
            print(
                "ERROR: Single-config mode requires --before and --after.\n"
                "       Use --18config for directory comparison.",
                file=sys.stderr,
            )
            sys.exit(1)
        sys.exit(_run_single(args))


if __name__ == "__main__":
    main()
