#!/usr/bin/env python3
"""
bench_mlx_vs_cpu.py — CatBoost-MLX (Metal GPU) vs CatBoost CPU benchmark.

Trains both backends across configurable dataset scales, bin counts, and loss
functions. Measures wall-clock training time, iterations/second, and final
train loss. Outputs a markdown table to stdout and optionally a JSON file.

Usage
-----
    python benchmarks/bench_mlx_vs_cpu.py
    python benchmarks/bench_mlx_vs_cpu.py --hardware "M2 Pro 16GB" --iterations 200
    python benchmarks/bench_mlx_vs_cpu.py --scales 10000,100000 --bins 32,128
    python benchmarks/bench_mlx_vs_cpu.py --output results.json --bins 128
    python benchmarks/bench_mlx_vs_cpu.py --save-baseline
    python benchmarks/bench_mlx_vs_cpu.py --mlx-stage-profile --output results.json

JSON output schema
------------------
{
  "meta": { "date", "hardware", "python", "catboost_version", "catboost_mlx_version",
            "iterations", "n_features", "depth", "learning_rate", "l2_leaf_reg",
            "random_seed" },
  "runs": [
    {
      "task":          str,           # "RMSE" | "Logloss" | "MultiClass"
      "scale":         int,           # n_rows
      "bins":          int,           # bin count used for this run
      "cpu_time_s":    float | null,  # wall-clock seconds (CPU backend)
      "mlx_time_s":    float | null,  # wall-clock seconds (MLX backend)
      "speedup":       float | null,  # cpu_time_s / mlx_time_s (>1 = MLX faster)
      "cpu_loss":      float | null,
      "mlx_loss":      float | null,
      "loss_delta":    float | null,  # abs(mlx_loss - cpu_loss)
      "cpu_baseline": {               # raw CPU fields for downstream consumers
        "elapsed_s", "iter_per_s", "train_loss"
      },
      "mlx_baseline": {               # raw MLX fields
        "elapsed_s", "iter_per_s", "train_loss"
      },
      "stage_timings": { ... } | null  # populated when --mlx-stage-profile is set
    },
    ...
  ]
}
"""

import argparse
import datetime
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Backend availability ──────────────────────────────────────────────────────

try:
    from catboost import CatBoostClassifier as CBClassifier
    from catboost import CatBoostRegressor as CBRegressor
    _HAS_CATBOOST_CPU = True
except ImportError:
    _HAS_CATBOOST_CPU = False
    print(
        "WARNING: catboost is not installed — CPU benchmarks will be skipped.\n"
        "         Install with: pip install catboost",
        file=sys.stderr,
    )

try:
    from catboost_mlx import CatBoostMLXClassifier, CatBoostMLXRegressor
    _HAS_CATBOOST_MLX = True
except ImportError:
    _HAS_CATBOOST_MLX = False
    print(
        "WARNING: catboost_mlx is not installed — MLX benchmarks will be skipped.\n"
        "         Build from source: see catboost/mlx/README.md",
        file=sys.stderr,
    )

# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class BenchResult:
    dataset_label: str    # e.g. "10k x 50"
    loss: str             # "RMSE", "Logloss", "MultiClass"
    backend: str          # "CPU" or "MLX"
    elapsed_s: float      # wall-clock seconds
    iter_per_s: float     # iterations / elapsed_s
    train_loss: float     # final train loss value (last iteration)
    bins: int = 128       # bin count used for this run
    stage_timings: Optional[Dict] = field(default=None)


@dataclass
class ParityResult:
    """Side-by-side comparison of CPU and MLX for one (task, scale, bins) triple."""
    task: str
    scale: int
    bins: int
    cpu_time_s: Optional[float]
    mlx_time_s: Optional[float]
    speedup: Optional[float]       # cpu / mlx; >1 means MLX is faster
    cpu_loss: Optional[float]
    mlx_loss: Optional[float]
    loss_delta: Optional[float]    # abs(mlx_loss - cpu_loss)
    cpu_bench: Optional[BenchResult]
    mlx_bench: Optional[BenchResult]


# ── Dataset generation ────────────────────────────────────────────────────────

def _make_dataset(
    n_rows: int,
    n_cols: int,
    loss: str,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic (X, y) pair appropriate for the given loss function.

    RMSE       — continuous targets drawn from a linear model + Gaussian noise.
    Logloss    — binary {0, 1} targets; class balance is ~50 / 50.
    MultiClass — 3-class integer targets {0, 1, 2}; balanced across classes.

    All feature columns are i.i.d. standard normal.
    """
    X = rng.standard_normal((n_rows, n_cols)).astype(np.float32)
    coeff = rng.standard_normal(n_cols).astype(np.float32)

    if loss == "RMSE":
        y = X @ coeff + rng.standard_normal(n_rows).astype(np.float32)

    elif loss == "Logloss":
        logit = X @ coeff
        prob = 1.0 / (1.0 + np.exp(-logit))
        y = rng.binomial(1, prob).astype(np.float32)

    elif loss == "MultiClass":
        # Produce 3 class scores and assign argmax label
        W = rng.standard_normal((n_cols, 3)).astype(np.float32)
        scores = X @ W
        y = np.argmax(scores, axis=1).astype(np.float32)

    else:
        raise ValueError(f"Unknown loss: {loss}")

    return X, y


# ── Warm-up ───────────────────────────────────────────────────────────────────

def _warmup(backend: str, loss: str, iterations: int, rng: np.random.Generator) -> None:
    """Run a tiny training job to absorb JIT / Metal compilation overhead.

    Both backends have a first-call penalty:
    - CatBoost CPU: shared library initialisation and thread pool spin-up.
    - CatBoost-MLX: Metal kernel compilation (cached after first call within
      a process, but not across processes on a cold Metal shader cache).

    Using 200 rows x 10 features keeps warm-up under a few seconds even on
    a cold Metal cache.
    """
    X_warm, y_warm = _make_dataset(200, 10, loss, rng)
    kwargs_common = dict(iterations=min(iterations, 20), depth=4, learning_rate=0.1)

    if backend == "CPU" and _HAS_CATBOOST_CPU:
        if loss == "RMSE":
            m = CBRegressor(**kwargs_common, verbose=0, random_seed=42)
        elif loss == "Logloss":
            m = CBClassifier(
                **kwargs_common, loss_function="Logloss", verbose=0, random_seed=42
            )
        else:
            m = CBClassifier(
                **kwargs_common, loss_function="MultiClass", verbose=0, random_seed=42
            )
        m.fit(X_warm, y_warm)

    elif backend == "MLX" and _HAS_CATBOOST_MLX:
        mlx_loss = {"RMSE": "rmse", "Logloss": "logloss", "MultiClass": "multiclass"}[loss]
        if loss == "RMSE":
            m = CatBoostMLXRegressor(
                **kwargs_common, loss=mlx_loss, random_seed=42, verbose=False
            )
        else:
            m = CatBoostMLXClassifier(
                **kwargs_common, loss=mlx_loss, random_seed=42, verbose=False
            )
        m.fit(X_warm, y_warm)


# ── Final-loss extraction helpers ─────────────────────────────────────────────

def _cpu_final_loss(model, loss: str) -> float:
    """Extract final train-loss value from a fitted CatBoost CPU model.

    CatBoost CPU exposes per-iteration metrics via evals_result_ once the
    model is fitted (no eval_set needed — learn metrics are always recorded).
    The dict structure is: {'learn': {'MetricName': [v0, v1, ..., vN]}}.
    """
    evals = model.evals_result_
    learn = evals.get("learn", {})
    if not learn:
        return float("nan")
    # Take the last value of whichever metric is present
    for val_list in learn.values():
        if val_list:
            return float(val_list[-1])
    return float("nan")


def _mlx_final_loss(model) -> float:
    """Extract final train-loss from a fitted CatBoostMLX model.

    CatBoostMLX exposes the per-iteration train loss via the
    `train_loss_history` property (list of floats, one per iteration).
    """
    history = model.train_loss_history
    if history:
        return float(history[-1])
    return float("nan")


def _mlx_stage_timings(model) -> Optional[Dict]:
    """Extract per-stage timing data from a fitted CatBoostMLX model.

    CatBoostMLX exposes stage timings via `stage_timing_history` when
    --stage-profile was enabled at training time. Returns None if the
    model does not carry timing data (e.g. stage profiling was not requested).

    The returned dict maps stage name -> list of per-iteration timings (seconds).
    Callers should summarise as mean/total as needed.
    """
    if not hasattr(model, "stage_timing_history"):
        return None
    history = model.stage_timing_history
    if not history:
        return None
    # history is a list of dicts (one per iteration); pivot to stage -> [times]
    stage_names = list(history[0].keys()) if history else []
    aggregated: Dict[str, Dict] = {}
    for stage in stage_names:
        values = [float(it.get(stage, float("nan"))) for it in history]
        aggregated[stage] = {
            "mean_s": float(np.nanmean(values)),
            "total_s": float(np.nansum(values)),
            "per_iter_s": values,
        }
    return aggregated


# ── Per-benchmark runner ──────────────────────────────────────────────────────

def _run_one(
    n_rows: int,
    n_cols: int,
    loss: str,
    backend: str,
    iterations: int,
    bins: int,
    rng: np.random.Generator,
    stage_profile: bool = False,
) -> Optional[BenchResult]:
    """Train one (scale, loss, backend, bins) combination and return a BenchResult.

    The `bins` parameter controls the number of histogram bins. CatBoost CPU
    uses `border_count`; CatBoost-MLX uses `n_bins`. Both are passed as-is —
    the interface mirrors what each backend accepts.

    Returns None if the required backend is unavailable.
    """
    dataset_label = f"{n_rows // 1000}k x {n_cols}"

    if backend == "CPU" and not _HAS_CATBOOST_CPU:
        return None
    if backend == "MLX" and not _HAS_CATBOOST_MLX:
        return None

    X, y = _make_dataset(n_rows, n_cols, loss, rng)

    # ---- Build model ----
    # border_count (CPU) / n_bins (MLX) maps to the histogram bin count.
    # Identical hyperparameters across backends ensure a fair comparison.
    common = dict(iterations=iterations, depth=6, learning_rate=0.1)

    if backend == "CPU":
        if loss == "RMSE":
            model = CBRegressor(
                **common,
                border_count=bins,
                l2_leaf_reg=3.0,
                verbose=0,
                random_seed=42,
            )
        elif loss == "Logloss":
            model = CBClassifier(
                **common,
                loss_function="Logloss",
                border_count=bins,
                l2_leaf_reg=3.0,
                verbose=0,
                random_seed=42,
            )
        else:  # MultiClass
            model = CBClassifier(
                **common,
                loss_function="MultiClass",
                border_count=bins,
                l2_leaf_reg=3.0,
                verbose=0,
                random_seed=42,
            )

    else:  # MLX
        mlx_loss = {"RMSE": "rmse", "Logloss": "logloss", "MultiClass": "multiclass"}[loss]
        mlx_kwargs = dict(
            **common,
            n_bins=bins,
            l2_reg_lambda=3.0,
            random_seed=42,
            verbose=False,
        )
        if stage_profile:
            mlx_kwargs["stage_profile"] = True

        if loss == "RMSE":
            model = CatBoostMLXRegressor(loss=mlx_loss, **mlx_kwargs)
        else:
            model = CatBoostMLXClassifier(loss=mlx_loss, **mlx_kwargs)

    # ---- Time the fit ----
    t0 = time.perf_counter()
    model.fit(X, y)
    elapsed = time.perf_counter() - t0

    # ---- Collect metrics ----
    train_loss = _cpu_final_loss(model, loss) if backend == "CPU" else _mlx_final_loss(model)

    # Stage timings are only extracted for the MLX backend and only when requested.
    timings = None
    if backend == "MLX" and stage_profile:
        timings = _mlx_stage_timings(model)

    return BenchResult(
        dataset_label=dataset_label,
        loss=loss,
        backend=backend,
        elapsed_s=elapsed,
        iter_per_s=iterations / elapsed,
        train_loss=train_loss,
        bins=bins,
        stage_timings=timings,
    )


# ── Parity comparison ─────────────────────────────────────────────────────────

def _make_parity_result(
    n_rows: int,
    n_cols: int,
    loss: str,
    bins: int,
    iterations: int,
    rng: np.random.Generator,
    stage_profile: bool = False,
) -> ParityResult:
    """Run both CPU and MLX for one (scale, loss, bins) triple and compute parity.

    The same `rng` state is used for both backends so both train on identical
    data. This is deliberate: we want a fair comparison, not random variance.
    """
    # Snapshot RNG state so both backends start from the same seed.
    rng_state = rng.bit_generator.state

    cpu_result = _run_one(n_rows, n_cols, loss, "CPU", iterations, bins, rng)

    # Restore RNG so MLX gets the identical dataset.
    rng.bit_generator.state = rng_state
    mlx_result = _run_one(
        n_rows, n_cols, loss, "MLX", iterations, bins, rng, stage_profile=stage_profile
    )

    cpu_time = cpu_result.elapsed_s if cpu_result is not None else None
    mlx_time = mlx_result.elapsed_s if mlx_result is not None else None
    cpu_loss = cpu_result.train_loss if cpu_result is not None else None
    mlx_loss = mlx_result.train_loss if mlx_result is not None else None

    speedup = None
    if cpu_time is not None and mlx_time is not None and mlx_time > 0:
        speedup = cpu_time / mlx_time

    loss_delta = None
    if cpu_loss is not None and mlx_loss is not None:
        if not (math.isnan(cpu_loss) or math.isnan(mlx_loss)):
            loss_delta = abs(mlx_loss - cpu_loss)

    return ParityResult(
        task=loss,
        scale=n_rows,
        bins=bins,
        cpu_time_s=cpu_time,
        mlx_time_s=mlx_time,
        speedup=speedup,
        cpu_loss=cpu_loss,
        mlx_loss=mlx_loss,
        loss_delta=loss_delta,
        cpu_bench=cpu_result,
        mlx_bench=mlx_result,
    )


# ── Output formatting ─────────────────────────────────────────────────────────

def _format_parity_table(results: List[ParityResult]) -> str:
    """Render a side-by-side CPU vs MLX comparison as a GitHub markdown table."""
    header = (
        "| Task       | Scale   | Bins | CPU (s) | MLX (s) | Speedup | "
        "CPU Loss  | MLX Loss | Loss Delta |\n"
        "|------------|---------|------|---------|---------|---------|"
        "----------|----------|------------|\n"
    )
    rows = []
    for r in results:
        cpu_t  = f"{r.cpu_time_s:>7.2f}" if r.cpu_time_s is not None else "    n/a"
        mlx_t  = f"{r.mlx_time_s:>7.2f}" if r.mlx_time_s is not None else "    n/a"
        spdup  = f"{r.speedup:>7.2f}x" if r.speedup is not None else "    n/a"
        c_loss = f"{r.cpu_loss:>8.4f}" if r.cpu_loss is not None else "     n/a"
        m_loss = f"{r.mlx_loss:>8.4f}" if r.mlx_loss is not None else "     n/a"
        delta  = f"{r.loss_delta:>10.4f}" if r.loss_delta is not None else "       n/a"
        scale  = f"{r.scale // 1000}k"
        rows.append(
            f"| {r.task:<10} "
            f"| {scale:>7} "
            f"| {r.bins:>4} "
            f"| {cpu_t} "
            f"| {mlx_t} "
            f"| {spdup} "
            f"| {c_loss} "
            f"| {m_loss} "
            f"| {delta} |"
        )
    return header + "\n".join(rows) + "\n"


def _format_header(hardware: str, iterations: int, bin_counts: List[int]) -> str:
    """Format a human-readable header section for the benchmark output."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    py_version = sys.version.split()[0]

    cb_version = "n/a"
    if _HAS_CATBOOST_CPU:
        try:
            import catboost
            cb_version = catboost.__version__
        except Exception:
            pass

    mlx_version = "n/a"
    if _HAS_CATBOOST_MLX:
        try:
            import catboost_mlx
            mlx_version = catboost_mlx.__version__
        except Exception:
            pass

    lines = [
        "# CatBoost-MLX vs CatBoost CPU Benchmark",
        "",
        f"- **Date**: {now}",
        f"- **Hardware**: {hardware}",
        f"- **Python**: {py_version}",
        f"- **catboost**: {cb_version}",
        f"- **catboost_mlx**: {mlx_version}",
        f"- **Iterations**: {iterations}",
        f"- **Bin counts**: {', '.join(str(b) for b in bin_counts)}",
        f"- **Hyperparameters**: depth=6, learning_rate=0.1, l2_leaf_reg=3.0, random_seed=42",
        "",
    ]
    return "\n".join(lines) + "\n"


# ── JSON serialisation ────────────────────────────────────────────────────────

def _build_json_output(
    results: List[ParityResult],
    hardware: str,
    iterations: int,
    bin_counts: List[int],
) -> Dict:
    """Build the full JSON output dict from parity results.

    Schema documented in the module docstring.
    """
    now = datetime.datetime.now().isoformat()
    py_version = sys.version.split()[0]

    cb_version = "n/a"
    if _HAS_CATBOOST_CPU:
        try:
            import catboost
            cb_version = catboost.__version__
        except Exception:
            pass

    mlx_version = "n/a"
    if _HAS_CATBOOST_MLX:
        try:
            import catboost_mlx
            mlx_version = catboost_mlx.__version__
        except Exception:
            pass

    runs = []
    for r in results:
        cpu_baseline = None
        if r.cpu_bench is not None:
            cpu_baseline = {
                "elapsed_s": r.cpu_bench.elapsed_s,
                "iter_per_s": r.cpu_bench.iter_per_s,
                "train_loss": r.cpu_bench.train_loss,
            }

        mlx_baseline = None
        if r.mlx_bench is not None:
            mlx_baseline = {
                "elapsed_s": r.mlx_bench.elapsed_s,
                "iter_per_s": r.mlx_bench.iter_per_s,
                "train_loss": r.mlx_bench.train_loss,
            }

        run_entry = {
            "task": r.task,
            "scale": r.scale,
            "bins": r.bins,
            "cpu_time_s": r.cpu_time_s,
            "mlx_time_s": r.mlx_time_s,
            "speedup": r.speedup,
            "cpu_loss": r.cpu_loss,
            "mlx_loss": r.mlx_loss,
            "loss_delta": r.loss_delta,
            "cpu_baseline": cpu_baseline,
            "mlx_baseline": mlx_baseline,
            "stage_timings": r.mlx_bench.stage_timings if r.mlx_bench is not None else None,
        }
        runs.append(run_entry)

    return {
        "meta": {
            "date": now,
            "hardware": hardware,
            "python": py_version,
            "catboost_version": cb_version,
            "catboost_mlx_version": mlx_version,
            "iterations": iterations,
            "n_features": 50,
            "depth": 6,
            "learning_rate": 0.1,
            "l2_leaf_reg": 3.0,
            "random_seed": 42,
            "bin_counts": bin_counts,
        },
        "runs": runs,
    }


# ── Baseline snapshot ─────────────────────────────────────────────────────────

_BASELINE_PATH = Path(".cache/benchmarks/sprint16_baseline.json")


def _save_baseline(json_output: Dict, path: Path) -> None:
    """Write results to the baseline snapshot file.

    The baseline is used by the CI regression gate (mlx-perf-regression.yaml)
    to detect regressions. The file is committed and never auto-updated by CI.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(json_output, f, indent=2)
    print(f"\nBaseline saved to: {path}", file=sys.stderr)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark CatBoost-MLX (Metal GPU) vs CatBoost CPU.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--hardware",
        default="unknown",
        help='Hardware tag included in the output header (e.g. "M2 Pro 16GB").',
    )
    parser.add_argument(
        "--scales",
        default="10000,100000,500000",
        help=(
            "Comma-separated list of row counts to benchmark. "
            "Default: 10000,100000,500000"
        ),
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of boosting iterations for every model. Default: 100",
    )
    parser.add_argument(
        "--bins",
        default="128",
        help=(
            "Comma-separated list of histogram bin counts to benchmark. "
            "Each (scale, task) combo is run once per bin count. "
            "Default: 128  Example: --bins 32,128"
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Write JSON output to this file path. "
            "If omitted, a markdown table is written to stdout."
        ),
    )
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        default=False,
        help=(
            f"After running, save results to {_BASELINE_PATH}. "
            "This snapshot is used by the CI regression gate."
        ),
    )
    parser.add_argument(
        "--mlx-stage-profile",
        action="store_true",
        default=False,
        help=(
            "Pass stage_profile=True to every MLX model. "
            "Embeds per-stage timing data in the JSON output under 'stage_timings'."
        ),
    )
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    # Parse --scales
    try:
        scales = [int(s.strip()) for s in args.scales.split(",") if s.strip()]
    except ValueError as exc:
        print(f"ERROR: --scales must be comma-separated integers: {exc}", file=sys.stderr)
        sys.exit(1)

    if not scales:
        print("ERROR: --scales produced an empty list.", file=sys.stderr)
        sys.exit(1)

    # Parse --bins
    try:
        bin_counts = [int(b.strip()) for b in args.bins.split(",") if b.strip()]
    except ValueError as exc:
        print(f"ERROR: --bins must be comma-separated integers: {exc}", file=sys.stderr)
        sys.exit(1)

    if not bin_counts:
        print("ERROR: --bins produced an empty list.", file=sys.stderr)
        sys.exit(1)

    n_features = 50
    losses = ["RMSE", "Logloss", "MultiClass"]
    iterations = args.iterations
    stage_profile = args.mlx_stage_profile

    if not _HAS_CATBOOST_CPU and not _HAS_CATBOOST_MLX:
        print(
            "ERROR: Neither catboost nor catboost_mlx is installed. "
            "Nothing to benchmark.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Deterministic RNG for all dataset generation — same X/y across runs.
    rng = np.random.default_rng(42)

    # ---- Warm-up pass ----
    print("Warming up backends...", file=sys.stderr)
    for backend in ["CPU", "MLX"]:
        for loss in losses:
            _warmup(backend, loss, iterations, np.random.default_rng(0))
    print("Warm-up complete.\n", file=sys.stderr)

    # ---- Benchmark runs ----
    parity_results: List[ParityResult] = []
    total_combos = len(scales) * len(losses) * len(bin_counts)
    combo_num = 0

    for bins in bin_counts:
        for n_rows in scales:
            for loss in losses:
                combo_num += 1
                label = f"{n_rows // 1000}k x {n_features}"
                print(
                    f"[{combo_num}/{total_combos}] "
                    f"{label}  loss={loss}  bins={bins} ...",
                    end="  ",
                    file=sys.stderr,
                    flush=True,
                )
                try:
                    parity = _make_parity_result(
                        n_rows=n_rows,
                        n_cols=n_features,
                        loss=loss,
                        bins=bins,
                        iterations=iterations,
                        rng=rng,
                        stage_profile=stage_profile,
                    )
                except Exception as exc:
                    print(f"FAILED: {exc}", file=sys.stderr)
                    continue

                parity_results.append(parity)

                # Progress line: show both backend times if available
                cpu_t = f"CPU={parity.cpu_time_s:.2f}s" if parity.cpu_time_s is not None else "CPU=n/a"
                mlx_t = f"MLX={parity.mlx_time_s:.2f}s" if parity.mlx_time_s is not None else "MLX=n/a"
                spd   = f"  {parity.speedup:.2f}x" if parity.speedup is not None else ""
                print(f"{cpu_t}  {mlx_t}{spd}", file=sys.stderr)

    if not parity_results:
        print("No benchmark results collected. Exiting.", file=sys.stderr)
        sys.exit(1)

    # ---- Build JSON payload ----
    json_output = _build_json_output(parity_results, args.hardware, iterations, bin_counts)

    # ---- Save baseline if requested ----
    if args.save_baseline:
        _save_baseline(json_output, _BASELINE_PATH)

    # ---- Emit output ----
    if args.output:
        with open(args.output, "w") as f:
            json.dump(json_output, f, indent=2)
        print(f"\nResults written to: {args.output}", file=sys.stderr)
    else:
        # Fall back to markdown table on stdout when no --output is given.
        header = _format_header(args.hardware, iterations, bin_counts)
        table = _format_parity_table(parity_results)
        print(header + table)


if __name__ == "__main__":
    main()
