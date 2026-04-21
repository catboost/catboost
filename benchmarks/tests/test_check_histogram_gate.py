"""
test_check_histogram_gate.py — Unit tests for check_histogram_gate.py.

Covers:
  - Single-config pass (no min-reduction, no regression) — Format A (raw profiler)
  - Single-config pass with min-reduction met
  - Single-config fail: regression exceeds max-regression
  - Single-config fail: missing histogram_ms key
  - Single-config pass — Format B (bench driver JSON with stage_timings)
  - 18-config pass: all configs within threshold
  - 18-config fail: one config regresses
  - 18-config: missing after-file
  - 18-config: custom --max-regression boundary

Each test writes minimal in-memory JSON fixtures to a tmp_path directory.
No real benchmark files are read.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SCRIPT = (
    Path(__file__).resolve().parent.parent / "check_histogram_gate.py"
)

_STAGE_NAMES = [
    "derivatives_ms", "init_partitions_ms", "partition_layout_ms",
    "histogram_ms", "suffix_scoring_ms", "leaf_sums_ms",
    "leaf_values_ms", "tree_apply_ms", "loss_eval_ms", "cpu_readback_ms",
]


def _make_bench_json(histogram_ms_values: list[float], task: str = "RMSE",
                     scale: int = 10000, bins: int = 128) -> dict:
    """Build a minimal bench-driver JSON (Format B) with the given per-iter histogram_ms values."""
    # per_iter_s is in seconds; convert from ms.
    per_iter_s = [v / 1000.0 for v in histogram_ms_values]
    mean_s = sum(per_iter_s) / len(per_iter_s)
    total_s = sum(per_iter_s)
    return {
        "meta": {"build": "TEST", "iterations": len(histogram_ms_values)},
        "runs": [
            {
                "task": task,
                "scale": scale,
                "bins": bins,
                "mlx_time_s": total_s,
                "stage_timings": {
                    "histogram_ms": {
                        "mean_s": mean_s,
                        "total_s": total_s,
                        "per_iter_s": per_iter_s,
                    },
                    "derivatives_ms": {
                        "mean_s": 0.0003,
                        "total_s": 0.015,
                        "per_iter_s": [0.0003] * len(histogram_ms_values),
                    },
                },
            }
        ],
    }


def _make_json(histogram_ms_values: list[float], extra_keys: dict | None = None) -> dict:
    """Build a minimal stage-profile JSON dict (Format A) with the given per-iter histogram_ms values."""
    iterations = []
    for i, hms in enumerate(histogram_ms_values):
        rec = {
            "iter": i,
            "derivatives_ms": 0.3,
            "init_partitions_ms": 0.0,
            "partition_layout_ms": 1.7,
            "histogram_ms": hms,
            "suffix_scoring_ms": 2.3,
            "leaf_sums_ms": 0.2,
            "leaf_values_ms": 0.2,
            "tree_apply_ms": 0.2,
            "loss_eval_ms": 0.2,
            "cpu_readback_ms": 0.1,
            "iter_total_ms": hms + 5.2,
        }
        if extra_keys:
            rec.update(extra_keys)
        iterations.append(rec)

    return {
        "meta": {"build": "TEST", "num_iterations": len(histogram_ms_values)},
        "stage_names": _STAGE_NAMES,
        "iterations": iterations,
    }


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2))


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(_SCRIPT)] + args,
        capture_output=True,
        text=True,
    )


# ---------------------------------------------------------------------------
# Single-config tests
# ---------------------------------------------------------------------------

class TestSingleConfig:
    def test_pass_no_regression(self, tmp_path):
        """Before and after are identical — no regression, no min-reduction required."""
        before = tmp_path / "before.json"
        after = tmp_path / "after.json"
        _write_json(before, _make_json([300.0, 310.0, 305.0]))
        _write_json(after, _make_json([300.0, 310.0, 305.0]))

        result = _run(["--before", str(before), "--after", str(after)])
        assert result.returncode == 0, result.stderr

    def test_pass_min_reduction_met(self, tmp_path):
        """After is 35% faster than before — satisfies --min-reduction 0.30."""
        before_vals = [300.0] * 5
        after_vals = [195.0] * 5   # 35% reduction
        before = tmp_path / "before.json"
        after = tmp_path / "after.json"
        _write_json(before, _make_json(before_vals))
        _write_json(after, _make_json(after_vals))

        result = _run([
            "--before", str(before),
            "--after", str(after),
            "--min-reduction", "0.30",
        ])
        assert result.returncode == 0, result.stderr

    def test_fail_regression_exceeds_limit(self, tmp_path):
        """After is 10% slower — exceeds the default 5% max-regression."""
        before = tmp_path / "before.json"
        after = tmp_path / "after.json"
        _write_json(before, _make_json([300.0] * 5))
        _write_json(after, _make_json([330.0] * 5))   # +10%

        result = _run(["--before", str(before), "--after", str(after)])
        assert result.returncode == 1
        assert "HISTOGRAM REGRESSION" in result.stderr

    def test_fail_min_reduction_not_met(self, tmp_path):
        """After is only 10% faster — does not satisfy --min-reduction 0.30."""
        before = tmp_path / "before.json"
        after = tmp_path / "after.json"
        _write_json(before, _make_json([300.0] * 5))
        _write_json(after, _make_json([270.0] * 5))   # -10%

        result = _run([
            "--before", str(before),
            "--after", str(after),
            "--min-reduction", "0.30",
        ])
        assert result.returncode == 1
        assert "INSUFFICIENT REDUCTION" in result.stderr

    def test_fail_missing_histogram_ms_key(self, tmp_path):
        """Iteration record missing histogram_ms — should exit 1 with a clear error."""
        before_data = _make_json([300.0] * 3)
        # Remove histogram_ms from all iteration records.
        for rec in before_data["iterations"]:
            del rec["histogram_ms"]
        after_data = _make_json([300.0] * 3)

        before = tmp_path / "before.json"
        after = tmp_path / "after.json"
        _write_json(before, before_data)
        _write_json(after, after_data)

        result = _run(["--before", str(before), "--after", str(after)])
        assert result.returncode == 1
        assert "histogram_ms" in result.stderr

    def test_pass_small_improvement_within_regression_limit(self, tmp_path):
        """After is 3% faster — within max-regression=5% and no min-reduction set."""
        before = tmp_path / "before.json"
        after = tmp_path / "after.json"
        _write_json(before, _make_json([300.0] * 5))
        _write_json(after, _make_json([291.0] * 5))   # -3%

        result = _run(["--before", str(before), "--after", str(after)])
        assert result.returncode == 0, result.stderr

    def test_regression_at_threshold_boundary_passes(self, tmp_path):
        """Exactly at the 5% threshold — should pass (limit is strictly greater-than)."""
        before = tmp_path / "before.json"
        after = tmp_path / "after.json"
        _write_json(before, _make_json([300.0] * 5))
        _write_json(after, _make_json([315.0] * 5))   # exactly +5%

        result = _run(["--before", str(before), "--after", str(after)])
        # delta_rel == 0.05, limit is > 0.05, so this is a pass.
        assert result.returncode == 0, result.stderr

    def test_regression_just_over_threshold_fails(self, tmp_path):
        """0.1 ms over the 5% threshold — should fail."""
        before = tmp_path / "before.json"
        after = tmp_path / "after.json"
        _write_json(before, _make_json([300.0] * 5))
        _write_json(after, _make_json([315.1] * 5))   # +5.03%

        result = _run(["--before", str(before), "--after", str(after)])
        assert result.returncode == 1
        assert "HISTOGRAM REGRESSION" in result.stderr

    def test_pass_format_b_bench_driver(self, tmp_path):
        """After file is Format B (bench driver JSON with stage_timings) — should pass."""
        before = tmp_path / "before.json"
        after = tmp_path / "after.json"
        _write_json(before, _make_json([300.0] * 5))
        # after uses Format B with 35% improvement (195 ms = 300 * 0.65)
        _write_json(after, _make_bench_json([195.0] * 5))

        result = _run([
            "--before", str(before),
            "--after", str(after),
            "--min-reduction", "0.30",
        ])
        assert result.returncode == 0, result.stderr

    def test_fail_format_b_regression(self, tmp_path):
        """After file is Format B and regresses by 10% — should fail."""
        before = tmp_path / "before.json"
        after = tmp_path / "after.json"
        _write_json(before, _make_json([300.0] * 5))
        _write_json(after, _make_bench_json([330.0] * 5))   # +10%

        result = _run(["--before", str(before), "--after", str(after)])
        assert result.returncode == 1
        assert "HISTOGRAM REGRESSION" in result.stderr

    def test_fail_unrecognised_json_format(self, tmp_path):
        """JSON file with neither 'iterations' nor 'runs' key — should exit 1."""
        before = tmp_path / "before.json"
        after = tmp_path / "after.json"
        _write_json(before, _make_json([300.0] * 5))
        _write_json(after, {"meta": {}, "unknown_key": []})

        result = _run(["--before", str(before), "--after", str(after)])
        assert result.returncode == 1


# ---------------------------------------------------------------------------
# 18-config mode tests
# ---------------------------------------------------------------------------

class Test18Config:
    def _populate_dirs(
        self,
        before_dir: Path,
        after_dir: Path,
        configs: list[tuple[str, float, float]],
    ) -> None:
        """Write before/after JSON pairs for the given (filename, before_ms, after_ms) list."""
        for name, bms, ams in configs:
            _write_json(before_dir / name, _make_json([bms] * 5))
            _write_json(after_dir / name, _make_json([ams] * 5))

    def test_18config_all_pass(self, tmp_path):
        """All 18 configs show improvement — gate should pass."""
        before_dir = tmp_path / "before"
        after_dir = tmp_path / "after"
        before_dir.mkdir()
        after_dir.mkdir()

        configs = [
            (f"baseline_{n}_{loss}_d6_{bins}bins.json", bms, bms * 0.65)
            for n, bms in [(1000, 290.0), (10000, 310.0), (50000, 470.0)]
            for loss in ("rmse", "logloss", "multiclass")
            for bins in (32, 128)
        ]
        self._populate_dirs(before_dir, after_dir, configs)

        result = _run([
            "--18config",
            "--before-dir", str(before_dir),
            "--after-dir", str(after_dir),
        ])
        assert result.returncode == 0, result.stderr

    def test_18config_one_config_regresses(self, tmp_path):
        """One config regresses by 10% — gate should fail and name the offending config."""
        before_dir = tmp_path / "before"
        after_dir = tmp_path / "after"
        before_dir.mkdir()
        after_dir.mkdir()

        configs = [
            (f"baseline_{n}_{loss}_d6_{bins}bins.json", 300.0, 195.0)
            for n in (1000, 10000, 50000)
            for loss in ("rmse", "logloss", "multiclass")
            for bins in (32, 128)
        ]
        # Inject a single regressing config.
        regressor_name = "baseline_10000_rmse_d6_128bins.json"
        configs = [
            (regressor_name, 310.0, 341.0)   # +10% regression
        ] + [c for c in configs if c[0] != regressor_name]
        self._populate_dirs(before_dir, after_dir, configs)

        result = _run([
            "--18config",
            "--before-dir", str(before_dir),
            "--after-dir", str(after_dir),
        ])
        assert result.returncode == 1
        assert "HISTOGRAM REGRESSION" in result.stderr
        assert regressor_name in result.stderr

    def test_18config_missing_after_file(self, tmp_path):
        """An after-file is missing — gate should exit 1 with an informative error."""
        before_dir = tmp_path / "before"
        after_dir = tmp_path / "after"
        before_dir.mkdir()
        after_dir.mkdir()

        _write_json(before_dir / "config_a.json", _make_json([300.0] * 5))
        # Intentionally do NOT write after_dir / "config_a.json".

        result = _run([
            "--18config",
            "--before-dir", str(before_dir),
            "--after-dir", str(after_dir),
        ])
        assert result.returncode == 1
        assert "missing" in result.stderr.lower() or "not found" in result.stderr.lower()

    def test_18config_custom_max_regression(self, tmp_path):
        """A 3% regression passes when --max-regression is set to 0.05 (default),
        but fails when tightened to 0.02."""
        before_dir = tmp_path / "before"
        after_dir = tmp_path / "after"
        before_dir.mkdir()
        after_dir.mkdir()

        _write_json(before_dir / "cfg.json", _make_json([300.0] * 5))
        _write_json(after_dir / "cfg.json", _make_json([309.0] * 5))  # +3%

        # Should pass at default threshold (5%).
        result_pass = _run([
            "--18config",
            "--before-dir", str(before_dir),
            "--after-dir", str(after_dir),
            "--max-regression", "0.05",
        ])
        assert result_pass.returncode == 0, result_pass.stderr

        # Should fail at tighter threshold (2%).
        result_fail = _run([
            "--18config",
            "--before-dir", str(before_dir),
            "--after-dir", str(after_dir),
            "--max-regression", "0.02",
        ])
        assert result_fail.returncode == 1
        assert "HISTOGRAM REGRESSION" in result_fail.stderr
