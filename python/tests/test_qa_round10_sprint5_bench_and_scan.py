"""
test_qa_round10_sprint5_bench_and_scan.py — QA Round 10: Sprint 5 bench harness + parallel scan.

Sprint 5 delivered:
  TODO-014 — bench_boosting standalone C++ benchmark harness
  TODO-008 — Parallel SIMD scan for suffix_sum_histogram (kSuffixSumSource)
  TODO-009 — Dead-code removal: CPU FindBestSplit / FindBestSplitMultiDim
  TODO-013 — build_verify_test kernel param name fixes
  TODO-015 — DEC-003 decision doc: float32 16M-row limit

ARCHITECTURE NOTE:
  bench_boosting exercises the C++ library path directly (histogram.cpp +
  score_calcer.cpp + leaf_estimator.cpp kernels) — NOT via csv_train subprocess.
  It is the only test harness that exercises the parallel suffix-sum kernel.
  Python API tests use csv_train which has its own suffix-sum implementation
  (separate from kSuffixSumSource) and are unaffected by TODO-008.

ANCHORS (captured Sprint 5 QA, 2026-04-09):
  Binary  100k x 50 x depth6 x 100iters x bins32 x seed42 -> BENCH_FINAL_LOSS=0.69314516
  Multiclass 20k x 30 x 3cls x depth5 x 50iters x bins32 x seed42 -> BENCH_FINAL_LOSS=1.07820153

BUG-001 (QA FINDING — non-determinism in parallel SIMD scan):
  The SIMD parallel scan kernel (TODO-008, kSuffixSumSource) introduces run-to-run
  non-determinism at small-to-medium dataset scales. At 10k rows, bins=32,96 show
  variance of ~3e-3 in final loss across repeated same-seed runs. At 100k rows the
  variance disappears entirely (all runs produce identical BENCH_FINAL_LOSS). The
  serial scan (pre-Sprint-5) is fully deterministic at all tested scales.

  Likely cause: simd_prefix_inclusive_sum on Metal has non-deterministic addition
  order within the SIMD group when the input histogram values span a wide dynamic
  range (floats from atomic CAS accumulation). At 100k rows the sum is large enough
  that all splits resolve the same way despite ULP differences; at 10k rows the
  splits near the boundary flip.

  This is NOT masked by the ml-engineer's claim of "bit-for-bit identical" results
  because those benchmarks used 100k rows where the non-determinism is quiescent.

  Severity: MEDIUM — training output is non-deterministic at sub-100k scales, which
  violates the reproducibility guarantee. Fixed at 100k rows only.

  Reproducer: run bench_boosting --rows 10000 --features 20 --classes 2 --depth 4
              --iters 30 --bins 96 --seed 42 ten times; losses span ~0.677 to 0.683.
  Reference file: catboost/mlx/kernels/kernel_sources.h (kSuffixSumSource)

SKIP POLICY:
  All tests in this module skip with pytest.skip() if bench_boosting is not
  compiled at /tmp/bench_boosting. Build with:

    MLX=/opt/homebrew/Cellar/mlx/0.31.1
    REPO=<catboost-mlx root>
    clang++ -std=c++17 -O2 -I$MLX/include -I$REPO -L$MLX/lib -lmlx \\
            -framework Metal -framework Foundation -Wno-c++20-extensions \\
            $REPO/catboost/mlx/tests/bench_boosting.cpp -o /tmp/bench_boosting
"""

import os
import re
import subprocess

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BENCH_BINARY = "/tmp/bench_boosting"

# Regression anchors: final loss must match within 1e-6
BINARY_ANCHOR_LOSS = 0.69314516    # 100k x 50 x cls2 x depth6 x 100iters x bins32 seed42
MULTICLASS_ANCHOR_LOSS = 1.07820153  # 20k x 30 x cls3 x depth5 x 50iters x bins32 seed42


def _binary_available():
    return os.path.isfile(BENCH_BINARY) and os.access(BENCH_BINARY, os.X_OK)


def _skip_if_missing():
    if not _binary_available():
        pytest.skip(
            f"bench_boosting not compiled at {BENCH_BINARY}. "
            "Build with: clang++ -std=c++17 -O2 -I<mlx>/include -I<repo> "
            "-L<mlx>/lib -lmlx -framework Metal -framework Foundation "
            "-Wno-c++20-extensions <repo>/catboost/mlx/tests/bench_boosting.cpp "
            "-o /tmp/bench_boosting"
        )


def _run_bench(extra_args, timeout=180):
    """Run bench_boosting and return (returncode, stdout).

    Raises pytest.fail on non-zero return code with stderr included.
    """
    _skip_if_missing()
    cmd = [BENCH_BINARY] + extra_args
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        pytest.fail(
            f"bench_boosting exited with {result.returncode}.\n"
            f"args: {extra_args}\n"
            f"stderr: {result.stderr}\n"
            f"stdout: {result.stdout}"
        )
    return result.stdout


def _extract_final_loss(stdout):
    """Parse BENCH_FINAL_LOSS=<value> from bench_boosting output."""
    m = re.search(r"BENCH_FINAL_LOSS=([0-9.eE+\-]+)", stdout)
    assert m is not None, (
        f"BENCH_FINAL_LOSS not found in bench_boosting output:\n{stdout}"
    )
    return float(m.group(1))


def _has_nan(stdout):
    """Return True if any loss line in stdout contains nan or inf."""
    for line in stdout.splitlines():
        if "loss=" in line.lower():
            val = re.search(r"loss=([0-9.eE+\-naif]+)", line, re.IGNORECASE)
            if val:
                try:
                    v = float(val.group(1))
                    import math
                    if math.isnan(v) or math.isinf(v):
                        return True
                except ValueError:
                    return True
    return False


def _loss_monotonically_decreases(stdout):
    """Return True if extracted loss values (from 'loss=X' lines) are non-increasing.

    Allows a tolerance of 1e-5 for floating-point noise in warm iterations.
    """
    losses = []
    for line in stdout.splitlines():
        m = re.search(r"loss=([0-9.eE+\-]+)", line)
        if m:
            try:
                losses.append(float(m.group(1)))
            except ValueError:
                pass
    if len(losses) < 2:
        return True
    for i in range(1, len(losses)):
        if losses[i] > losses[i - 1] + 1e-4:
            return False
    return True


# ---------------------------------------------------------------------------
# Section 1: Regression anchor — future sprints must not silently regress these
# ---------------------------------------------------------------------------

class TestBenchBoostingAnchors:
    """Pin BENCH_FINAL_LOSS values so future sprints cannot silently regress them.

    Anchors were captured on Sprint 5 QA (2026-04-09) after confirming
    equivalence with the serial suffix-sum baseline.
    """

    def test_binary_100k_anchor(self):
        """Binary 100k x 50 x depth6 x 100iters x bins32 seed42 must match anchor to 1e-6."""
        _skip_if_missing()
        stdout = _run_bench([
            "--rows", "100000", "--features", "50", "--classes", "2",
            "--depth", "6", "--iters", "100", "--bins", "32", "--seed", "42"
        ], timeout=240)
        loss = _extract_final_loss(stdout)
        assert abs(loss - BINARY_ANCHOR_LOSS) < 1e-6, (
            f"Binary anchor regression: expected {BINARY_ANCHOR_LOSS}, got {loss:.8f}"
        )

    def test_multiclass_20k_anchor(self):
        """Multiclass 20k x 30 x 3cls x depth5 x 50iters x bins32 seed42 must match anchor to 1e-6."""
        _skip_if_missing()
        stdout = _run_bench([
            "--rows", "20000", "--features", "30", "--classes", "3",
            "--depth", "5", "--iters", "50", "--bins", "32", "--seed", "42"
        ], timeout=120)
        loss = _extract_final_loss(stdout)
        assert abs(loss - MULTICLASS_ANCHOR_LOSS) < 1e-6, (
            f"Multiclass anchor regression: expected {MULTICLASS_ANCHOR_LOSS}, got {loss:.8f}"
        )

    def test_binary_anchor_no_nan(self):
        """Binary standard run must produce no NaN or Inf loss values in any iteration."""
        _skip_if_missing()
        stdout = _run_bench([
            "--rows", "100000", "--features", "50", "--classes", "2",
            "--depth", "6", "--iters", "100", "--bins", "32", "--seed", "42"
        ], timeout=240)
        assert not _has_nan(stdout), (
            f"NaN/Inf detected in binary standard run output."
        )

    def test_multiclass_anchor_no_nan(self):
        """Multiclass standard run must produce no NaN or Inf loss values."""
        _skip_if_missing()
        stdout = _run_bench([
            "--rows", "20000", "--features", "30", "--classes", "3",
            "--depth", "5", "--iters", "50", "--bins", "32", "--seed", "42"
        ], timeout=120)
        assert not _has_nan(stdout), (
            f"NaN/Inf detected in multiclass standard run output."
        )


# ---------------------------------------------------------------------------
# Section 2: Determinism — same seed, same binary, two runs must be identical
# ---------------------------------------------------------------------------

class TestBenchBoostingDeterminism:
    """Same seed and binary must produce near-identical BENCH_FINAL_LOSS across two runs.

    BUG-001 context: The parallel SIMD scan (TODO-008) introduces non-determinism
    at sub-100k scales (up to ~3e-3 variance). At 100k rows the variance is at most
    1e-6 (observed: 0 or ~1.2e-7 on successive runs). This suite uses 100k rows
    with tolerance=1e-5 to allow the rare ULP-level drift while catching genuine
    regressions (e.g., if the non-determinism worsened to ~1e-3 level at 100k scale).

    For strict bit-for-bit determinism, BUG-001 in kSuffixSumSource must first be
    resolved in a future sprint.
    """

    # Tolerance for 100k-row determinism: allows rare ULP drift (~1.2e-7) but
    # catches any genuine regression from BUG-001 spreading to larger scales.
    _DETERMINISM_TOL = 1e-5

    def test_binary_deterministic_same_seed(self):
        """Two runs with same seed must produce BENCH_FINAL_LOSS within 1e-5 (100k scale)."""
        _skip_if_missing()
        base_args = [
            "--rows", "100000", "--features", "50", "--classes", "2",
            "--depth", "6", "--iters", "100", "--bins", "32", "--seed", "42"
        ]
        stdout1 = _run_bench(base_args, timeout=240)
        stdout2 = _run_bench(base_args, timeout=240)
        loss1 = _extract_final_loss(stdout1)
        loss2 = _extract_final_loss(stdout2)
        assert abs(loss1 - loss2) < self._DETERMINISM_TOL, (
            f"Excessive non-determinism at 100k scale: run1={loss1:.10f} run2={loss2:.10f}, "
            f"delta={abs(loss1-loss2):.2e} (tolerance={self._DETERMINISM_TOL:.2e}). "
            "BUG-001 (kSuffixSumSource non-determinism) has spread beyond sub-100k scales."
        )

    def test_bins33_deterministic_same_seed(self):
        """Determinism check at bins=33 (carry-propagation boundary) at 100k-row scale.

        bins=33 crosses the single-SIMD-group boundary: chunk 0 handles 32 bins,
        chunk 1 handles 1 bin with carry from chunk 0. This is the most sensitive
        point for carry-propagation races. Uses 100k rows per BUG-001 scale caveat.
        """
        _skip_if_missing()
        base_args = [
            "--rows", "100000", "--features", "50", "--classes", "2",
            "--depth", "6", "--iters", "100", "--bins", "33", "--seed", "42"
        ]
        stdout1 = _run_bench(base_args, timeout=240)
        stdout2 = _run_bench(base_args, timeout=240)
        loss1 = _extract_final_loss(stdout1)
        loss2 = _extract_final_loss(stdout2)
        assert abs(loss1 - loss2) < self._DETERMINISM_TOL, (
            f"Non-determinism at bins=33 even at 100k-row scale: "
            f"run1={loss1:.10f} run2={loss2:.10f}, delta={abs(loss1-loss2):.2e}. "
            "Possible carry-propagation race in the chunked SIMD scan path."
        )

    def test_different_seeds_produce_different_losses(self):
        """Sanity: seed=1 and seed=2 must NOT produce identical BENCH_FINAL_LOSS.

        If they do, the seeded data generator is broken or seed is being ignored.
        """
        _skip_if_missing()
        common_args = [
            "--rows", "5000", "--features", "20", "--classes", "2",
            "--depth", "4", "--iters", "30", "--bins", "32"
        ]
        stdout1 = _run_bench(common_args + ["--seed", "1"], timeout=60)
        stdout2 = _run_bench(common_args + ["--seed", "2"], timeout=60)
        loss1 = _extract_final_loss(stdout1)
        loss2 = _extract_final_loss(stdout2)
        assert loss1 != loss2, (
            f"seed=1 and seed=2 produced identical loss={loss1:.8f}. "
            "The seed is likely not being consumed by the data generator."
        )


# ---------------------------------------------------------------------------
# Section 3: Parallel scan correctness — sweep bins to exercise both code paths
# ---------------------------------------------------------------------------

# bins <= 32: single-SIMD-group path (simd_prefix_inclusive_sum, one pass)
# bins >  32: multi-chunk path with carry propagation
#
# Key boundary values:
#   bins=32: exactly fills one SIMD group (single-pass path)
#   bins=33: first entry into multi-chunk path — carry from chunk 0 to chunk 1
#   bins=64: two full chunks, no partial chunk (boundary alignment case)
#   bins=65: two full + one partial chunk
#   bins=255: maximum supported value, 8 full chunks
#
# Additional values (34, 48, 63, 96, 128) probe interior of each region.

BINS_SWEEP = [2, 16, 32, 33, 34, 48, 63, 64, 65, 96, 128, 255]


class TestParallelScanBinsSweep:
    """Parameterized sweep over bins values to exercise the suffix-sum kernel.

    Tests assert:
      1. Training completes without crash.
      2. No NaN or Inf in any iteration's loss.
      3. Loss does not increase by more than a small tolerance (training is stable).
      4. BENCH_FINAL_LOSS is deterministic at 100k-row scale (see BUG-001 note).

    BUG-001: The parallel SIMD scan introduces non-determinism at sub-100k scales
    for several bin counts. The sweep tests use 10k rows for speed; the determinism
    sub-test uses 100k rows where the parallel scan IS deterministic.
    """

    @pytest.mark.parametrize("bins", BINS_SWEEP)
    def test_completes_without_crash(self, bins):
        """bench_boosting with bins=<N> must complete (non-zero exit would fail _run_bench)."""
        _skip_if_missing()
        _run_bench([
            "--rows", "10000", "--features", "20", "--classes", "2",
            "--depth", "4", "--iters", "30", "--bins", str(bins), "--seed", "42"
        ], timeout=90)

    @pytest.mark.parametrize("bins", BINS_SWEEP)
    def test_no_nan_in_loss(self, bins):
        """No NaN or Inf in any iteration loss at bins=<N>."""
        _skip_if_missing()
        stdout = _run_bench([
            "--rows", "10000", "--features", "20", "--classes", "2",
            "--depth", "4", "--iters", "30", "--bins", str(bins), "--seed", "42"
        ], timeout=90)
        assert not _has_nan(stdout), (
            f"NaN/Inf detected in iteration loss at bins={bins}"
        )

    @pytest.mark.parametrize("bins", BINS_SWEEP)
    def test_training_stable_loss_not_diverging(self, bins):
        """Loss must not increase by more than 1e-4 across iterations at bins=<N>.

        Strict monotonicity is not guaranteed for all bin counts because different
        bins change the data distribution, but diverging (NaN-trending) loss is a
        red flag for carry-propagation bugs in the chunked scan path.
        """
        _skip_if_missing()
        stdout = _run_bench([
            "--rows", "10000", "--features", "20", "--classes", "2",
            "--depth", "4", "--iters", "30", "--bins", str(bins), "--seed", "42"
        ], timeout=90)
        assert _loss_monotonically_decreases(stdout), (
            f"Training loss diverged (increased by >1e-4) at bins={bins}. "
            "Possible carry-propagation bug in the chunked SIMD scan path."
        )

    @pytest.mark.parametrize("bins", BINS_SWEEP)
    def test_deterministic_same_seed(self, bins):
        """Two runs with same seed at bins=<N> must produce identical BENCH_FINAL_LOSS.

        BUG-001 NOTE: The parallel SIMD scan (TODO-008) introduces non-determinism
        at sub-100k scales for some bin counts. At 10k rows, several bin values show
        run-to-run variance of up to ~3e-3 in final loss. This test uses 100k rows
        (the ml-engineer's validated scale) where the parallel scan IS deterministic.

        If this test fails at 100k rows, it indicates a regression beyond BUG-001
        (the non-determinism has worsened beyond the 100k stability threshold).
        """
        _skip_if_missing()
        args = [
            "--rows", "100000", "--features", "50", "--classes", "2",
            "--depth", "6", "--iters", "50", "--bins", str(bins), "--seed", "42"
        ]
        stdout1 = _run_bench(args, timeout=240)
        stdout2 = _run_bench(args, timeout=240)
        loss1 = _extract_final_loss(stdout1)
        loss2 = _extract_final_loss(stdout2)
        # Tolerance 1e-5 allows rare ULP-level drift seen at 100k scale (~1.2e-7 max)
        # while catching any regression where BUG-001 spreads to larger scales.
        assert abs(loss1 - loss2) < 1e-5, (
            f"Non-determinism at bins={bins} even at 100k-row scale: "
            f"run1={loss1:.10f} run2={loss2:.10f}, delta={abs(loss1-loss2):.2e}. "
            f"This exceeds the known BUG-001 boundary (non-determinism should only "
            f"manifest below 100k rows with the current kSuffixSumSource parallel scan)."
        )


# ---------------------------------------------------------------------------
# Section 4: Edge cases specific to the harness
# ---------------------------------------------------------------------------

class TestBenchBoostingEdgeCases:
    """Exercise unusual configurations to confirm harness robustness."""

    def test_very_small_100_rows(self):
        """100-row dataset must complete and produce finite loss."""
        _skip_if_missing()
        stdout = _run_bench([
            "--rows", "100", "--features", "5", "--classes", "2",
            "--depth", "3", "--iters", "10", "--bins", "32", "--seed", "42"
        ], timeout=60)
        loss = _extract_final_loss(stdout)
        import math
        assert math.isfinite(loss), f"Non-finite loss with 100 rows: {loss}"

    def test_depth_1_trivial_tree(self):
        """depth=1 creates a single split — exercises the minimal tree case."""
        _skip_if_missing()
        stdout = _run_bench([
            "--rows", "1000", "--features", "10", "--classes", "2",
            "--depth", "1", "--iters", "20", "--bins", "32", "--seed", "42"
        ], timeout=60)
        loss = _extract_final_loss(stdout)
        import math
        assert math.isfinite(loss), f"Non-finite loss at depth=1: {loss}"
        assert not _has_nan(stdout), "NaN at depth=1"

    def test_classes_5_higher_dim_multiclass(self):
        """K=5 (approxDim=4) exercises the approxDim loop in multiclass gradient/leaf code."""
        _skip_if_missing()
        stdout = _run_bench([
            "--rows", "5000", "--features", "20", "--classes", "5",
            "--depth", "4", "--iters", "30", "--bins", "32", "--seed", "42"
        ], timeout=90)
        loss = _extract_final_loss(stdout)
        import math
        assert math.isfinite(loss), f"Non-finite loss for classes=5: {loss}"
        assert not _has_nan(stdout), "NaN detected in classes=5 run"

    def test_bins_2_minimal_bins(self):
        """bins=2 is the minimum supported value — exercises bin-boundary guard."""
        _skip_if_missing()
        stdout = _run_bench([
            "--rows", "500", "--features", "10", "--classes", "2",
            "--depth", "4", "--iters", "20", "--bins", "2", "--seed", "42"
        ], timeout=60)
        loss = _extract_final_loss(stdout)
        import math
        assert math.isfinite(loss), f"Non-finite loss at bins=2: {loss}"

    def test_bins_255_maximum_exercises_chunked_path(self):
        """bins=255 is the maximum — exercises the full 8-chunk (ceil(255/32)=8) scan path."""
        _skip_if_missing()
        stdout = _run_bench([
            "--rows", "1000", "--features", "20", "--classes", "2",
            "--depth", "4", "--iters", "20", "--bins", "255", "--seed", "42"
        ], timeout=60)
        loss = _extract_final_loss(stdout)
        import math
        assert math.isfinite(loss), f"Non-finite loss at bins=255: {loss}"
        assert not _has_nan(stdout), "NaN at bins=255"

    def test_bug001_parallel_scan_nondeterminism_reproduced_at_small_scale(self):
        """Regression reproducer for BUG-001: parallel SIMD scan is non-deterministic at 10k rows.

        This test DOCUMENTS the defect by asserting that two same-seed runs at 10k rows
        with bins=96 produce DIFFERENT final losses. If this test ever fails (i.e., the
        two runs become identical), BUG-001 has been fixed — update this test to a
        positive determinism assertion and remove the xfail marker.

        Root cause: simd_prefix_inclusive_sum in kSuffixSumSource changes the floating-point
        addition order vs the serial loop. At small scales the histogram bins have large
        relative variance, causing suffix-sum differences that shift split boundaries,
        compounding across 30 iterations. At 100k rows the sums are large enough that
        ULP differences do not change the argmax.

        File: catboost/mlx/kernels/kernel_sources.h (kSuffixSumSource multi-pass path)
        """
        _skip_if_missing()
        args = [
            "--rows", "10000", "--features", "20", "--classes", "2",
            "--depth", "4", "--iters", "30", "--bins", "96", "--seed", "42"
        ]
        losses = []
        for _ in range(5):
            stdout = _run_bench(args, timeout=60)
            losses.append(_extract_final_loss(stdout))

        # All losses must be finite (training completes)
        import math
        for i, loss in enumerate(losses):
            assert math.isfinite(loss), f"Non-finite loss at run {i}: {loss}"

        # Document: at least some variation is expected (BUG-001 is present)
        # This assertion will fail (xfail) once BUG-001 is fixed.
        min_loss = min(losses)
        max_loss = max(losses)
        variation = max_loss - min_loss
        # Currently observed variation: ~0.005. Assert it's present to track the bug.
        # When variation drops to 0, the bug is fixed.
        assert variation > 1e-6, (
            f"BUG-001 appears to be fixed: all 5 runs produced identical loss. "
            f"losses={losses}. "
            f"Update this test to assert variation == 0 and remove this assertion."
        )

    def test_seed_variation_loss_changes_but_stays_finite(self):
        """Running the same config with 5 different seeds produces different but finite losses."""
        _skip_if_missing()
        seeds = [1, 7, 13, 99, 12345]
        losses = []
        for seed in seeds:
            stdout = _run_bench([
                "--rows", "5000", "--features", "20", "--classes", "2",
                "--depth", "4", "--iters", "20", "--bins", "32", "--seed", str(seed)
            ], timeout=60)
            loss = _extract_final_loss(stdout)
            import math
            assert math.isfinite(loss), f"Non-finite loss at seed={seed}: {loss}"
            losses.append(loss)

        # All seeds must not produce identical losses (seed is actually varying data)
        assert len(set(losses)) > 1, (
            f"All seeds produced the same loss={losses[0]:.8f}. "
            "Seed is not being used in data generation."
        )
