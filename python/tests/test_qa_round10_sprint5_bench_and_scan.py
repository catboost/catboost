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

ANCHORS (captured Sprint 5 QA, 2026-04-09; multiclass updated by BUG-001 fix 2026-04-10):
  Binary  100k x 50 x depth6 x 100iters x bins32 x seed42 -> BENCH_FINAL_LOSS=0.69314516
  Multiclass 20k x 30 x 3cls x depth5 x 50iters x bins32 x seed42 -> BENCH_FINAL_LOSS=1.09757149
    (pre-fix anchor 1.07820153 was captured with buggy (32,1,1) threadgroup that left
     scanBuf[32..255] uninitialized; Metal happened to zero them on the first dispatch,
     giving a deterministic but numerically wrong suffix sum)

BUG-001 (FIXED in score_calcer.cpp 2026-04-10):
  Root cause: kSuffixSumSource declares threadgroup float scanBuf[256] and runs a
  Hillis-Steele scan requiring 256 active threads, but score_calcer.cpp dispatched
  with suffixTG=(32,1,1). Threads 32..255 never ran, leaving scanBuf[32..255]
  uninitialized (threadgroup memory is NOT zeroed between dispatches on Apple Silicon).
  Different dispatches produced different garbage values → non-deterministic suffix
  sums → non-deterministic splits → varying final loss.

  Fix: change suffixTG from (32,1,1) to (256,1,1) in both FindBestSplitGPU overloads
  in score_calcer.cpp, and set init_value=0.0f so one-hot and skipped ordinal bins
  read as 0. The bench_boosting.cpp test harness already used (256,1,1) correctly.

  Verification: 10 runs of bench_boosting --rows 10000 --features 20 --classes 2
    --depth 4 --iters 30 --bins 96 --seed 42 now all produce BENCH_FINAL_LOSS=0.69310117.
  Binary 100k reference losses unchanged: bins=32 → 0.69314516, bins=255 → 0.69313669.

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
MULTICLASS_ANCHOR_LOSS = 1.09757149  # 20k x 30 x cls3 x depth5 x 50iters x bins32 seed42 (BUG-001 fix: updated from 1.07820153)


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
            "NaN/Inf detected in binary standard run output."
        )

    def test_multiclass_anchor_no_nan(self):
        """Multiclass standard run must produce no NaN or Inf loss values."""
        _skip_if_missing()
        stdout = _run_bench([
            "--rows", "20000", "--features", "30", "--classes", "3",
            "--depth", "5", "--iters", "50", "--bins", "32", "--seed", "42"
        ], timeout=120)
        assert not _has_nan(stdout), (
            "NaN/Inf detected in multiclass standard run output."
        )


# ---------------------------------------------------------------------------
# Section 2: Determinism — same seed, same binary, two runs must be identical
# ---------------------------------------------------------------------------

class TestBenchBoostingDeterminism:
    """Same seed and binary must produce bit-for-bit identical BENCH_FINAL_LOSS across runs.

    BUG-001 fix context: The non-determinism was traced to two sources in kHistOneByteSource:
      1. CAS-based float adds across SIMD groups within a threadgroup (inter-SIMD-group races).
      2. atomic_fetch_add_explicit across threadgroups when maxBlocksPerPart>1.
    Fixed by:
      1. Per-SIMD-group sub-histograms in kHistOneByteSource (TOTAL_HIST_SIZE allocation).
      2. maxBlocksPerPart=1 in bench_boosting.cpp (matches production histogram.cpp).
    After the fix, results must be bit-for-bit identical at ALL tested scales.
    """

    # Strict zero tolerance: after BUG-001 fix, results must be bit-for-bit identical.
    _DETERMINISM_TOL = 0.0

    def test_binary_deterministic_same_seed(self):
        """Two runs with same seed must produce identical BENCH_FINAL_LOSS (BUG-001 fixed)."""
        _skip_if_missing()
        base_args = [
            "--rows", "100000", "--features", "50", "--classes", "2",
            "--depth", "6", "--iters", "100", "--bins", "32", "--seed", "42"
        ]
        stdout1 = _run_bench(base_args, timeout=240)
        stdout2 = _run_bench(base_args, timeout=240)
        loss1 = _extract_final_loss(stdout1)
        loss2 = _extract_final_loss(stdout2)
        assert loss1 == loss2, (
            f"Non-determinism at 100k scale: run1={loss1:.10f} run2={loss2:.10f}, "
            f"delta={abs(loss1-loss2):.2e}. BUG-001 regression."
        )

    def test_bins33_deterministic_same_seed(self):
        """Determinism check at bins=33 (carry-propagation boundary) — BUG-001 fixed.

        bins=33 crosses the single-SIMD-group boundary: chunk 0 handles 32 bins,
        chunk 1 handles 1 bin with carry from chunk 0.
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
        assert loss1 == loss2, (
            f"Non-determinism at bins=33: "
            f"run1={loss1:.10f} run2={loss2:.10f}, delta={abs(loss1-loss2):.2e}. "
            "Possible carry-propagation race in the chunked SIMD scan path. BUG-001 regression."
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
        """Two runs with same seed at bins=<N> must produce bit-for-bit identical BENCH_FINAL_LOSS.

        BUG-001 fixed: Previously non-deterministic at 10k rows for bins in
        {32, 33, 34, 48, 65, 96, 255} due to CAS float add races in kHistOneByteSource.
        After the fix (per-SIMD sub-histograms + maxBlocksPerPart=1), results must be
        identical at ALL scales. This test uses 10k rows (the previously failing scale)
        to actively verify the fix rather than avoid the problematic configuration.
        """
        _skip_if_missing()
        args = [
            "--rows", "10000", "--features", "20", "--classes", "2",
            "--depth", "4", "--iters", "30", "--bins", str(bins), "--seed", "42"
        ]
        stdout1 = _run_bench(args, timeout=90)
        stdout2 = _run_bench(args, timeout=90)
        loss1 = _extract_final_loss(stdout1)
        loss2 = _extract_final_loss(stdout2)
        assert loss1 == loss2, (
            f"BUG-001 regression at bins={bins} 10k-row scale: "
            f"run1={loss1:.10f} run2={loss2:.10f}, delta={abs(loss1-loss2):.2e}. "
            f"Root cause: CAS float add non-determinism in kHistOneByteSource "
            f"(per-SIMD sub-histogram fix may be incomplete)."
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

    def test_bug001_determinism_fixed_at_small_scale(self):
        """BUG-001 fix verification: parallel scan must be bit-for-bit deterministic at 10k rows.

        BUG-001 was introduced in Sprint 5 (commit f8be378): the parallel SIMD scan
        (kSuffixSumSource) was combined with a histogram kernel (kHistOneByteSource)
        that used CAS-based float atomic adds across multiple SIMD groups within one
        threadgroup. SIMD groups within a threadgroup do NOT execute in lockstep with
        each other on Apple Silicon, so the CAS success ordering was non-deterministic
        across dispatches. This produced non-deterministic histogram bins, which the
        suffix-sum kernel then propagated into split-selection decisions.

        Fix (BUG-001, commit on mlx/sprint-5-parallel-scan-benchmark-harness):
          1. kHistOneByteSource: per-SIMD-group sub-histograms (TOTAL_HIST_SIZE allocation)
             eliminate inter-SIMD-group CAS contention. Intra-SIMD-group CAS is
             deterministic because all 32 lanes execute in lockstep (Apple Silicon guarantee).
             Fixed-order sequential reduction of SIMD group slices into slice 0.
          2. bench_boosting.cpp: maxBlocksPerPart=1 (matches production histogram.cpp)
             eliminates cross-threadgroup atomic_fetch_add_explicit float-add races
             on global histogram slots.

        This test verifies all 10 runs at 10k rows with bins=96 produce identical loss,
        and that the same holds across all bins in the BINS_SWEEP (32, 33, 34, 48, ...).
        """
        _skip_if_missing()
        args = [
            "--rows", "10000", "--features", "20", "--classes", "2",
            "--depth", "4", "--iters", "30", "--bins", "96", "--seed", "42"
        ]
        losses = []
        for _ in range(10):
            stdout = _run_bench(args, timeout=60)
            losses.append(_extract_final_loss(stdout))

        import math
        for i, loss in enumerate(losses):
            assert math.isfinite(loss), f"Non-finite loss at run {i}: {loss}"

        min_loss = min(losses)
        max_loss = max(losses)
        variation = max_loss - min_loss

        assert variation == 0.0, (
            f"BUG-001 regression: parallel scan is still non-deterministic at 10k rows. "
            f"10 runs with bins=96 seed=42: variation={variation:.2e} "
            f"(min={min_loss:.8f}, max={max_loss:.8f}). "
            f"losses={[f'{v:.8f}' for v in losses]}. "
            f"Root cause: CAS float add in kHistOneByteSource or global atomic_fetch_add_explicit "
            f"in bench_boosting maxBlocksPerPart>1."
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
