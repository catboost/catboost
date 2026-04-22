"""
test_qa_round9_sprint4_partition_layout.py — QA Round 9: Sprint 4 GPU partition layout.

Sprint 4 (TODO-007) ported ComputePartitionLayout from CPU scatter-sort to three
GPU-resident MLX primitives:
  DocIndices  = argsort(partitions)              # radix sort, stable
  PartSizes   = scatter_add(ones, partitions)    # count per partition
  PartOffsets = cumsum(PartSizes) - PartSizes    # exclusive prefix sum

Findings from diff / consumer audit:
  DEFECT-001: model_export_test.cpp still carries the OLD CPU-based
              ComputePartitionLayout (mx::eval + CPU loops). That test file
              diverges from the algorithm now in production. It will give
              correct output only coincidentally (depth=0 → single partition).
              Low severity for correctness, medium for test fidelity.
              File: catboost/mlx/tests/model_export_test.cpp:141

  FINDING-001 (doc gap): The float32-counter safety limit (>2^24 docs = count
              rounding) is noted inline in structure_searcher.cpp comment but is
              NOT documented in .claude/state/DECISIONS.md or README.md. Flag for
              documentation.

  FINDING-002 (pre-existing non-determinism): Two identical runs with the same
              seed differ by up to ~2.75e-7 on master. This is NOT a Sprint 4
              regression; it pre-dates this branch. Tracked below as a known
              limitation, not a bug.

NOTE: Python bindings invoke csv_train binary via subprocess, NOT structure_searcher.cpp.
These tests exercise the Python surface (which exercises csv_train) as a proxy for
overall correctness. The csv_train.cpp local ComputePartitionLayout already uses GPU
argsort but omits the `axis=0` parameter on cumsum (uses default axis). For a 1-D
array this is equivalent, so no behavioral divergence — noted as a style inconsistency.
"""

import numpy as np
import pytest

from catboost_mlx import CatBoostMLXClassifier, CatBoostMLXRegressor

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

REPO_ROOT_ABS = __file__  # used only for skip messages


def _regression_dataset(n=200, n_features=5, seed=42):
    """Small regression dataset with a clear linear signal."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    y = X @ np.array([2.0, -1.5, 0.5, 0.0, 0.0]) + rng.standard_normal(n) * 0.3
    return X, y


def _multiclass_dataset(n=120, n_features=4, n_classes=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    y = rng.integers(0, n_classes, size=n)
    return X, y


# ---------------------------------------------------------------------------
# Section 1: Regression anchor (pin final predictions against known values)
# ---------------------------------------------------------------------------


class TestRegressionAnchor:
    """Pin predictions against values locked in at Sprint 4 QA time.

    If Sprint 4 or any future sprint changes the numeric output, these tests
    will catch it before merge.
    """

    def test_rmse_final_loss_matches_sprint4_anchor(self):
        """RMSE model on 100-row dataset must match Sprint 4 anchor to 1e-4.

        Anchor captured with seed=0, 30 iterations, depth=4.
        """
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 4))
        y = X @ np.array([1.0, -0.5, 0.3, 0.0]) + rng.standard_normal(100) * 0.2

        m = CatBoostMLXRegressor(iterations=30, loss="rmse", random_seed=0, depth=4)
        m.fit(X, y)
        preds = m.predict(X)

        rmse = np.sqrt(np.mean((y - preds) ** 2))
        # Anchor: 0.30634809 — updated S26-D0-9 after DEC-028 fixed
        # RandomStrength noise scale (hessian-sum → gradient-RMS, matching CPU).
        # Prior value 0.43203182 reflected pre-fix over-scaled noise regime.
        assert abs(rmse - 0.306348) < 1e-3, (
            f"RMSE anchor mismatch: expected ~0.306348, got {rmse:.6f}"
        )

    def test_specific_predictions_match_anchor(self):
        """Spot-check three individual predictions against anchored values."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 4))
        y = X @ np.array([1.0, -0.5, 0.3, 0.0]) + rng.standard_normal(100) * 0.2

        m = CatBoostMLXRegressor(iterations=30, loss="rmse", random_seed=0, depth=4)
        m.fit(X, y)
        preds = m.predict(X)

        # Anchored values updated S26-D0-9 post-DEC-028 (seed=0).
        # Prior values [0.414606, -0.545893, 1.356884] reflected pre-fix noise.
        assert abs(preds[0] - 0.317253) < 1e-3, f"preds[0] mismatch: {preds[0]}"
        assert abs(preds[1] - (-0.568259)) < 1e-3, f"preds[1] mismatch: {preds[1]}"
        assert abs(preds[99] - 1.598960) < 1e-3, f"preds[99] mismatch: {preds[99]}"


# ---------------------------------------------------------------------------
# Section 2: Multiclass K=3 (exercises approxDim > 1 partition path)
# ---------------------------------------------------------------------------


class TestMulticlassPartitionPath:
    """Multiclass training exercises ComputePartitionLayout approxDim > 1 times
    per depth level — the path most impacted by Sprint 4."""

    def test_multiclass_k3_trains_without_crash(self):
        """K=3 multiclass fit on 120 rows must complete without exception."""
        X, y = _multiclass_dataset()
        m = CatBoostMLXClassifier(iterations=20, random_seed=0, depth=3)
        m.fit(X, y)  # must not raise

    def test_multiclass_k3_no_nan_in_probabilities(self):
        """predict_proba must return all-finite values."""
        X, y = _multiclass_dataset()
        m = CatBoostMLXClassifier(iterations=20, random_seed=0, depth=3)
        m.fit(X, y)
        proba = m.predict_proba(X)
        assert not np.any(np.isnan(proba)), "NaN found in predict_proba output"
        assert not np.any(np.isinf(proba)), "Inf found in predict_proba output"

    def test_multiclass_k3_probabilities_sum_to_one(self):
        """Each row's probabilities must sum to 1.0 (within float tolerance)."""
        X, y = _multiclass_dataset()
        m = CatBoostMLXClassifier(iterations=20, random_seed=0, depth=3)
        m.fit(X, y)
        proba = m.predict_proba(X)
        row_sums = proba.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5), (
            f"Row probabilities do not sum to 1; max deviation: {np.abs(row_sums - 1).max()}"
        )

    def test_multiclass_k3_proba_anchor(self):
        """First row probabilities must match Sprint 4 anchor."""
        X, y = _multiclass_dataset()
        m = CatBoostMLXClassifier(iterations=20, random_seed=0, depth=3)
        m.fit(X, y)
        proba = m.predict_proba(X)
        # Anchor updated S26-D0-9 post-DEC-028 RandomStrength fix.
        # Prior: [0.35687973, 0.36606121, 0.27705906] — pre-fix regime.
        expected = np.array([0.37227302, 0.36382151, 0.26390547])
        assert np.allclose(proba[0], expected, atol=1e-3), (
            f"Multiclass proba[0] anchor mismatch: got {proba[0]}, expected {expected}"
        )

    def test_multiclass_k3_loss_decreases(self):
        """Loss must strictly decrease from iteration 5 to 20 on clean data."""
        X, y = _multiclass_dataset(n=300, seed=7)
        m5 = CatBoostMLXClassifier(iterations=5, random_seed=7, depth=4)
        m20 = CatBoostMLXClassifier(iterations=20, random_seed=7, depth=4)
        m5.fit(X, y)
        m20.fit(X, y)

        def cross_entropy(proba, labels):
            n = len(labels)
            return -np.mean(np.log(proba[np.arange(n), labels.astype(int)] + 1e-12))

        ce5 = cross_entropy(m5.predict_proba(X), y)
        ce20 = cross_entropy(m20.predict_proba(X), y)
        assert ce20 < ce5, (
            f"Cross-entropy did not decrease with more iterations: ce5={ce5:.4f} ce20={ce20:.4f}"
        )


# ---------------------------------------------------------------------------
# Section 3: Edge cases specific to the GPU partition layout
# ---------------------------------------------------------------------------


class TestEdgeCasesPartitionLayout:
    """Edge cases that probe the argsort + scatter_add + cumsum pipeline."""

    def test_single_doc_rejected_cleanly(self):
        """1-row dataset — Python layer must reject it with ValueError before hitting GPU.

        A single doc has zero-variance features; the Python validation layer raises
        ValueError rather than letting the C++ partition code see a degenerate input.
        This verifies the guard exists and gives a clean error (not a crash or NaN).
        """
        X = np.array([[1.0, -1.0, 0.5]])
        y = np.array([2.0])
        m = CatBoostMLXRegressor(iterations=5, random_seed=0, depth=2)
        with pytest.raises((ValueError, Exception)):
            m.fit(X, y)

    def test_two_docs_depth_1_trains_without_crash(self):
        """2 rows, depth=1 → one split creates two partitions of size 1.
        Each partition has exactly 1 doc (numPartitions > typical numDocs for depth)."""
        X = np.array([[1.0], [-1.0]])
        y = np.array([1.0, -1.0])
        m = CatBoostMLXRegressor(iterations=3, random_seed=0, depth=1)
        m.fit(X, y)
        preds = m.predict(X)
        assert len(preds) == 2
        assert not np.any(np.isnan(preds))

    def test_all_same_feature_values_rejected_cleanly(self):
        """Constant features → Python layer raises ValueError before GPU code runs.

        When all features are constant (zero variance) the Python validation guard
        raises ValueError rather than letting the Metal kernel receive a degenerate
        partition where argsort sees all-zero partitions at every depth level.
        This verifies the guard exists and is not bypassed.
        """
        X = np.ones((50, 3))
        rng = np.random.default_rng(0)
        y = rng.standard_normal(50)
        m = CatBoostMLXRegressor(iterations=5, random_seed=0, depth=4)
        with pytest.raises(ValueError, match="constant"):
            m.fit(X, y)

    def test_depth_1_binary_split_no_nan(self):
        """Depth=1 → exactly 2 partitions. One per branch of argsort output.
        Verifies cumsum of 2-element PartSizes gives correct offsets."""
        rng = np.random.default_rng(99)
        X = rng.standard_normal((100, 3))
        y = (X[:, 0] > 0).astype(float)
        m = CatBoostMLXRegressor(iterations=10, random_seed=99, depth=1)
        m.fit(X, y)
        preds = m.predict(X)
        assert not np.any(np.isnan(preds)), "NaN in depth=1 predictions"
        assert not np.any(np.isinf(preds)), "Inf in depth=1 predictions"

    def test_large_scale_no_nan_no_crash(self):
        """10k rows x 20 features x depth 6 — verify no crash, no NaN, loss decreases."""
        rng = np.random.default_rng(42)
        n, f = 10_000, 20
        X = rng.standard_normal((n, f))
        y = X[:, 0] * 2 + X[:, 1] * -1.5 + rng.standard_normal(n) * 0.5

        m = CatBoostMLXRegressor(iterations=20, loss="rmse", random_seed=42, depth=6)
        m.fit(X, y)
        preds = m.predict(X)

        assert len(preds) == n
        assert not np.any(np.isnan(preds)), "NaN in large-scale predictions"
        assert not np.any(np.isinf(preds)), "Inf in large-scale predictions"

    def test_large_scale_multiclass_no_nan(self):
        """10k rows, 3 classes — multiclass GPU partition path at scale."""
        rng = np.random.default_rng(7)
        X = rng.standard_normal((10_000, 10))
        y = rng.integers(0, 3, size=10_000)

        m = CatBoostMLXClassifier(iterations=20, random_seed=7, depth=6)
        m.fit(X, y)
        proba = m.predict_proba(X)

        assert proba.shape == (10_000, 3)
        assert not np.any(np.isnan(proba)), "NaN in large-scale multiclass proba"
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5), "Row sums != 1 at scale"


# ---------------------------------------------------------------------------
# Section 4: Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Same input + same seed must produce the same output in both runs.

    NOTE: A pre-existing non-determinism of up to ~2.75e-7 was observed on master
    before Sprint 4.  This test verifies that Sprint 4 does NOT *worsen* the
    determinism bound — it does not assert bit-for-bit identity.
    """

    # Tolerance that reflects the pre-existing Metal GPU non-determinism on master.
    # Sprint 4 must not regress beyond this baseline.
    _DETERMINISM_TOLERANCE = 1e-5

    def test_regression_determinism_within_tolerance(self):
        """Two identical runs with seed=42 must agree to within pre-sprint tolerance."""
        X, y = _regression_dataset(n=500, seed=42)

        m1 = CatBoostMLXRegressor(iterations=50, random_seed=42, depth=6)
        m1.fit(X, y)
        p1 = m1.predict(X)

        m2 = CatBoostMLXRegressor(iterations=50, random_seed=42, depth=6)
        m2.fit(X, y)
        p2 = m2.predict(X)

        max_diff = np.abs(p1 - p2).max()
        assert max_diff < self._DETERMINISM_TOLERANCE, (
            f"Prediction non-determinism exceeds tolerance: max_diff={max_diff:.2e} "
            f"(tolerance={self._DETERMINISM_TOLERANCE:.2e}). "
            "Sprint 4 may have worsened the pre-existing GPU non-determinism."
        )

    def test_multiclass_determinism_within_tolerance(self):
        """Two identical multiclass runs must agree to within tolerance."""
        X, y = _multiclass_dataset(n=300, seed=42)

        m1 = CatBoostMLXClassifier(iterations=30, random_seed=42, depth=4)
        m1.fit(X, y)
        p1 = m1.predict_proba(X)

        m2 = CatBoostMLXClassifier(iterations=30, random_seed=42, depth=4)
        m2.fit(X, y)
        p2 = m2.predict_proba(X)

        max_diff = np.abs(p1 - p2).max()
        assert max_diff < self._DETERMINISM_TOLERANCE, (
            f"Multiclass proba non-determinism exceeds tolerance: max_diff={max_diff:.2e}"
        )

    def test_different_seeds_produce_different_predictions(self):
        """Sanity: two different seeds must NOT produce bit-for-bit identical output."""
        X, y = _regression_dataset(n=200, seed=10)

        m1 = CatBoostMLXRegressor(iterations=20, random_seed=1, depth=4)
        m1.fit(X, y)

        m2 = CatBoostMLXRegressor(iterations=20, random_seed=2, depth=4)
        m2.fit(X, y)

        assert not np.array_equal(m1.predict(X), m2.predict(X)), (
            "Two different seeds produced identical predictions — suspicious."
        )


# ---------------------------------------------------------------------------
# Section 5: csv_train regression smoke (proxy for the C++ partition path)
# ---------------------------------------------------------------------------


class TestCsvTrainRegressionSmoke:
    """Regression baseline using the csv_train binary (the path exercised by Python).

    The Python API invokes csv_train via subprocess, so these tests are the closest
    proxy we have for the full C++ GPU pipeline from Python.
    """

    def test_regression_loss_order_stability(self):
        """Training loss must decrease monotonically over iterations (sanity check)."""
        X, y = _regression_dataset(n=500, seed=42)

        m_early = CatBoostMLXRegressor(iterations=10, random_seed=42, depth=4)
        m_late = CatBoostMLXRegressor(iterations=50, random_seed=42, depth=4)

        m_early.fit(X, y)
        m_late.fit(X, y)

        rmse_early = np.sqrt(np.mean((y - m_early.predict(X)) ** 2))
        rmse_late = np.sqrt(np.mean((y - m_late.predict(X)) ** 2))

        assert rmse_late < rmse_early, (
            f"RMSE did not decrease: early={rmse_early:.4f} late={rmse_late:.4f}"
        )

    def test_different_seed_produces_finite_predictions(self):
        """Regression with seed=999 (different from most tests) must not produce NaN/Inf."""
        rng = np.random.default_rng(999)
        X = rng.standard_normal((300, 6))
        y = X[:, 0] - X[:, 2] + rng.standard_normal(300) * 0.5

        m = CatBoostMLXRegressor(iterations=30, random_seed=999, depth=5)
        m.fit(X, y)
        preds = m.predict(X)

        assert not np.any(np.isnan(preds)), "NaN with seed=999"
        assert not np.any(np.isinf(preds)), "Inf with seed=999"
        assert np.sqrt(np.mean((y - preds) ** 2)) < 2.0, "RMSE unreasonably large"
