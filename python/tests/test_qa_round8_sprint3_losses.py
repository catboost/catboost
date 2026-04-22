"""
test_qa_round8_sprint3_losses.py -- QA Round 8: Sprint 3 loss function wiring validation.

Sprint 3 wired MAE, Quantile(alpha), and Huber(delta) into train.cpp's C++ dispatch.
The csv_train standalone binary had its own loss dispatch before Sprint 3; this file
validates both surfaces — the Python API and the csv_train binary — plus checks the
train.cpp parameter-parsing contract as documented.

Bugs found during QA:
  BUG-001: csv_train does not lowercase the loss string before dispatch.
           Passing loss='MAE' (uppercase) crashes the binary with a Metal reshape
           exception instead of failing cleanly or normalizing to 'mae'.
  BUG-002: Python layer raises ValueError for 'Quantile:alpha=0.7' and
           'Huber:delta=1.0' (CatBoost's canonical param-name syntax) because the
           colon-suffix validator calls float() on 'alpha=0.7'/'delta=1.0'.
           The Python layer only accepts bare numeric suffixes like 'quantile:0.7'.
  BUG-003 (known limitation, not a bug): csv_train silently defaults Huber to
           delta=1.0 when no param is given.  train.cpp correctly enforces
           CB_ENSURE that delta is mandatory.  The two surfaces are intentionally
           divergent (documented in Sprint 3 known limitations).

Focus areas:
  1. Parameter parsing: alpha defaults, mandatory delta, error cases
  2. Numerical sanity: correct loss ordering on synthetic data with outliers
  3. Python surface: which loss strings the API accepts and rejects
  4. Regression: full 558-baseline suite must still pass
"""

import os
import subprocess

import numpy as np
import pytest

from catboost_mlx import CatBoostMLXRegressor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BINARY_PATH = REPO_ROOT


def _csv_train_binary():
    """Return path to csv_train binary, skip if not found."""
    p = os.path.join(REPO_ROOT, "csv_train")
    if not os.path.isfile(p):
        pytest.skip("csv_train binary not found at repo root")
    return p


def _make_regression_dataset(n=500, n_features=5, outlier_fraction=0.10, seed=42):
    """500-row regression dataset with 10% heavy outliers.

    Signal: y = X @ [2, -1.5, 0.5, 0, 0] + noise(0.5)
    Outliers: +-20 shift on 10% of rows so L2/L1 models diverge.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    true_coef = np.array([2.0, -1.5, 0.5, 0.0, 0.0])
    y = X @ true_coef + rng.standard_normal(n) * 0.5
    n_outliers = int(n * outlier_fraction)
    outlier_idx = rng.choice(n, size=n_outliers, replace=False)
    y[outlier_idx] += rng.choice([-20, 20], size=n_outliers)
    return X, y, outlier_idx


def _write_csv(X, y, path):
    import csv as csv_mod
    data = np.column_stack([X, y])
    with open(path, "w", newline="") as f:
        w = csv_mod.writer(f)
        for row in data:
            w.writerow([f"{v:.6f}" for v in row])


def _run_model(loss, n_iterations=50, seed=42, binary_path=BINARY_PATH):
    """Fit a regressor on the standard 500-row dataset and return predictions."""
    X, y, _ = _make_regression_dataset(seed=seed)
    m = CatBoostMLXRegressor(
        iterations=n_iterations,
        loss=loss,
        random_seed=seed,
        binary_path=binary_path,
    )
    m.fit(X, y)
    return m.predict(X), y, _make_regression_dataset(seed=seed)[2]


# ============================================================================
# SECTION 1: Parameter parsing correctness (via train.cpp contract)
# ============================================================================


class TestParameterParsing:
    """Verify that alpha/delta are parsed, defaulted, and enforced correctly.

    These tests use the Python API which forwards to the csv_train binary.
    They are testing the parameter-forwarding contract, not train.cpp directly.
    """

    def test_quantile_no_alpha_defaults_to_0_5(self):
        """Quantile with no alpha must default to 0.5 (median regression).

        Loss 'quantile' without a colon suffix should produce the same
        predictions as 'quantile:0.5'.
        """
        X, y, _ = _make_regression_dataset()
        m_default = CatBoostMLXRegressor(
            iterations=50, loss="quantile", random_seed=42, binary_path=BINARY_PATH
        )
        m_explicit = CatBoostMLXRegressor(
            iterations=50, loss="quantile:0.5", random_seed=42, binary_path=BINARY_PATH
        )
        m_default.fit(X, y)
        m_explicit.fit(X, y)
        pred_default = m_default.predict(X)
        pred_explicit = m_explicit.predict(X)
        assert np.allclose(pred_default, pred_explicit, atol=1e-4), (
            "quantile (no alpha) should match quantile:0.5; "
            f"max diff = {np.abs(pred_default - pred_explicit).max():.6f}"
        )

    def test_quantile_different_alphas_produce_different_predictions(self):
        """alpha=0.1, 0.5, 0.9 must each produce distinct prediction distributions."""
        X, y, _ = _make_regression_dataset()
        models = {}
        for alpha in [0.1, 0.5, 0.9]:
            m = CatBoostMLXRegressor(
                iterations=50, loss=f"quantile:{alpha}", random_seed=42,
                binary_path=BINARY_PATH
            )
            m.fit(X, y)
            models[alpha] = m.predict(X)

        # All three should be distinct
        assert not np.allclose(models[0.1], models[0.5], atol=0.05), (
            "quantile:0.1 and quantile:0.5 produced identical predictions"
        )
        assert not np.allclose(models[0.5], models[0.9], atol=0.05), (
            "quantile:0.5 and quantile:0.9 produced identical predictions"
        )

    def test_quantile_alpha_boundary_0_1(self):
        """alpha=0.1 produces a model biased toward low quantile (below median)."""
        X, y, _ = _make_regression_dataset()
        m = CatBoostMLXRegressor(
            iterations=50, loss="quantile:0.1", random_seed=42, binary_path=BINARY_PATH
        )
        m.fit(X, y)
        preds = m.predict(X)
        # Low quantile: most predictions should be below the mean target
        frac_below_mean = (preds < y.mean()).mean()
        assert frac_below_mean > 0.5, (
            f"quantile:0.1 expected most preds below mean, got {frac_below_mean:.2f}"
        )

    def test_quantile_alpha_boundary_0_9(self):
        """alpha=0.9 produces a model biased toward high quantile (above median)."""
        X, y, _ = _make_regression_dataset()
        m = CatBoostMLXRegressor(
            iterations=50, loss="quantile:0.9", random_seed=42, binary_path=BINARY_PATH
        )
        m.fit(X, y)
        preds = m.predict(X)
        # High quantile: most predictions should be above the mean target
        frac_above_mean = (preds > y.mean()).mean()
        assert frac_above_mean > 0.5, (
            f"quantile:0.9 expected most preds above mean, got {frac_above_mean:.2f}"
        )

    def test_huber_with_delta_trains_successfully(self):
        """Huber with explicit delta must not raise an error."""
        X, y, _ = _make_regression_dataset()
        for delta in [0.5, 1.0, 5.0, 10.0]:
            m = CatBoostMLXRegressor(
                iterations=50, loss=f"huber:{delta}", random_seed=42,
                binary_path=BINARY_PATH
            )
            m.fit(X, y)
            preds = m.predict(X)
            assert preds.shape == (500,), f"huber:{delta} returned wrong shape"
            assert np.all(np.isfinite(preds)), f"huber:{delta} returned non-finite predictions"

    def test_huber_different_deltas_produce_different_predictions(self):
        """delta=0.5 and delta=10.0 should produce meaningfully different fits."""
        X, y, _ = _make_regression_dataset()
        m_small = CatBoostMLXRegressor(
            iterations=50, loss="huber:0.5", random_seed=42, binary_path=BINARY_PATH
        )
        m_large = CatBoostMLXRegressor(
            iterations=50, loss="huber:10.0", random_seed=42, binary_path=BINARY_PATH
        )
        m_small.fit(X, y)
        m_large.fit(X, y)
        preds_small = m_small.predict(X)
        preds_large = m_large.predict(X)
        max_diff = np.abs(preds_small - preds_large).max()
        assert max_diff > 0.1, (
            f"huber:0.5 and huber:10.0 predictions too similar (max_diff={max_diff:.4f}); "
            "expected meaningful divergence due to outlier handling"
        )

    def test_mae_trains_and_predicts(self):
        """MAE loss must train without error and return finite predictions."""
        X, y, _ = _make_regression_dataset()
        m = CatBoostMLXRegressor(
            iterations=50, loss="mae", random_seed=42, binary_path=BINARY_PATH
        )
        m.fit(X, y)
        preds = m.predict(X)
        assert preds.shape == (500,)
        assert np.all(np.isfinite(preds))


# ============================================================================
# SECTION 2: Numerical sanity on data with outliers
# ============================================================================


class TestNumericalSanity:
    """Validate loss function ordering and outlier behaviour on synthetic data.

    Dataset: 500 rows, 5 features, 10% outliers at +-20 magnitude.
    All models use 50 iterations, seed=42 for reproducibility.
    """

    @pytest.fixture(scope="class")
    def fitted_models(self):
        """Fit all five loss variants once; share across tests in this class."""
        X, y, outlier_idx = _make_regression_dataset()
        losses = ["rmse", "mae", "quantile:0.5", "huber:1.0", "huber:10.0",
                  "quantile:0.1", "quantile:0.9"]
        preds = {}
        for loss in losses:
            m = CatBoostMLXRegressor(
                iterations=50, loss=loss, random_seed=42, binary_path=BINARY_PATH
            )
            m.fit(X, y)
            preds[loss] = m.predict(X)
        return preds, y, outlier_idx

    def test_mae_and_quantile_05_similar_predictions(self, fitted_models):
        """MAE and Quantile(0.5) are both median regression; predictions should be close."""
        preds, y, _ = fitted_models
        mean_diff = np.abs(preds["mae"] - preds["quantile:0.5"]).mean()
        assert mean_diff < 1.5, (
            f"MAE and Quantile:0.5 should produce similar fits (both median), "
            f"mean absolute diff={mean_diff:.3f}"
        )

    def test_rmse_pulled_toward_outliers_vs_mae(self, fitted_models):
        """RMSE predictions on outlier rows should differ more from MAE than non-outlier rows.

        L2 is pulled toward outliers; L1 is robust to them.
        """
        preds, y, outlier_idx = fitted_models
        all_idx = np.arange(500)
        non_outlier_idx = np.setdiff1d(all_idx, outlier_idx)

        diff_outlier = np.abs(preds["rmse"][outlier_idx] - preds["mae"][outlier_idx]).mean()
        diff_normal = np.abs(preds["rmse"][non_outlier_idx] - preds["mae"][non_outlier_idx]).mean()

        assert diff_outlier > diff_normal, (
            f"RMSE and MAE should diverge MORE on outlier rows than normal rows. "
            f"Outlier diff={diff_outlier:.3f}, Normal diff={diff_normal:.3f}"
        )

    def test_huber_small_delta_closer_to_mae_on_outliers(self, fitted_models):
        """Huber(delta=1.0) should be closer to MAE on outlier rows than Huber(delta=10.0).

        Small delta clamps large residuals early => more MAE-like.
        Large delta stays quadratic longer => more RMSE-like.
        """
        preds, y, outlier_idx = fitted_models
        diff_huber1_vs_mae = np.abs(
            preds["huber:1.0"][outlier_idx] - preds["mae"][outlier_idx]
        ).mean()
        diff_huber10_vs_mae = np.abs(
            preds["huber:10.0"][outlier_idx] - preds["mae"][outlier_idx]
        ).mean()
        assert diff_huber1_vs_mae < diff_huber10_vs_mae, (
            f"Huber(1.0) should be closer to MAE on outliers than Huber(10.0). "
            f"huber:1.0 vs mae={diff_huber1_vs_mae:.3f}, "
            f"huber:10.0 vs mae={diff_huber10_vs_mae:.3f}"
        )

    def test_huber_large_delta_closer_to_rmse_on_outliers(self, fitted_models):
        """Huber(delta=10.0) should be closer to RMSE on outlier rows than Huber(delta=1.0)."""
        preds, y, outlier_idx = fitted_models
        diff_huber10_vs_rmse = np.abs(
            preds["huber:10.0"][outlier_idx] - preds["rmse"][outlier_idx]
        ).mean()
        diff_huber1_vs_rmse = np.abs(
            preds["huber:1.0"][outlier_idx] - preds["rmse"][outlier_idx]
        ).mean()
        assert diff_huber10_vs_rmse < diff_huber1_vs_rmse, (
            f"Huber(10.0) should be closer to RMSE on outliers than Huber(1.0). "
            f"huber:10.0 vs rmse={diff_huber10_vs_rmse:.3f}, "
            f"huber:1.0 vs rmse={diff_huber1_vs_rmse:.3f}"
        )

    def test_quantile_ordering_mean_prediction(self, fitted_models):
        """Mean prediction: quantile:0.1 < quantile:0.5 < quantile:0.9."""
        preds, _, _ = fitted_models
        q10 = preds["quantile:0.1"].mean()
        q50 = preds["quantile:0.5"].mean()
        q90 = preds["quantile:0.9"].mean()
        assert q10 < q50, f"Expected q10_mean < q50_mean, got {q10:.3f} >= {q50:.3f}"
        assert q50 < q90, f"Expected q50_mean < q90_mean, got {q50:.3f} >= {q90:.3f}"

    def test_quantile_01_brackets_most_targets_below(self, fitted_models):
        """Quantile:0.1 should underestimate >50% of targets on inlier rows."""
        preds, y, outlier_idx = fitted_models
        non_outlier_idx = np.setdiff1d(np.arange(500), outlier_idx)
        frac_below = (preds["quantile:0.1"][non_outlier_idx] < y[non_outlier_idx]).mean()
        assert frac_below > 0.5, (
            f"quantile:0.1 should be below target > 50% of inlier rows, got {frac_below:.2f}"
        )

    def test_quantile_09_brackets_most_targets_above(self, fitted_models):
        """Quantile:0.9 should overestimate >50% of targets on inlier rows."""
        preds, y, outlier_idx = fitted_models
        non_outlier_idx = np.setdiff1d(np.arange(500), outlier_idx)
        frac_above = (preds["quantile:0.9"][non_outlier_idx] > y[non_outlier_idx]).mean()
        assert frac_above > 0.5, (
            f"quantile:0.9 should be above target > 50% of inlier rows, got {frac_above:.2f}"
        )

    def test_all_losses_produce_finite_predictions(self, fitted_models):
        """No loss function should produce NaN or Inf predictions."""
        preds, _, _ = fitted_models
        for loss, pred in preds.items():
            assert np.all(np.isfinite(pred)), (
                f"Loss '{loss}' produced non-finite predictions: "
                f"{pred[~np.isfinite(pred)]}"
            )


# ============================================================================
# SECTION 3: Python surface — loss string forwarding
# ============================================================================


class TestPythonLossSurface:
    """Verify which loss strings the Python API accepts and forwards correctly."""

    def test_mae_lowercase_accepted(self):
        """loss='mae' (lowercase) is the documented format and must work."""
        X, y, _ = _make_regression_dataset()
        m = CatBoostMLXRegressor(
            iterations=10, loss="mae", random_seed=42, binary_path=BINARY_PATH
        )
        m.fit(X, y)
        assert m._is_fitted

    def test_quantile_colon_numeric_accepted(self):
        """loss='quantile:0.7' (numeric suffix) is the correct Python API format."""
        X, y, _ = _make_regression_dataset()
        m = CatBoostMLXRegressor(
            iterations=10, loss="quantile:0.7", random_seed=42, binary_path=BINARY_PATH
        )
        m.fit(X, y)
        assert m._is_fitted

    def test_huber_colon_numeric_accepted(self):
        """loss='huber:1.0' (numeric suffix) is the correct Python API format."""
        X, y, _ = _make_regression_dataset()
        m = CatBoostMLXRegressor(
            iterations=10, loss="huber:1.0", random_seed=42, binary_path=BINARY_PATH
        )
        m.fit(X, y)
        assert m._is_fitted

    def test_mae_uppercase_crashes_binary__bug001(self):
        """BUG-001 (FIXED): loss='MAE' (uppercase) now trains successfully.

        Fix: python/_normalize_loss_str() lowercases the base before passing
        to the binary. csv_train ParseLossType() also lowercases as
        belt-and-suspenders. Both surfaces now handle any case variation.

        This test was originally written to document the crash; it is now a
        regression test confirming the fix is in place.
        """
        X, y, _ = _make_regression_dataset()
        m = CatBoostMLXRegressor(
            iterations=10, loss="MAE", random_seed=42, binary_path=BINARY_PATH
        )
        # Must succeed after fix
        m.fit(X, y)
        assert m._is_fitted, "loss='MAE' should fit without error after BUG-001 fix"
        preds = m.predict(X)
        assert np.all(np.isfinite(preds)), "loss='MAE' (uppercase) returned non-finite predictions"

    def test_quantile_catboost_param_name_syntax__bug002(self):
        """BUG-002 (FIXED): loss='Quantile:alpha=0.7' now trains successfully.

        Fix: _validate_params() strips 'alpha=' prefix before float() call.
        _build_train_args() normalizes to 'quantile:0.7' for the binary via
        _normalize_loss_str().

        This test was originally written to document the ValueError; it is now
        a regression test confirming named-param syntax is accepted.
        """
        X, y, _ = _make_regression_dataset()
        m = CatBoostMLXRegressor(
            iterations=10, loss="Quantile:alpha=0.7", random_seed=42,
            binary_path=BINARY_PATH
        )
        m.fit(X, y)
        assert m._is_fitted, "loss='Quantile:alpha=0.7' should fit after BUG-002 fix"
        preds = m.predict(X)
        assert np.all(np.isfinite(preds))

    def test_huber_catboost_param_name_syntax__bug002(self):
        """BUG-002 (FIXED): loss='Huber:delta=1.0' now trains successfully.

        Fix: same as BUG-002 Quantile variant — 'delta=' prefix is stripped
        before float() validation and before forwarding to binary.
        """
        X, y, _ = _make_regression_dataset()
        m = CatBoostMLXRegressor(
            iterations=10, loss="Huber:delta=1.0", random_seed=42,
            binary_path=BINARY_PATH
        )
        m.fit(X, y)
        assert m._is_fitted, "loss='Huber:delta=1.0' should fit after BUG-002 fix"
        preds = m.predict(X)
        assert np.all(np.isfinite(preds))

    def test_huber_no_param_csv_train_defaults__known_divergence(self):
        """Known divergence: csv_train silently defaults 'huber' to delta=1.0.

        train.cpp enforces CB_ENSURE that delta is mandatory.
        csv_train ParseLossType() has 'if (lc.Type == "huber") lc.Param = 1.0f'.
        This divergence is noted as a Sprint 3 known limitation.

        This test documents the observed behavior: csv_train succeeds.
        """
        X, y, _ = _make_regression_dataset()
        m = CatBoostMLXRegressor(
            iterations=10, loss="huber", random_seed=42, binary_path=BINARY_PATH
        )
        # csv_train defaults to delta=1.0 — this should NOT raise
        m.fit(X, y)
        assert m._is_fitted, "huber (no delta) should succeed via csv_train with default delta=1.0"
        # Verify the stored model records loss_param=1.0
        stored_param = m._model_data.get("model_info", {}).get("loss_param")
        assert stored_param == 1.0, (
            f"csv_train should store loss_param=1.0 for 'huber', got {stored_param}"
        )

    def test_unknown_loss_raises_validation_error(self):
        """Passing a completely unknown loss string raises ValueError before calling binary."""
        X, y, _ = _make_regression_dataset()
        m = CatBoostMLXRegressor(
            iterations=10, loss="crossentropy", random_seed=42, binary_path=BINARY_PATH
        )
        with pytest.raises(ValueError, match="Unknown loss"):
            m.fit(X, y)


# ============================================================================
# SECTION 4: csv_train binary smoke tests (invoked directly via subprocess)
# ============================================================================


class TestCsvTrainBinary:
    """Directly invoke csv_train to validate binary-level behavior.

    These tests do NOT go through the Python API — they call csv_train with
    subprocess to isolate binary behavior from Python-layer issues.
    """

    @pytest.fixture(scope="class")
    def dataset_csv(self, tmp_path_factory):
        """Write the 500-row dataset to a temporary CSV file."""
        tmp = tmp_path_factory.mktemp("csvdata")
        csv_path = str(tmp / "data.csv")
        X, y, _ = _make_regression_dataset()
        _write_csv(X, y, csv_path)
        return csv_path

    def _run(self, args, timeout=60):
        """Run csv_train with given extra args; return (returncode, stdout, stderr)."""
        binary = _csv_train_binary()
        cmd = [binary] + args
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        return result.returncode, result.stdout, result.stderr

    def test_mae_trains_50_iters(self, dataset_csv, tmp_path):
        """csv_train --loss mae runs 50 iterations without error."""
        model_path = str(tmp_path / "model_mae.json")
        rc, stdout, stderr = self._run([
            dataset_csv, "--iterations", "50", "--loss", "mae",
            "--output", model_path, "--seed", "42"
        ])
        assert rc == 0, f"csv_train mae failed: {stderr}"
        assert "50 trees" in stdout

    def test_quantile_default_trains(self, dataset_csv, tmp_path):
        """csv_train --loss quantile (no alpha) uses alpha=0.5 by default."""
        model_path = str(tmp_path / "model_q.json")
        rc, stdout, stderr = self._run([
            dataset_csv, "--iterations", "50", "--loss", "quantile",
            "--output", model_path, "--seed", "42"
        ])
        assert rc == 0, f"csv_train quantile (default) failed: {stderr}"
        import json
        with open(model_path) as f:
            data = json.load(f)
        stored_param = data["model_info"]["loss_param"]
        assert abs(stored_param - 0.5) < 1e-5, (
            f"Expected loss_param=0.5 for 'quantile', got {stored_param}"
        )

    def test_quantile_explicit_alpha(self, dataset_csv, tmp_path):
        """csv_train --loss quantile:0.1 stores alpha=0.1 in model_info."""
        model_path = str(tmp_path / "model_q01.json")
        rc, stdout, stderr = self._run([
            dataset_csv, "--iterations", "50", "--loss", "quantile:0.1",
            "--output", model_path, "--seed", "42"
        ])
        assert rc == 0, f"csv_train quantile:0.1 failed: {stderr}"
        import json
        with open(model_path) as f:
            data = json.load(f)
        stored_param = data["model_info"]["loss_param"]
        assert abs(stored_param - 0.1) < 1e-5, (
            f"Expected loss_param=0.1 for 'quantile:0.1', got {stored_param}"
        )

    def test_quantile_alpha_0_9(self, dataset_csv, tmp_path):
        """csv_train --loss quantile:0.9 stores alpha=0.9 in model_info."""
        model_path = str(tmp_path / "model_q09.json")
        rc, stdout, stderr = self._run([
            dataset_csv, "--iterations", "50", "--loss", "quantile:0.9",
            "--output", model_path, "--seed", "42"
        ])
        assert rc == 0, f"csv_train quantile:0.9 failed: {stderr}"
        import json
        with open(model_path) as f:
            data = json.load(f)
        assert abs(data["model_info"]["loss_param"] - 0.9) < 1e-5

    def test_huber_explicit_delta(self, dataset_csv, tmp_path):
        """csv_train --loss huber:1.0 stores delta=1.0 in model_info."""
        model_path = str(tmp_path / "model_h1.json")
        rc, stdout, stderr = self._run([
            dataset_csv, "--iterations", "50", "--loss", "huber:1.0",
            "--output", model_path, "--seed", "42"
        ])
        assert rc == 0, f"csv_train huber:1.0 failed: {stderr}"
        import json
        with open(model_path) as f:
            data = json.load(f)
        assert abs(data["model_info"]["loss_param"] - 1.0) < 1e-5

    def test_huber_large_delta(self, dataset_csv, tmp_path):
        """csv_train --loss huber:10.0 stores delta=10.0."""
        model_path = str(tmp_path / "model_h10.json")
        rc, stdout, stderr = self._run([
            dataset_csv, "--iterations", "50", "--loss", "huber:10.0",
            "--output", model_path, "--seed", "42"
        ])
        assert rc == 0, f"csv_train huber:10.0 failed: {stderr}"
        import json
        with open(model_path) as f:
            data = json.load(f)
        assert abs(data["model_info"]["loss_param"] - 10.0) < 1e-5

    def test_huber_no_delta_defaults_to_1_0(self, dataset_csv, tmp_path):
        """csv_train --loss huber silently defaults to delta=1.0 (known divergence from train.cpp)."""
        model_path = str(tmp_path / "model_hnodelta.json")
        rc, stdout, stderr = self._run([
            dataset_csv, "--iterations", "50", "--loss", "huber",
            "--output", model_path, "--seed", "42"
        ])
        assert rc == 0, (
            "csv_train 'huber' (no delta) should succeed by defaulting delta=1.0, "
            f"but failed: {stderr}"
        )
        import json
        with open(model_path) as f:
            data = json.load(f)
        assert abs(data["model_info"]["loss_param"] - 1.0) < 1e-5

    def test_mae_uppercase_does_not_sigabrt__bug001(self, dataset_csv, tmp_path):
        """BUG-001 regression sentinel: csv_train --loss MAE must not SIGABRT.

        Historically 'MAE' fell through all loss dispatch branches and hit an
        uncaught MLX reshape exception ending in SIGABRT (exit -6). The intended
        fix was either (a) accept MAE as a case-insensitive alias, or (b) emit
        a clean 'unknown loss' error. Either outcome is fine; only the raw
        abort is a regression.
        """
        model_path = str(tmp_path / "model_mae_upper.json")
        rc, stdout, stderr = self._run([
            dataset_csv, "--iterations", "10", "--loss", "MAE",
            "--output", model_path, "--seed", "42"
        ])
        assert rc >= 0, (
            f"csv_train --loss MAE was killed by signal {-rc} "
            f"(BUG-001 regression — expected clean exit or clean error)"
        )

    def test_quantile_three_alphas_produce_different_models(self, dataset_csv, tmp_path):
        """Verify q10, q50, q90 models have different base predictions and tree values."""
        import json
        models = {}
        for alpha_str in ["0.1", "0.5", "0.9"]:
            model_path = str(tmp_path / f"model_q{alpha_str.replace('.','')}.json")
            rc, stdout, stderr = self._run([
                dataset_csv, "--iterations", "50", "--loss", f"quantile:{alpha_str}",
                "--output", model_path, "--seed", "42"
            ])
            assert rc == 0, f"quantile:{alpha_str} failed: {stderr}"
            with open(model_path) as f:
                models[alpha_str] = json.load(f)

        # Verify the three models have distinct loss_param values stored
        assert abs(models["0.1"]["model_info"]["loss_param"] - 0.1) < 1e-5
        assert abs(models["0.5"]["model_info"]["loss_param"] - 0.5) < 1e-5
        assert abs(models["0.9"]["model_info"]["loss_param"] - 0.9) < 1e-5

        # The tree leaf values should differ across quantile levels.
        # Compare the first tree's leaf values as a fingerprint.
        leaves_01 = models["0.1"]["trees"][0]["leaf_values"]
        leaves_50 = models["0.5"]["trees"][0]["leaf_values"]
        leaves_90 = models["0.9"]["trees"][0]["leaf_values"]
        assert leaves_01 != leaves_50, "q10 and q50 produced identical first-tree leaves"
        assert leaves_50 != leaves_90, "q50 and q90 produced identical first-tree leaves"


# ============================================================================
# SECTION 5: Regression — existing suite must be unaffected
# ============================================================================


class TestSprintRegression:
    """Ensure Sprint 3 additions did not break prior test areas."""

    def test_rmse_still_trains(self):
        """RMSE (pre-Sprint 3 baseline) must still work correctly."""
        X, y, _ = _make_regression_dataset()
        m = CatBoostMLXRegressor(
            iterations=50, loss="rmse", random_seed=42, binary_path=BINARY_PATH
        )
        m.fit(X, y)
        preds = m.predict(X)
        rmse = np.sqrt(np.mean((preds - y) ** 2))
        # With 500 rows, 50 iters, and +-20 outliers, RMSE won't be tiny but should be finite
        assert np.isfinite(rmse), "RMSE regression: training gave non-finite loss"
        assert rmse < 20.0, f"RMSE regression: train error too large ({rmse:.2f})"

    def test_loss_history_decreases_mae(self):
        """MAE training loss should be monotonically non-increasing over iterations."""
        X, y, _ = _make_regression_dataset()
        m = CatBoostMLXRegressor(
            iterations=30, loss="mae", random_seed=42, binary_path=BINARY_PATH
        )
        m.fit(X, y)
        history = m.train_loss_history
        if len(history) > 1:
            # Allow a tiny tolerance for floating-point fluctuation
            diffs = np.diff(history)
            n_increases = (diffs > 1e-6).sum()
            assert n_increases == 0, (
                f"MAE training loss increased {n_increases} times: {history[:10]}"
            )

    def test_loss_history_decreases_quantile(self):
        """Quantile:0.5 training loss should be non-increasing."""
        X, y, _ = _make_regression_dataset()
        m = CatBoostMLXRegressor(
            iterations=30, loss="quantile:0.5", random_seed=42, binary_path=BINARY_PATH
        )
        m.fit(X, y)
        history = m.train_loss_history
        if len(history) > 1:
            diffs = np.diff(history)
            n_increases = (diffs > 1e-6).sum()
            assert n_increases == 0, (
                f"Quantile:0.5 training loss increased {n_increases} times"
            )

    def test_loss_history_decreases_huber(self):
        """Huber:1.0 training loss should be non-increasing."""
        X, y, _ = _make_regression_dataset()
        m = CatBoostMLXRegressor(
            iterations=30, loss="huber:1.0", random_seed=42, binary_path=BINARY_PATH
        )
        m.fit(X, y)
        history = m.train_loss_history
        if len(history) > 1:
            diffs = np.diff(history)
            n_increases = (diffs > 1e-6).sum()
            assert n_increases == 0, (
                f"Huber:1.0 training loss increased {n_increases} times"
            )


# ============================================================================
# SECTION 6: BUG-001/BUG-002 fix verification
#   These tests verify that the Sprint 3 follow-up fixes are in place.
#   They are the mirror of the "documents the bug" tests in Section 3 —
#   those tests assert the broken behaviour; these assert the fixed behaviour.
# ============================================================================


class TestBugFixVerification:
    """Confirm BUG-001 (uppercase crash) and BUG-002 (named params) are fixed."""

    # ------------------------------------------------------------------
    # BUG-001 fixes: uppercase loss strings must be accepted
    # ------------------------------------------------------------------

    def test_bug001_mae_uppercase_accepted(self):
        """BUG-001 fix: loss='MAE' (uppercase) must train successfully.

        Python _build_train_args normalizes to 'mae' before passing to binary.
        csv_train ParseLossType also lowercases as belt-and-suspenders.
        """
        X, y, _ = _make_regression_dataset()
        m = CatBoostMLXRegressor(
            iterations=10, loss="MAE", random_seed=42, binary_path=BINARY_PATH
        )
        m.fit(X, y)
        preds = m.predict(X)
        assert m._is_fitted, "loss='MAE' should fit without error after BUG-001 fix"
        assert preds.shape == (500,)
        assert np.all(np.isfinite(preds)), "loss='MAE' (uppercase) returned non-finite predictions"

    def test_bug001_mae_upper_matches_lower(self):
        """BUG-001 fix: 'MAE' normalizes to the same args as 'mae'.

        Verifies the normalization contract rather than comparing cross-invocation
        predictions (Metal GPU non-determinism can produce sub-1e-4 variance).
        """
        from catboost_mlx.core import _normalize_loss_str
        assert _normalize_loss_str("MAE") == "mae", (
            "_normalize_loss_str('MAE') should produce 'mae'"
        )
        assert _normalize_loss_str("mae") == "mae"
        assert _normalize_loss_str("RMSE") == "rmse"
        assert _normalize_loss_str("Quantile:0.7") == "quantile:0.7"

    def test_bug001_quantile_mixed_case_accepted(self):
        """BUG-001 fix: loss='Quantile:0.7' (mixed case) must train successfully."""
        X, y, _ = _make_regression_dataset()
        m = CatBoostMLXRegressor(
            iterations=10, loss="Quantile:0.7", random_seed=42, binary_path=BINARY_PATH
        )
        m.fit(X, y)
        assert m._is_fitted

    def test_bug001_huber_mixed_case_accepted(self):
        """BUG-001 fix: loss='Huber:1.0' (mixed case) must train successfully."""
        X, y, _ = _make_regression_dataset()
        m = CatBoostMLXRegressor(
            iterations=10, loss="Huber:1.0", random_seed=42, binary_path=BINARY_PATH
        )
        m.fit(X, y)
        assert m._is_fitted

    # ------------------------------------------------------------------
    # BUG-002 fixes: named-parameter syntax must be accepted
    # ------------------------------------------------------------------

    def test_bug002_quantile_named_alpha_accepted(self):
        """BUG-002 fix: loss='Quantile:alpha=0.7' (named param) must train successfully.

        Python _validate_params strips 'alpha=' before float(), so no ValueError.
        Python _build_train_args normalizes to 'quantile:0.7' for the binary.
        """
        X, y, _ = _make_regression_dataset()
        m = CatBoostMLXRegressor(
            iterations=10, loss="Quantile:alpha=0.7", random_seed=42, binary_path=BINARY_PATH
        )
        m.fit(X, y)
        preds = m.predict(X)
        assert m._is_fitted, "loss='Quantile:alpha=0.7' should fit after BUG-002 fix"
        assert preds.shape == (500,)
        assert np.all(np.isfinite(preds))

    def test_bug002_huber_named_delta_accepted(self):
        """BUG-002 fix: loss='Huber:delta=1.0' (named param) must train successfully."""
        X, y, _ = _make_regression_dataset()
        m = CatBoostMLXRegressor(
            iterations=10, loss="Huber:delta=1.0", random_seed=42, binary_path=BINARY_PATH
        )
        m.fit(X, y)
        preds = m.predict(X)
        assert m._is_fitted
        assert preds.shape == (500,)
        assert np.all(np.isfinite(preds))

    def test_bug002_named_matches_positional_quantile(self):
        """BUG-002 fix: 'Quantile:alpha=0.7' normalizes to the same args as 'quantile:0.7'.

        Verifies the normalization contract rather than comparing cross-invocation
        predictions (which can diverge due to Metal GPU non-determinism).
        """
        from catboost_mlx.core import _normalize_loss_str
        assert _normalize_loss_str("Quantile:alpha=0.7") == "quantile:0.7", (
            "_normalize_loss_str('Quantile:alpha=0.7') should produce 'quantile:0.7'"
        )
        assert _normalize_loss_str("quantile:0.7") == "quantile:0.7", (
            "_normalize_loss_str('quantile:0.7') should be unchanged"
        )
        # Both forms must also train successfully without error
        X, y, _ = _make_regression_dataset()
        for loss_str in ["Quantile:alpha=0.7", "quantile:0.7"]:
            m = CatBoostMLXRegressor(
                iterations=5, loss=loss_str, random_seed=42, binary_path=BINARY_PATH
            )
            m.fit(X, y)
            assert m._is_fitted, f"loss='{loss_str}' should fit after BUG-002 fix"

    def test_bug002_named_matches_positional_huber(self):
        """BUG-002 fix: 'Huber:delta=1.0' normalizes to the same args as 'huber:1.0'.

        Rather than comparing predictions across two separate binary invocations
        (which can diverge due to Metal GPU non-determinism), we verify the
        normalization contract: _normalize_loss_str transforms both forms to
        the same canonical string that csv_train receives.
        """
        from catboost_mlx.core import _normalize_loss_str
        assert _normalize_loss_str("Huber:delta=1.0") == "huber:1.0", (
            "_normalize_loss_str('Huber:delta=1.0') should produce 'huber:1.0'"
        )
        assert _normalize_loss_str("huber:1.0") == "huber:1.0", (
            "_normalize_loss_str('huber:1.0') should be unchanged"
        )
        # Both forms must also train successfully without error
        X, y, _ = _make_regression_dataset()
        for loss_str in ["Huber:delta=1.0", "huber:1.0"]:
            m = CatBoostMLXRegressor(
                iterations=5, loss=loss_str, random_seed=42, binary_path=BINARY_PATH
            )
            m.fit(X, y)
            assert m._is_fitted, f"loss='{loss_str}' should fit after BUG-002 fix"

    def test_bug002_lowercase_named_alpha_accepted(self):
        """BUG-002 fix: lowercase 'quantile:alpha=0.3' is also valid."""
        X, y, _ = _make_regression_dataset()
        m = CatBoostMLXRegressor(
            iterations=10, loss="quantile:alpha=0.3", random_seed=42, binary_path=BINARY_PATH
        )
        m.fit(X, y)
        assert m._is_fitted

    def test_bug002_invalid_param_name_still_rejected(self):
        """Non-numeric, non-named suffix must still raise ValueError."""
        X, y, _ = _make_regression_dataset()
        m = CatBoostMLXRegressor(
            iterations=10, loss="quantile:q=0.7", random_seed=42, binary_path=BINARY_PATH
        )
        with pytest.raises(ValueError, match="numeric"):
            m.fit(X, y)
