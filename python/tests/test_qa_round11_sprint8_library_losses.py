"""
test_qa_round11_sprint8_library_losses.py -- QA Round 11: Poisson, Tweedie, MAPE loss validation.

TODO-011 added Poisson, Tweedie, and MAPE loss functions to the library path
(pointwise_target.h + train.cpp switch cases). The Python API exercises these
losses via the csv_train binary, which has had its own implementations since
earlier sprints.

This file validates:
  1. Poisson regression: exp-link predictions, non-negative outputs, loss descent
  2. Tweedie regression: default p=1.5, custom powers (1.2, 1.8), boundary powers
  3. MAPE regression: positive-target datasets, different scales, loss descent
  4. Edge cases: near-zero targets, boundary power values, single iteration
  5. Parameter validation: invalid Tweedie power string, variance_power= named param

Bugs found during QA:
  BUG-004: loss='tweedie:variance_power=1.5' raises ValueError instead of training.
           _validate_params() strips 'alpha=' and 'delta=' prefixes but NOT
           'variance_power='. float('variance_power=1.5') then throws ValueError.
           Workaround: use positional syntax 'tweedie:1.5'.

Focus areas:
  1. Poisson output domain (predictions must be non-negative; model uses exp link)
  2. Tweedie parameter parsing and effect of variance_power on predictions
  3. MAPE with strictly positive and near-zero targets
  4. Loss decreases over training (sanity check on gradient direction)
  5. Python API: which strings are accepted/rejected
"""

import os

import numpy as np
import pytest

from catboost_mlx import CatBoostMLXRegressor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BINARY_PATH = REPO_ROOT

# Small dataset parameters for speed
_N = 300
_N_FEATURES = 6
_SEED = 42


# ---------------------------------------------------------------------------
# Dataset factories
# ---------------------------------------------------------------------------

def _make_count_dataset(n=_N, n_features=_N_FEATURES, seed=_SEED):
    """Count regression dataset with non-negative integer-like targets.

    Signal: y = Poisson(lambda=exp(X @ coef)), with coef in [-1, 1].
    Targets are strictly non-negative (always >= 0).
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    coef = rng.uniform(-1.0, 1.0, size=n_features)
    log_lam = X @ coef
    # Clip to avoid extreme lambda values
    log_lam = np.clip(log_lam, -2.0, 2.0)
    lam = np.exp(log_lam)
    y = rng.poisson(lam).astype(float)
    # Ensure at least a few non-zero targets to avoid degenerate data
    y = np.maximum(y, 0.0)
    return X, y


def _make_nonneg_dataset(n=_N, n_features=_N_FEATURES, seed=_SEED):
    """Non-negative continuous regression dataset.

    Signal: y = exp(X @ coef) * noise, so targets are always > 0.
    Suitable for Tweedie regression.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    coef = rng.uniform(-0.5, 0.5, size=n_features)
    log_y = X @ coef + rng.standard_normal(n) * 0.3
    log_y = np.clip(log_y, -2.0, 2.0)
    y = np.exp(log_y)  # always positive
    return X, y


def _make_positive_dataset(n=_N, n_features=_N_FEATURES, seed=_SEED):
    """Strictly positive dataset for MAPE testing.

    Signal: y = 2 + |X @ coef| + noise, so y > 0 always.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    coef = np.array([1.5, -1.0, 0.8, -0.5, 0.3, -0.2])[:n_features]
    y = 2.0 + np.abs(X @ coef) + rng.uniform(0.1, 0.5, size=n)
    return X, y


def _make_large_scale_dataset(n=_N, n_features=_N_FEATURES, seed=_SEED):
    """Strictly positive targets at large scale (y in [100, 1000]) for MAPE."""
    X, y_small = _make_positive_dataset(n=n, n_features=n_features, seed=seed)
    y = y_small * 100.0
    return X, y


def _make_small_scale_dataset(n=_N, n_features=_N_FEATURES, seed=_SEED):
    """Strictly positive targets at small scale (y in [0.01, 1.0]) for MAPE."""
    X, y_large = _make_positive_dataset(n=n, n_features=n_features, seed=seed)
    y = y_large / 100.0
    return X, y


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fit_and_predict(loss, X, y, n_iterations=20, seed=_SEED):
    """Fit a regressor and return (model, predictions)."""
    m = CatBoostMLXRegressor(
        iterations=n_iterations,
        loss=loss,
        random_seed=seed,
        binary_path=BINARY_PATH,
    )
    m.fit(X, y)
    preds = m.predict(X)
    return m, preds


# ============================================================================
# SECTION 1: Poisson regression
# ============================================================================


class TestPoissonRegression:
    """Validate Poisson regression (log-link) end-to-end.

    Poisson uses exp(cursor) as the prediction, so all predictions must be
    strictly positive. The loss is the Poisson negative log-likelihood.
    """

    @pytest.fixture(scope="class")
    def count_data(self):
        return _make_count_dataset()

    def test_poisson_basic_fit_predict(self, count_data):
        """Poisson trains without error and returns a prediction array of correct shape."""
        X, y = count_data
        m, preds = _fit_and_predict("poisson", X, y)
        assert m._is_fitted, "Model should be marked as fitted after fit()"
        assert preds.shape == (len(y),), (
            f"Expected shape ({len(y)},), got {preds.shape}"
        )

    def test_poisson_predictions_non_negative(self, count_data):
        """All Poisson predictions must be >= 0 (exp-link guarantees this).

        The model predicts exp(cursor), so the output should always be
        non-negative. A negative prediction would indicate the binary is
        not applying the link function.
        """
        X, y = count_data
        _, preds = _fit_and_predict("poisson", X, y)
        assert np.all(preds >= 0.0), (
            f"Poisson predictions must be non-negative. "
            f"Min prediction: {preds.min():.6f}"
        )

    def test_poisson_predictions_finite(self, count_data):
        """Poisson predictions must be finite (no NaN or Inf)."""
        X, y = count_data
        _, preds = _fit_and_predict("poisson", X, y)
        assert np.all(np.isfinite(preds)), (
            f"Poisson produced non-finite predictions: {preds[~np.isfinite(preds)]}"
        )

    def test_poisson_predictions_strictly_positive(self, count_data):
        """Poisson predictions from exp-link should be strictly positive (> 0).

        exp(cursor) is always > 0 for any finite cursor value.
        """
        X, y = count_data
        _, preds = _fit_and_predict("poisson", X, y)
        assert np.all(preds > 0.0), (
            f"Poisson predictions should be strictly positive (exp link). "
            f"Found {np.sum(preds <= 0)} non-positive values."
        )

    def test_poisson_more_iterations_reduces_train_loss(self, count_data):
        """More training iterations must yield a lower (or equal) training loss.

        This verifies the gradient direction is correct and the model is
        actually learning.
        """
        X, y = count_data

        def poisson_loss(preds, targets):
            """Poisson NLL: mean(exp(log_preds) - targets * log_preds).
            Since preds = exp(cursor), log_preds = log(preds).
            NLL = mean(preds - targets * log(preds)).
            """
            eps = 1e-10
            return np.mean(preds - targets * np.log(np.maximum(preds, eps)))

        _, preds_10 = _fit_and_predict("poisson", X, y, n_iterations=10)
        _, preds_30 = _fit_and_predict("poisson", X, y, n_iterations=30)

        loss_10 = poisson_loss(preds_10, y)
        loss_30 = poisson_loss(preds_30, y)
        assert loss_30 <= loss_10 + 0.01, (
            f"Poisson: 30-iteration model loss ({loss_30:.4f}) should be <= "
            f"10-iteration model loss ({loss_10:.4f})"
        )

    def test_poisson_with_very_small_targets(self):
        """Poisson on near-zero count data (mix of 0s and 1s) must not crash."""
        rng = np.random.default_rng(seed=7)
        X = rng.standard_normal((200, 5))
        # Near-zero counts: mostly 0s and 1s
        y = rng.integers(0, 2, size=200).astype(float)
        m, preds = _fit_and_predict("poisson", X, y, n_iterations=10)
        assert m._is_fitted
        assert np.all(np.isfinite(preds)), "Poisson on near-zero targets produced non-finite output"
        assert np.all(preds >= 0.0), "Poisson on near-zero targets produced negative predictions"

    def test_poisson_single_feature_single_iteration(self):
        """Poisson with 1 feature and 1 iteration must not crash."""
        rng = np.random.default_rng(seed=99)
        X = rng.standard_normal((200, 1))
        y = rng.poisson(2.0, size=200).astype(float)
        m, preds = _fit_and_predict("poisson", X, y, n_iterations=1)
        assert m._is_fitted
        assert preds.shape == (200,)
        assert np.all(np.isfinite(preds))

    def test_poisson_uppercase_loss_accepted(self, count_data):
        """loss='POISSON' (uppercase) must be normalized and accepted."""
        X, y = count_data
        m, preds = _fit_and_predict("POISSON", X, y, n_iterations=10)
        assert m._is_fitted, "loss='POISSON' should be case-normalized and accepted"
        assert np.all(np.isfinite(preds))


# ============================================================================
# SECTION 2: Tweedie regression
# ============================================================================


class TestTweedieRegression:
    """Validate Tweedie regression with various variance power values.

    Tweedie is a generalization of Poisson (p=1) and Gamma (p=2).
    The variance power p must be in (1, 2) for the compound Poisson-Gamma.
    csv_train defaults p=1.5 when no parameter is given.
    """

    @pytest.fixture(scope="class")
    def nonneg_data(self):
        return _make_nonneg_dataset()

    def test_tweedie_default_trains_successfully(self, nonneg_data):
        """loss='tweedie' (no power suffix) trains with default p=1.5."""
        X, y = nonneg_data
        m, preds = _fit_and_predict("tweedie", X, y)
        assert m._is_fitted
        assert preds.shape == (len(y),)

    def test_tweedie_default_predictions_finite(self, nonneg_data):
        """Tweedie default predictions must be finite."""
        X, y = nonneg_data
        _, preds = _fit_and_predict("tweedie", X, y)
        assert np.all(np.isfinite(preds)), (
            f"Tweedie(default) produced non-finite predictions"
        )

    def test_tweedie_default_matches_explicit_15(self, nonneg_data):
        """loss='tweedie' and loss='tweedie:1.5' must produce identical predictions.

        The csv_train binary defaults p=1.5 when no suffix is given.
        """
        X, y = nonneg_data
        _, preds_default = _fit_and_predict("tweedie", X, y, n_iterations=15)
        _, preds_explicit = _fit_and_predict("tweedie:1.5", X, y, n_iterations=15)
        assert np.allclose(preds_default, preds_explicit, atol=1e-4), (
            "tweedie (no power) should match tweedie:1.5. "
            f"Max diff: {np.abs(preds_default - preds_explicit).max():.6f}"
        )

    def test_tweedie_power_1_2_trains(self, nonneg_data):
        """Tweedie with p=1.2 (near-Poisson) must train without error."""
        X, y = nonneg_data
        m, preds = _fit_and_predict("tweedie:1.2", X, y)
        assert m._is_fitted
        assert np.all(np.isfinite(preds))

    def test_tweedie_power_1_8_trains(self, nonneg_data):
        """Tweedie with p=1.8 (near-Gamma) must train without error."""
        X, y = nonneg_data
        m, preds = _fit_and_predict("tweedie:1.8", X, y)
        assert m._is_fitted
        assert np.all(np.isfinite(preds))

    def test_tweedie_different_powers_produce_different_predictions(self, nonneg_data):
        """p=1.2 and p=1.8 must produce meaningfully different predictions.

        Changing variance power shifts the distribution assumption; models
        trained with different powers should diverge on the same data.
        """
        X, y = nonneg_data
        _, preds_12 = _fit_and_predict("tweedie:1.2", X, y, n_iterations=20)
        _, preds_18 = _fit_and_predict("tweedie:1.8", X, y, n_iterations=20)
        max_diff = np.abs(preds_12 - preds_18).max()
        assert max_diff > 0.01, (
            f"tweedie:1.2 and tweedie:1.8 predictions are nearly identical "
            f"(max_diff={max_diff:.6f}); expected meaningful divergence"
        )

    def test_tweedie_power_boundary_1_01(self, nonneg_data):
        """Tweedie with p=1.01 (boundary near-Poisson) must not crash."""
        X, y = nonneg_data
        m, preds = _fit_and_predict("tweedie:1.01", X, y, n_iterations=10)
        assert m._is_fitted
        assert np.all(np.isfinite(preds)), (
            "tweedie:1.01 (near-Poisson boundary) produced non-finite predictions"
        )

    def test_tweedie_power_boundary_1_99(self, nonneg_data):
        """Tweedie with p=1.99 (boundary near-Gamma) must not crash."""
        X, y = nonneg_data
        m, preds = _fit_and_predict("tweedie:1.99", X, y, n_iterations=10)
        assert m._is_fitted
        assert np.all(np.isfinite(preds)), (
            "tweedie:1.99 (near-Gamma boundary) produced non-finite predictions"
        )

    def test_tweedie_loss_decreases_over_iterations(self, nonneg_data):
        """More iterations must yield lower or equal Tweedie deviance on training data."""
        X, y = nonneg_data
        p = 1.5

        def tweedie_deviance(preds, targets, power):
            """Tweedie deviance (mean): -t*exp((1-p)*log_mu)/(1-p) + exp((2-p)*log_mu)/(2-p)."""
            eps = 1e-10
            log_preds = np.log(np.maximum(preds, eps))
            term1 = -targets * np.exp((1.0 - power) * log_preds) / (1.0 - power)
            term2 = np.exp((2.0 - power) * log_preds) / (2.0 - power)
            return np.mean(term1 + term2)

        _, preds_5 = _fit_and_predict("tweedie:1.5", X, y, n_iterations=5)
        _, preds_25 = _fit_and_predict("tweedie:1.5", X, y, n_iterations=25)

        dev_5 = tweedie_deviance(preds_5, y, p)
        dev_25 = tweedie_deviance(preds_25, y, p)
        assert dev_25 <= dev_5 + 0.01, (
            f"Tweedie:1.5 deviance should decrease with more iterations. "
            f"5 iters: {dev_5:.4f}, 25 iters: {dev_25:.4f}"
        )

    def test_tweedie_uppercase_accepted(self, nonneg_data):
        """loss='TWEEDIE' (uppercase, no power) must be normalized and accepted."""
        X, y = nonneg_data
        m, preds = _fit_and_predict("TWEEDIE", X, y, n_iterations=10)
        assert m._is_fitted, "loss='TWEEDIE' should be case-normalized and accepted"
        assert np.all(np.isfinite(preds))

    def test_tweedie_single_feature_single_iteration(self):
        """Tweedie with 1 feature and 1 iteration must not crash."""
        rng = np.random.default_rng(seed=77)
        X = rng.standard_normal((200, 1))
        y = np.exp(rng.standard_normal(200) * 0.5)  # log-normal, always positive
        m, preds = _fit_and_predict("tweedie:1.5", X, y, n_iterations=1)
        assert m._is_fitted
        assert preds.shape == (200,)
        assert np.all(np.isfinite(preds))


# ============================================================================
# SECTION 3: MAPE regression
# ============================================================================


class TestMapeRegression:
    """Validate MAPE (Mean Absolute Percentage Error) regression end-to-end.

    MAPE divides residuals by |target|, so the behavior is undefined at
    target=0. The implementation clamps |target| to epsilon=1e-6, which
    means near-zero targets are handled without crashing but may produce
    large gradients.
    """

    @pytest.fixture(scope="class")
    def positive_data(self):
        return _make_positive_dataset()

    def test_mape_basic_fit_predict(self, positive_data):
        """MAPE trains without error and returns correct-shape predictions."""
        X, y = positive_data
        m, preds = _fit_and_predict("mape", X, y)
        assert m._is_fitted
        assert preds.shape == (len(y),)

    def test_mape_predictions_finite(self, positive_data):
        """MAPE predictions must be finite on strictly positive targets."""
        X, y = positive_data
        _, preds = _fit_and_predict("mape", X, y)
        assert np.all(np.isfinite(preds)), (
            f"MAPE produced non-finite predictions on positive data"
        )

    def test_mape_loss_decreases_over_iterations(self, positive_data):
        """More iterations must yield a lower or equal MAPE on training data."""
        X, y = positive_data

        def mape(preds, targets):
            eps = 1e-6
            return np.mean(np.abs(preds - targets) / np.maximum(np.abs(targets), eps))

        _, preds_5 = _fit_and_predict("mape", X, y, n_iterations=5)
        _, preds_25 = _fit_and_predict("mape", X, y, n_iterations=25)

        loss_5 = mape(preds_5, y)
        loss_25 = mape(preds_25, y)
        assert loss_25 <= loss_5 + 0.01, (
            f"MAPE: 25-iteration loss ({loss_25:.4f}) should be <= "
            f"5-iteration loss ({loss_5:.4f})"
        )

    def test_mape_large_scale_targets(self):
        """MAPE on large-scale targets (y ~ [100, 1000]) must produce finite predictions."""
        X, y = _make_large_scale_dataset()
        _, preds = _fit_and_predict("mape", X, y, n_iterations=15)
        assert np.all(np.isfinite(preds)), (
            "MAPE on large-scale targets produced non-finite predictions"
        )

    def test_mape_small_scale_targets(self):
        """MAPE on small-scale targets (y ~ [0.01, 1.0]) must produce finite predictions."""
        X, y = _make_small_scale_dataset()
        _, preds = _fit_and_predict("mape", X, y, n_iterations=15)
        assert np.all(np.isfinite(preds)), (
            "MAPE on small-scale targets produced non-finite predictions"
        )

    def test_mape_scale_invariance_property(self):
        """MAPE predictions should be proportionally scaled when targets are scaled.

        If targets are multiplied by a constant C, MAPE predictions should
        also scale by approximately C (percentage errors are scale-invariant
        in the limit). This is a soft check with loose tolerance.
        """
        X, y_small = _make_positive_dataset(seed=11)
        y_large = y_small * 10.0

        _, preds_small = _fit_and_predict("mape", X, y_small, n_iterations=20)
        _, preds_large = _fit_and_predict("mape", X, y_large, n_iterations=20)

        # Mean prediction should scale with the data
        mean_small = np.mean(preds_small)
        mean_large = np.mean(preds_large)
        ratio = mean_large / mean_small
        # MAPE is not perfectly scale-invariant due to initialization and finite
        # iterations, but the ratio should be somewhere in [5, 15]
        assert 5.0 <= ratio <= 15.0, (
            f"MAPE predictions on 10x-scaled targets had mean ratio={ratio:.2f}; "
            "expected roughly 10x scaling"
        )

    def test_mape_near_zero_targets_no_crash(self):
        """MAPE with near-zero targets (0.001-0.01) must not crash.

        The implementation clamps |target| to 1e-6, so even targets very
        close to zero should not cause division-by-zero or NaN.
        """
        rng = np.random.default_rng(seed=55)
        X = rng.standard_normal((200, 5))
        # Targets very close to zero but positive
        y = rng.uniform(0.001, 0.01, size=200)
        m, preds = _fit_and_predict("mape", X, y, n_iterations=10)
        assert m._is_fitted
        assert np.all(np.isfinite(preds)), (
            "MAPE on near-zero targets (0.001-0.01) produced non-finite predictions"
        )

    def test_mape_single_feature_single_iteration(self):
        """MAPE with 1 feature and 1 iteration must not crash."""
        rng = np.random.default_rng(seed=88)
        X = rng.standard_normal((200, 1))
        y = rng.uniform(1.0, 5.0, size=200)
        m, preds = _fit_and_predict("mape", X, y, n_iterations=1)
        assert m._is_fitted
        assert preds.shape == (200,)
        assert np.all(np.isfinite(preds))

    def test_mape_uppercase_accepted(self, positive_data):
        """loss='MAPE' (uppercase) must be normalized and accepted."""
        X, y = positive_data
        m, preds = _fit_and_predict("MAPE", X, y, n_iterations=10)
        assert m._is_fitted, "loss='MAPE' should be case-normalized and accepted"
        assert np.all(np.isfinite(preds))


# ============================================================================
# SECTION 4: Parameter validation
# ============================================================================


class TestParameterValidation:
    """Validate parameter parsing for Tweedie and error handling for bad inputs.

    These tests probe the Python-layer validation (core.py _validate_params)
    and document known behavior divergences.
    """

    def test_tweedie_numeric_suffix_accepted(self):
        """loss='tweedie:1.5' (numeric positional suffix) must be accepted."""
        rng = np.random.default_rng(seed=_SEED)
        X = rng.standard_normal((_N, _N_FEATURES))
        y = np.exp(rng.standard_normal(_N) * 0.3)
        m = CatBoostMLXRegressor(
            iterations=10, loss="tweedie:1.5", random_seed=_SEED,
            binary_path=BINARY_PATH,
        )
        m.fit(X, y)
        assert m._is_fitted

    def test_tweedie_invalid_power_string_raises_value_error(self):
        """loss='tweedie:abc' must raise ValueError before calling the binary.

        The Python validator calls float('abc') which raises ValueError.
        """
        rng = np.random.default_rng(seed=_SEED)
        X = rng.standard_normal((_N, _N_FEATURES))
        y = np.exp(rng.standard_normal(_N) * 0.3)
        m = CatBoostMLXRegressor(
            iterations=10, loss="tweedie:abc", random_seed=_SEED,
            binary_path=BINARY_PATH,
        )
        with pytest.raises(ValueError):
            m.fit(X, y)

    def test_tweedie_variance_power_named_param_accepted(self):
        """BUG-004 fixed: loss='Tweedie:variance_power=1.5' now trains successfully.

        _validate_params() and _normalize_loss_str() strip 'variance_power='
        prefix (same pattern as 'alpha=' and 'delta=').
        """
        rng = np.random.default_rng(seed=_SEED)
        X = rng.standard_normal((_N, _N_FEATURES))
        y = np.exp(rng.standard_normal(_N) * 0.3)
        m = CatBoostMLXRegressor(
            iterations=10, loss="Tweedie:variance_power=1.5", random_seed=_SEED,
            binary_path=BINARY_PATH,
        )
        m.fit(X, y)
        preds = m.predict(X)
        assert preds.shape == (_N,)
        assert np.all(np.isfinite(preds))

    def test_unknown_loss_raises_value_error(self):
        """Completely unknown loss string raises ValueError before binary is called."""
        rng = np.random.default_rng(seed=_SEED)
        X = rng.standard_normal((_N, _N_FEATURES))
        y = rng.standard_normal(_N)
        m = CatBoostMLXRegressor(
            iterations=10, loss="gamma", random_seed=_SEED,
            binary_path=BINARY_PATH,
        )
        with pytest.raises(ValueError, match="Unknown loss"):
            m.fit(X, y)

    def test_poisson_no_param_suffix_accepted(self):
        """loss='poisson' (no suffix) must be accepted — Poisson has no free parameter."""
        rng = np.random.default_rng(seed=_SEED)
        X = rng.standard_normal((_N, _N_FEATURES))
        y = rng.poisson(2.0, size=_N).astype(float)
        m = CatBoostMLXRegressor(
            iterations=10, loss="poisson", random_seed=_SEED,
            binary_path=BINARY_PATH,
        )
        m.fit(X, y)
        assert m._is_fitted

    def test_mape_no_param_suffix_accepted(self):
        """loss='mape' (no suffix) must be accepted — MAPE has no free parameter."""
        rng = np.random.default_rng(seed=_SEED)
        X = rng.standard_normal((_N, _N_FEATURES))
        y = rng.uniform(1.0, 5.0, size=_N)
        m = CatBoostMLXRegressor(
            iterations=10, loss="mape", random_seed=_SEED,
            binary_path=BINARY_PATH,
        )
        m.fit(X, y)
        assert m._is_fitted

    def test_tweedie_no_suffix_defaults_to_1_5(self):
        """loss='tweedie' (no suffix) trains with default p=1.5.

        csv_train ParseLossType() sets lc.Param = 1.5f when base type is
        'tweedie' and no colon suffix is present.
        """
        rng = np.random.default_rng(seed=_SEED)
        X = rng.standard_normal((_N, _N_FEATURES))
        y = np.exp(rng.standard_normal(_N) * 0.3)
        m_default = CatBoostMLXRegressor(
            iterations=15, loss="tweedie", random_seed=_SEED,
            binary_path=BINARY_PATH,
        )
        m_explicit = CatBoostMLXRegressor(
            iterations=15, loss="tweedie:1.5", random_seed=_SEED,
            binary_path=BINARY_PATH,
        )
        m_default.fit(X, y)
        m_explicit.fit(X, y)
        preds_default = m_default.predict(X)
        preds_explicit = m_explicit.predict(X)
        assert np.allclose(preds_default, preds_explicit, atol=1e-4), (
            "tweedie (no power) should match tweedie:1.5. "
            f"Max diff: {np.abs(preds_default - preds_explicit).max():.6f}"
        )


# ============================================================================
# SECTION 5: Cross-loss comparison on shared data
# ============================================================================


class TestCrossLossComparison:
    """Compare Poisson, Tweedie, and MAPE on the same non-negative dataset.

    Confirms that different loss functions produce meaningfully different
    predictions and all complete without error.
    """

    @pytest.fixture(scope="class")
    def shared_data(self):
        """Non-negative dataset used across all three losses."""
        return _make_nonneg_dataset(seed=123)

    @pytest.fixture(scope="class")
    def fitted_preds(self, shared_data):
        """Fit all three losses on the same data; share across tests."""
        X, y = shared_data
        losses = ["poisson", "tweedie:1.5", "mape"]
        preds = {}
        for loss in losses:
            m = CatBoostMLXRegressor(
                iterations=20, loss=loss, random_seed=_SEED,
                binary_path=BINARY_PATH,
            )
            m.fit(X, y)
            preds[loss] = m.predict(X)
        return preds

    def test_all_three_losses_produce_finite_predictions(self, fitted_preds):
        """All three losses must produce finite predictions on non-negative data."""
        for loss, preds in fitted_preds.items():
            assert np.all(np.isfinite(preds)), (
                f"Loss '{loss}' produced non-finite predictions"
            )

    def test_poisson_and_tweedie_produce_non_negative_outputs(self, fitted_preds):
        """Poisson and Tweedie both use exp-link; predictions must be non-negative."""
        for loss in ["poisson", "tweedie:1.5"]:
            preds = fitted_preds[loss]
            assert np.all(preds >= 0.0), (
                f"Loss '{loss}' (exp-link) produced negative predictions. "
                f"Min: {preds.min():.6f}"
            )

    def test_all_losses_produce_distinct_predictions(self, fitted_preds):
        """Poisson, Tweedie, and MAPE must produce non-identical predictions.

        Three distinct loss functions should not converge to the same model
        on this dataset.
        """
        losses = ["poisson", "tweedie:1.5", "mape"]
        for i in range(len(losses)):
            for j in range(i + 1, len(losses)):
                diff = np.abs(fitted_preds[losses[i]] - fitted_preds[losses[j]]).mean()
                assert diff > 1e-4, (
                    f"Losses '{losses[i]}' and '{losses[j]}' produced nearly "
                    f"identical predictions (mean diff={diff:.6f})"
                )

    def test_tweedie_12_and_18_diverge_from_15(self, shared_data):
        """Tweedie p=1.2 and p=1.8 should both diverge from the default p=1.5."""
        X, y = shared_data
        _, preds_12 = _fit_and_predict("tweedie:1.2", X, y, n_iterations=20)
        _, preds_15 = _fit_and_predict("tweedie:1.5", X, y, n_iterations=20)
        _, preds_18 = _fit_and_predict("tweedie:1.8", X, y, n_iterations=20)

        diff_12_15 = np.abs(preds_12 - preds_15).mean()
        diff_18_15 = np.abs(preds_18 - preds_15).mean()

        assert diff_12_15 > 1e-4, (
            f"tweedie:1.2 too similar to tweedie:1.5 (mean diff={diff_12_15:.6f})"
        )
        assert diff_18_15 > 1e-4, (
            f"tweedie:1.8 too similar to tweedie:1.5 (mean diff={diff_18_15:.6f})"
        )
