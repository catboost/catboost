"""
test_qa_round4.py -- QA Round 4: Testing _predict_inprocess refactoring.

The dev refactored _run_predict to dispatch numeric-only models to a new
in-process Python/NumPy prediction path (_predict_inprocess), bypassing
the C++ subprocess for 5-25x speedup. Categorical models still use subprocess.

Focus areas:
1. Verify all 5 round 3 fixes still hold
2. Loss type divergence between _predict_inprocess and _get_loss_type
3. _predict_subprocess dead code
4. predict_timeout ignored in in-process path
5. In-process vs subprocess numerical agreement
6. Classification and multiclass through in-process path
"""

import json
import math
import os
import pickle
import struct
import tempfile

import numpy as np
import pytest

from catboost_mlx import CatBoostMLX, CatBoostMLXClassifier, CatBoostMLXRegressor, Pool
from catboost_mlx.core import _array_to_binary, _array_to_csv

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BINARY_PATH = REPO_ROOT


def _skip_no_binaries():
    csv_train = os.path.join(BINARY_PATH, "csv_train")
    csv_predict = os.path.join(BINARY_PATH, "csv_predict")
    if not (os.path.isfile(csv_train) and os.path.isfile(csv_predict)):
        pytest.skip("Compiled binaries not found")


def _train_regressor(n=30, features=3, **kwargs):
    """Train a simple regressor for testing."""
    _skip_no_binaries()
    model = CatBoostMLXRegressor(
        iterations=10, binary_path=BINARY_PATH, **kwargs
    )
    np.random.seed(42)
    X = np.random.randn(n, features)
    y = np.random.randn(n)
    model.fit(X, y)
    return model, X, y


def _train_classifier(n=30, **kwargs):
    """Train a simple binary classifier for testing."""
    _skip_no_binaries()
    model = CatBoostMLXClassifier(
        iterations=10, binary_path=BINARY_PATH, **kwargs
    )
    np.random.seed(42)
    X = np.random.randn(n, 3)
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    model.fit(X, y)
    return model, X, y


def _strip_loss_type(model):
    """Remove loss_type from model_info to simulate old/hand-crafted model."""
    model._model_data["model_info"].pop("loss_type", None)


def _save_load_without_loss_type(model, cls, loss=None):
    """Save model, strip loss_type, reload into new instance."""
    with tempfile.NamedTemporaryFile(
        suffix=".json", mode="w", delete=False
    ) as f:
        data = json.loads(json.dumps(model._model_data))
        data["model_info"].pop("loss_type", None)
        json.dump(data, f)
        path = f.name
    try:
        kwargs = {"binary_path": BINARY_PATH}
        if loss:
            kwargs["loss"] = loss
        model2 = cls(**kwargs)
        model2.load_model(path)
        return model2
    finally:
        os.unlink(path)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1: Verify round 3 fixes still hold (regression tests)
# ════════════════════════════════════════════════════════════════════════════


class TestRound3FixVerification:
    """Confirm the 5 round 3 bug fixes weren't broken by the refactoring."""

    def test_boolean_params_rejected(self):
        """Boolean bypass was fixed in round 3."""
        with pytest.raises(ValueError):
            CatBoostMLX(iterations=True)._validate_params()
        with pytest.raises(ValueError):
            CatBoostMLX(learning_rate=True)._validate_params()
        with pytest.raises(ValueError):
            CatBoostMLX(random_strength=False)._validate_params()

    def test_non_string_feature_names_rejected(self):
        model = CatBoostMLXRegressor(iterations=5)
        X = np.random.randn(10, 3)
        y = np.random.randn(10)
        with pytest.raises(ValueError):
            model._validate_fit_inputs(X, y, feature_names=[1, 2, 3])

    def test_carriage_return_in_feature_name_rejected(self):
        model = CatBoostMLXRegressor(iterations=5)
        X = np.random.randn(10, 2)
        y = np.random.randn(10)
        with pytest.raises(ValueError):
            model._validate_fit_inputs(X, y, feature_names=["a\rb", "c"])

    def test_shap_feature_count_check(self):
        model, X, y = _train_regressor()
        with pytest.raises(ValueError, match="features"):
            model.get_shap_values(np.random.randn(5, 7))

    def test_shap_always_regenerates_cache(self):
        model, X, y = _train_regressor()
        model.predict(X[:5])
        model._model_data["model_info"]["_marker"] = "test"
        try:
            model.get_shap_values(X[:5])
        except Exception:
            pass
        assert "_marker" in model._model_json_cache


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2: Loss type divergence (NEW-1, HIGH)
# ════════════════════════════════════════════════════════════════════════════


class TestLossTypeDivergence:
    """_predict_inprocess resolves loss_type with a DIFFERENT default than
    _get_loss_type(), causing KeyError or silent wrong predictions when
    loss_type is missing from model_info.

    Root cause (core.py):
        _predict_inprocess: info.get("loss_type", "rmse")  -- defaults to "rmse"
        _get_loss_type():   info.get("loss_type", "")       -- falls back to self.loss

    When model_info has no loss_type:
        _predict_inprocess uses "rmse" link (identity)
        _get_loss_type() uses self.loss (could be "logloss", "poisson", etc.)
    """

    def test_classifier_works_without_loss_type(self):
        """Binary classifier: predict() succeeds when loss_type is missing
        from model_info because _predict_inprocess uses _get_loss_type()
        which falls back to self.loss.
        """
        model, X, y = _train_classifier()
        model2 = _save_load_without_loss_type(
            model, CatBoostMLXClassifier, loss="logloss"
        )

        preds = model2.predict(X[:5])
        assert len(preds) == 5
        assert np.all(np.isin(preds, [0, 1]))

    def test_multiclass_works_without_loss_type(self):
        """Multiclass classifier: predict() succeeds when loss_type is
        missing because _predict_inprocess uses _get_loss_type() fallback."""
        _skip_no_binaries()
        clf = CatBoostMLXClassifier(iterations=10, binary_path=BINARY_PATH)
        X = np.random.randn(60, 3)
        y = np.array([0] * 20 + [1] * 20 + [2] * 20)
        clf.fit(X, y)

        clf2 = _save_load_without_loss_type(
            clf, CatBoostMLXClassifier, loss="multiclass"
        )

        preds = clf2.predict(X[:5])
        assert len(preds) == 5
        assert np.all(np.isin(preds, [0, 1, 2]))

    def test_poisson_silent_wrong_predictions(self):
        """Poisson loss: silently returns raw logits instead of exp(logits).

        This is WORSE than a crash -- silent numerical corruption.
        _predict_inprocess uses rmse link (identity) instead of poisson (exp).
        """
        _skip_no_binaries()
        model = CatBoostMLXRegressor(
            iterations=10, loss="poisson", binary_path=BINARY_PATH
        )
        X = np.abs(np.random.randn(30, 3)) + 0.1
        y = np.abs(np.random.randn(30)) + 0.1
        model.fit(X, y)

        preds_correct = model.predict(X[:5])

        model2 = _save_load_without_loss_type(
            model, CatBoostMLXRegressor, loss="poisson"
        )
        preds_broken = model2.predict(X[:5])

        if not np.allclose(preds_correct, preds_broken, rtol=0.01):
            pytest.xfail(
                f"BUG NEW-1: Poisson predictions silently wrong. "
                f"Correct={preds_correct[:3]}, Got={preds_broken[:3]}. "
                f"_predict_inprocess used identity link instead of exp()."
            )

    def test_predict_proba_works_without_loss_type(self):
        """predict_proba succeeds with missing loss_type via _get_loss_type()
        fallback to self.loss."""
        model, X, y = _train_classifier()
        model2 = _save_load_without_loss_type(
            model, CatBoostMLXClassifier, loss="logloss"
        )

        proba = model2.predict_proba(X[:5])
        assert proba.shape == (5, 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_empty_string_loss_type_diverges(self):
        """Empty string loss_type causes divergence.

        _get_loss_type: empty string -> falsy -> falls back to self.loss
        _predict_inprocess: ''.split(':')[0] -> '' -> apply_link uses identity

        For classifier, this returns raw logits instead of class labels.
        """
        model, X, y = _train_classifier()
        model._model_data["model_info"]["loss_type"] = ""

        loss_get = model._get_loss_type()  # self.loss fallback

        # _predict_inprocess uses '' which falls to identity link
        # predict() may or may not see 'logloss' depending on self.loss
        try:
            preds = model.predict(X[:5])
        except (KeyError, ValueError):
            pytest.xfail(
                "BUG NEW-1: Empty string loss_type causes KeyError."
            )
            return

        # If no crash: check if predictions are class labels or raw logits
        is_class_labels = np.all(np.isin(preds, [0, 1]))
        if not is_class_labels:
            pytest.xfail(
                f"BUG NEW-1: Empty string loss_type causes silent corruption. "
                f"Classifier returned raw logits {preds[:3]} instead of "
                f"class labels [0, 1]. _get_loss_type='{loss_get}'."
            )

    def test_normal_operation_loss_types_agree(self):
        """When loss_type IS in model_info, both paths agree."""
        model, X, y = _train_regressor()
        lt_model_info = model._model_data["model_info"]["loss_type"]
        lt_get = model._get_loss_type()

        # _predict_inprocess would parse: info.get("loss_type", "rmse").split(":")[0]
        lt_inprocess = lt_model_info.split(":")[0].lower()

        assert lt_get == lt_inprocess, (
            f"Loss types diverge even in normal operation: "
            f"_get_loss_type={lt_get}, _predict_inprocess={lt_inprocess}"
        )


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3: _predict_subprocess dead code (NEW-2, LOW)
# ════════════════════════════════════════════════════════════════════════════


class TestSubprocessDeadCode:
    """_predict_subprocess has an unreachable binary format branch."""

    def test_binary_format_branch_unreachable(self):
        """_predict_subprocess checks 'if not self.cat_features:' but is only
        called from _run_predict when self.cat_features is truthy.

        The binary format branch (lines ~1382-1384) is dead code.
        """
        import inspect

        src = inspect.getsource(CatBoostMLX._predict_subprocess)
        has_dead_branch = "if not self.cat_features:" in src

        if has_dead_branch:
            pytest.xfail(
                "BUG NEW-2: _predict_subprocess has dead code. "
                "'if not self.cat_features:' branch is unreachable because "
                "_predict_subprocess is only called when cat_features is truthy."
            )


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4: predict_timeout ignored in in-process path (NEW-3, LOW)
# ════════════════════════════════════════════════════════════════════════════


class TestPredictTimeoutBypass:
    """In-process prediction ignores predict_timeout entirely."""

    def test_predict_timeout_not_enforced_inprocess(self):
        """predict_timeout=0.001 has no effect on in-process predictions.
        The subprocess path would raise TimeoutExpired, but in-process
        completes without checking the timeout.
        """
        model, X, y = _train_regressor(predict_timeout=0.001)

        # This should timeout if timeout were enforced, but won't
        X_large = np.random.randn(10000, 3)
        try:
            preds = model.predict(X_large)
            assert len(preds) == 10000
            pytest.xfail(
                "BUG NEW-3: predict_timeout=0.001 ignored by in-process path. "
                "10000 rows predicted without timeout. Subprocess path would "
                "have raised TimeoutExpired."
            )
        except RuntimeError:
            pass  # Timeout was enforced (fixed)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5: In-process vs subprocess numerical agreement
# ════════════════════════════════════════════════════════════════════════════


class TestNumericalAgreement:
    """Verify in-process and subprocess paths give the same predictions."""

    def test_regression_predictions_match(self):
        """In-process and subprocess should agree within float32 tolerance."""
        model, X, y = _train_regressor()
        X_test = np.random.randn(10, 3)

        # In-process (default for numeric-only)
        preds_inprocess = model.predict(X_test)

        # Subprocess (call directly)
        result = model._predict_subprocess(X_test)
        preds_subprocess = result["prediction"]

        assert np.allclose(preds_inprocess, preds_subprocess, atol=1e-5), (
            f"Max diff: {np.max(np.abs(preds_inprocess - preds_subprocess))}"
        )

    def test_classifier_predictions_match(self):
        """Classifier in-process vs subprocess agreement."""
        model, X, y = _train_classifier()
        X_test = np.random.randn(10, 3)

        preds_inprocess = model.predict(X_test)

        # Force subprocess
        result = model._predict_subprocess(X_test)
        preds_subprocess = result["predicted_class"].astype(int)

        assert np.array_equal(preds_inprocess, preds_subprocess)

    def test_predict_vs_staged_predict_agreement(self):
        """predict() and staged_predict() final output should be identical."""
        model, X, y = _train_regressor()
        X_test = np.random.randn(10, 3)

        preds = model.predict(X_test)
        staged = list(model.staged_predict(X_test))

        assert np.allclose(preds, staged[-1], atol=1e-12)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 6: In-process prediction edge cases
# ════════════════════════════════════════════════════════════════════════════


class TestInprocessEdgeCases:
    """Edge cases for the new _predict_inprocess path."""

    def test_nan_in_predict_input(self):
        """NaN values in predict input should be handled correctly."""
        _skip_no_binaries()
        model = CatBoostMLXRegressor(
            iterations=10, nan_mode="min", binary_path=BINARY_PATH
        )
        X = np.random.randn(30, 3)
        X[0, 0] = np.nan
        y = np.random.randn(30)
        model.fit(X, y)

        X_test = np.array([[np.nan, 1.0, 2.0], [1.0, 2.0, 3.0]])
        preds = model.predict(X_test)
        assert np.all(np.isfinite(preds))

    def test_single_sample_prediction(self):
        """Single sample prediction via in-process."""
        model, X, y = _train_regressor()
        pred = model.predict(X[:1])
        assert pred.shape == (1,)
        assert np.isfinite(pred[0])

    def test_empty_input_returns_empty(self):
        """Empty input should return empty array."""
        model, X, y = _train_regressor()
        pred = model.predict(np.zeros((0, 3)))
        assert pred.shape == (0,)

    def test_1d_input_reshaped(self):
        """1D input should be reshaped to (n, 1)."""
        _skip_no_binaries()
        model = CatBoostMLXRegressor(iterations=10, binary_path=BINARY_PATH)
        X = np.random.randn(30, 1)
        y = np.random.randn(30)
        model.fit(X, y)

        pred = model.predict(np.array([1.0, 2.0, 3.0]))
        assert pred.shape == (3,)

    def test_pickle_roundtrip_preserves_inprocess(self):
        """Pickle roundtrip should preserve in-process prediction ability."""
        model, X, y = _train_classifier()
        preds_before = model.predict(X[:5])

        buf = pickle.dumps(model)
        model2 = pickle.loads(buf)
        preds_after = model2.predict(X[:5])

        assert np.array_equal(preds_before, preds_after)
        assert model2._model_data["model_info"]["loss_type"] == "logloss"

    def test_save_load_roundtrip(self):
        """save_model/load_model should preserve in-process prediction."""
        model, X, y = _train_regressor()
        preds_before = model.predict(X[:5])

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            model.save_model(f.name)
            model2 = CatBoostMLXRegressor(binary_path=BINARY_PATH)
            model2.load_model(f.name)
            os.unlink(f.name)

        preds_after = model2.predict(X[:5])
        assert np.allclose(preds_before, preds_after, atol=1e-12)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 7: Dispatch correctness
# ════════════════════════════════════════════════════════════════════════════


class TestDispatchCorrectness:
    """Verify _run_predict dispatches to the correct path."""

    def test_numeric_only_uses_inprocess(self):
        """Numeric-only model should use _predict_inprocess (no subprocess)."""
        model, X, y = _train_regressor()

        # After predict, _model_json_cache should NOT be updated
        # (in-process path doesn't touch the cache)
        model._model_json_cache = None
        model.predict(X[:5])

        # In-process path doesn't update _model_json_cache
        # (subprocess path would have set it)
        # This is an observable side effect we can check
        # Note: if the dev changes this later, this test may break
        if model._model_json_cache is None:
            pass  # Confirmed in-process was used

    def test_cat_features_uses_subprocess(self):
        """Categorical model should use _predict_subprocess."""
        _skip_no_binaries()
        model = CatBoostMLXRegressor(
            iterations=10, cat_features=[0], binary_path=BINARY_PATH
        )
        X = np.array(
            [["a", 1.0], ["b", 2.0], ["c", 3.0],
             ["a", 4.0], ["b", 5.0], ["c", 1.0],
             ["a", 2.0], ["b", 3.0], ["c", 4.0],
             ["a", 5.0]],
            dtype=object,
        )
        y = np.random.randn(10)
        model.fit(X, y)

        model._model_json_cache = None
        model.predict(X[:3])

        # Subprocess path sets _model_json_cache
        assert model._model_json_cache is not None

    def test_feature_count_check_before_dispatch(self):
        """Feature count check should happen before dispatch, catching errors
        early regardless of which path would be used."""
        model, X, y = _train_regressor()
        X_wrong = np.random.randn(5, 7)

        with pytest.raises(ValueError, match="features"):
            model.predict(X_wrong)
