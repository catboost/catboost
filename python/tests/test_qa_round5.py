"""
test_qa_round5.py -- QA Round 5: Testing round 4 fixes + load_model gaps.

The dev fixed the loss_type divergence in _predict_inprocess and removed
dead code from _predict_subprocess. QA verified those fixes and found new
issues in load_model, particularly around restoring categorical features
and classifier state.

Focus areas:
1. Verify round 4 fixes (loss_type, dead code)
2. load_model doesn't restore cat_features -> SHAP crash, wrong dispatch
3. load_model doesn't restore classes_ for classifier
4. cross_validate doesn't validate n_folds
"""

import json
import os
import pickle
import tempfile

import numpy as np
import pytest

from catboost_mlx import CatBoostMLXClassifier, CatBoostMLXRegressor

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BINARY_PATH = REPO_ROOT


def _skip_no_binaries():
    csv_train = os.path.join(BINARY_PATH, "csv_train")
    csv_predict = os.path.join(BINARY_PATH, "csv_predict")
    if not (os.path.isfile(csv_train) and os.path.isfile(csv_predict)):
        pytest.skip("Compiled binaries not found")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1: Verify round 4 fixes
# ════════════════════════════════════════════════════════════════════════════


class TestRound4FixVerification:
    """Confirm the 2 round 4 fixes are still good."""

    def test_predict_inprocess_uses_get_loss_type(self):
        """_predict_inprocess should use self._get_loss_type() not
        info.get('loss_type', 'rmse')."""
        import inspect

        from catboost_mlx.core import CatBoostMLX

        src = inspect.getsource(CatBoostMLX._predict_inprocess)
        assert "self._get_loss_type()" in src
        assert 'info.get("loss_type"' not in src

    def test_subprocess_no_dead_code(self):
        """_predict_subprocess should not have the unreachable binary branch."""
        import inspect

        from catboost_mlx.core import CatBoostMLX

        src = inspect.getsource(CatBoostMLX._predict_subprocess)
        assert "if not self.cat_features:" not in src

    def test_classifier_predict_without_loss_type(self):
        """Classifier predict works after stripping loss_type (round 4 fix)."""
        _skip_no_binaries()
        clf = CatBoostMLXClassifier(iterations=10, binary_path=BINARY_PATH)
        X = np.random.randn(30, 3)
        y = np.array([0] * 15 + [1] * 15)
        clf.fit(X, y)

        # Strip loss_type, reload
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            data = json.loads(json.dumps(clf._model_data))
            data["model_info"].pop("loss_type", None)
            json.dump(data, f)
            path = f.name

        try:
            clf2 = CatBoostMLXClassifier(loss="logloss", binary_path=BINARY_PATH)
            clf2.load_model(path)
            preds = clf2.predict(X[:5])
            assert len(preds) == 5
            assert np.all(np.isin(preds, [0, 1]))
        finally:
            os.unlink(path)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2: load_model doesn't restore cat_features (NEW-1, HIGH)
# ════════════════════════════════════════════════════════════════════════════


class TestLoadModelCatFeatures:
    """load_model doesn't restore cat_features from the model JSON even though
    is_categorical=True is present in the features list. This causes:
    - Wrong prediction dispatch (in-process instead of subprocess)
    - SHAP crash on categorical data (tries binary format with strings)
    - Prediction inconsistency (works by accident via feature metadata)

    The fix is trivial -- reconstruct cat_features from features[i].is_categorical:
        self.cat_features = [i for i, f in enumerate(features) if f.get('is_categorical')]
    """

    def _train_cat_model(self):
        _skip_no_binaries()
        model = CatBoostMLXRegressor(
            iterations=10, cat_features=[0], binary_path=BINARY_PATH
        )
        X = np.array(
            [["a", 1.0], ["b", 2.0], ["c", 3.0], ["a", 4.0], ["b", 5.0],
             ["c", 1.0], ["a", 2.0], ["b", 3.0], ["c", 4.0], ["a", 5.0]],
            dtype=object,
        )
        y = np.random.randn(10)
        model.fit(X, y)
        return model, X, y

    def _save_and_load(self, model):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            model.save_model(f.name)
            path = f.name
        model2 = CatBoostMLXRegressor(binary_path=BINARY_PATH)
        model2.load_model(path)
        os.unlink(path)
        return model2

    def test_cat_features_lost_after_load(self):
        """cat_features becomes None after save_model + load_model.

        BUG: load_model (line ~1446) doesn't reconstruct cat_features from
        the is_categorical flags in the features list.
        """
        model, X, y = self._train_cat_model()
        assert model.cat_features == [0]

        model2 = self._save_and_load(model)

        if model2.cat_features is None:
            pytest.xfail(
                "BUG NEW-1: load_model doesn't restore cat_features. "
                "is_categorical=True in features[0] but cat_features stays None. "
                "Fix: self.cat_features = [i for i, f in enumerate(features) "
                "if f.get('is_categorical')]"
            )
        else:
            assert model2.cat_features == [0]

    def test_shap_crashes_after_load_with_cat(self):
        """SHAP crashes after load_model because cat_features=None causes
        binary format to be used, which can't handle string data.
        """
        model, X, y = self._train_cat_model()

        # SHAP works before save/load
        shap1 = model.get_shap_values(X[:3])
        assert "shap_values" in shap1

        model2 = self._save_and_load(model)

        try:
            shap2 = model2.get_shap_values(X[:3])
            assert "shap_values" in shap2
        except (ValueError, TypeError) as e:
            if "could not convert string to float" in str(e):
                pytest.xfail(
                    "BUG NEW-1: SHAP crashes after load_model with categorical data. "
                    "cat_features=None -> binary format -> can't convert 'a' to float. "
                    "Root cause: load_model doesn't restore cat_features."
                )
            raise

    def test_predict_dispatch_wrong_after_load(self):
        """After load_model, predict routes to _predict_inprocess instead of
        _predict_subprocess because cat_features is None."""
        model, X, y = self._train_cat_model()
        model2 = self._save_and_load(model)

        # Check dispatch: in-process doesn't update _model_json_cache
        model2._model_json_cache = "sentinel"
        model2.predict(X[:3])

        if model2._model_json_cache == "sentinel":
            pytest.xfail(
                "BUG NEW-1: After load_model, predict uses in-process path "
                "(cat_features=None) instead of subprocess path. "
                "Predictions work by accident via feature metadata."
            )

    def test_pickle_preserves_cat_features(self):
        """Pickle correctly preserves cat_features (unlike load_model)."""
        model, X, y = self._train_cat_model()

        buf = pickle.dumps(model)
        model2 = pickle.loads(buf)

        assert model2.cat_features == [0]

    def test_is_categorical_available_in_model_json(self):
        """The model JSON has is_categorical flags that load_model could use."""
        model, X, y = self._train_cat_model()

        features = model._model_data["features"]
        cat_from_metadata = [
            i for i, f in enumerate(features) if f.get("is_categorical", False)
        ]
        assert cat_from_metadata == [0], (
            "is_categorical flag not present in model JSON features"
        )


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3: load_model doesn't restore classes_ for classifier (NEW-2, MEDIUM)
# ════════════════════════════════════════════════════════════════════════════


class TestLoadModelClassifierState:
    """CatBoostMLXClassifier sets classes_ during fit(). This attribute is
    not restored by load_model, making the loaded model incompatible with
    some sklearn utilities that check for classes_.
    """

    def test_classes_lost_after_load(self):
        """classes_ attribute missing after load_model."""
        _skip_no_binaries()
        clf = CatBoostMLXClassifier(iterations=10, binary_path=BINARY_PATH)
        X = np.random.randn(30, 3)
        y = np.array([0] * 15 + [1] * 15)
        clf.fit(X, y)

        assert hasattr(clf, "classes_")
        assert list(clf.classes_) == [0, 1]

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            clf.save_model(f.name)
            path = f.name

        try:
            clf2 = CatBoostMLXClassifier.load(path, binary_path=BINARY_PATH)
            if not hasattr(clf2, "classes_"):
                pytest.xfail(
                    "BUG NEW-2: classes_ not restored by load_model. "
                    "CatBoostMLXClassifier sets classes_ in fit() but "
                    "load_model doesn't reconstruct it from model_info."
                )
            assert list(clf2.classes_) == [0, 1]
        finally:
            os.unlink(path)

    def test_multiclass_classes_lost_after_load(self):
        """Multiclass classes_ also lost after load."""
        _skip_no_binaries()
        clf = CatBoostMLXClassifier(iterations=10, binary_path=BINARY_PATH)
        X = np.random.randn(60, 3)
        y = np.array([0] * 20 + [1] * 20 + [2] * 20)
        clf.fit(X, y)

        assert list(clf.classes_) == [0, 1, 2]

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            clf.save_model(f.name)
            path = f.name

        try:
            clf2 = CatBoostMLXClassifier.load(path, binary_path=BINARY_PATH)
            if not hasattr(clf2, "classes_"):
                pytest.xfail(
                    "BUG NEW-2: Multiclass classes_ not restored. "
                    "num_classes is in model_info but classes_ is not reconstructed."
                )
        finally:
            os.unlink(path)

    def test_pickle_preserves_classes(self):
        """Pickle correctly preserves classes_ (unlike load_model)."""
        _skip_no_binaries()
        clf = CatBoostMLXClassifier(iterations=10, binary_path=BINARY_PATH)
        X = np.random.randn(30, 3)
        y = np.array([0] * 15 + [1] * 15)
        clf.fit(X, y)

        buf = pickle.dumps(clf)
        clf2 = pickle.loads(buf)
        assert hasattr(clf2, "classes_")
        assert list(clf2.classes_) == [0, 1]

    def test_predict_proba_works_without_classes(self):
        """predict_proba still works without classes_ (it doesn't use it).
        But some downstream sklearn tools may break."""
        _skip_no_binaries()
        clf = CatBoostMLXClassifier(iterations=10, binary_path=BINARY_PATH)
        X = np.random.randn(30, 3)
        y = np.array([0] * 15 + [1] * 15)
        clf.fit(X, y)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            clf.save_model(f.name)
            path = f.name

        try:
            clf2 = CatBoostMLXClassifier.load(path, binary_path=BINARY_PATH)
            probs = clf2.predict_proba(X[:5])
            assert probs.shape == (5, 2)
            assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
        finally:
            os.unlink(path)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4: cross_validate doesn't validate n_folds (NEW-3, LOW)
# ════════════════════════════════════════════════════════════════════════════


class TestCrossValidateValidation:
    """cross_validate passes n_folds directly to the C++ binary without
    validation. Invalid values cause confusing errors or hangs."""

    def test_n_folds_zero_no_validation(self):
        """n_folds=0 should be rejected with ValueError, not cause a
        confusing timeout or C++ error."""
        _skip_no_binaries()
        model = CatBoostMLXRegressor(
            iterations=3, train_timeout=5, binary_path=BINARY_PATH
        )
        X = np.random.randn(20, 2)
        y = np.random.randn(20)

        try:
            model.cross_validate(X, y, n_folds=0)
            pytest.xfail(
                "BUG NEW-3: n_folds=0 accepted without validation."
            )
        except ValueError:
            pass  # Fixed: proper validation
        except RuntimeError:
            pytest.xfail(
                "BUG NEW-3: n_folds=0 causes RuntimeError (timeout or C++ error) "
                "instead of clear ValueError."
            )

    def test_n_folds_negative_no_validation(self):
        """n_folds=-1 should be rejected."""
        _skip_no_binaries()
        model = CatBoostMLXRegressor(
            iterations=3, train_timeout=5, binary_path=BINARY_PATH
        )
        X = np.random.randn(20, 2)
        y = np.random.randn(20)

        try:
            model.cross_validate(X, y, n_folds=-1)
            pytest.xfail("BUG NEW-3: n_folds=-1 accepted.")
        except ValueError:
            pass
        except RuntimeError:
            pytest.xfail(
                "BUG NEW-3: n_folds=-1 causes C++ error, not ValueError."
            )


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5: load_model robustness
# ════════════════════════════════════════════════════════════════════════════


class TestLoadModelRobustness:
    """Additional load_model edge cases."""

    def test_load_model_validates_required_keys(self):
        """Model JSON missing required keys should raise ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump({"model_info": {}, "features": []}, f)
            path = f.name

        try:
            model = CatBoostMLXRegressor()
            with pytest.raises(ValueError, match="missing required keys"):
                model.load_model(path)
        finally:
            os.unlink(path)

    def test_load_model_syncs_loss_param(self):
        """load_model should sync self.loss from model_info.loss_type."""
        _skip_no_binaries()
        model = CatBoostMLXRegressor(
            iterations=10, loss="mae", binary_path=BINARY_PATH
        )
        X = np.random.randn(30, 3)
        y = np.random.randn(30)
        model.fit(X, y)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            model.save_model(f.name)
            path = f.name

        try:
            model2 = CatBoostMLXRegressor()
            assert model2.loss == "rmse"  # default
            model2.load_model(path)
            assert model2.loss == "mae"  # synced from model JSON
        finally:
            os.unlink(path)

    def test_load_model_sets_n_features_from_features_list(self):
        """n_features_in_ should match the features list."""
        _skip_no_binaries()
        model = CatBoostMLXRegressor(
            iterations=10, binary_path=BINARY_PATH
        )
        X = np.random.randn(30, 5)
        y = np.random.randn(30)
        model.fit(X, y)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            model.save_model(f.name)
            path = f.name

        try:
            model2 = CatBoostMLXRegressor()
            model2.load_model(path)
            assert model2.n_features_in_ == 5
        finally:
            os.unlink(path)

    def test_save_load_predict_roundtrip_regression(self):
        """Full roundtrip: train -> save -> load -> predict."""
        _skip_no_binaries()
        model = CatBoostMLXRegressor(
            iterations=10, binary_path=BINARY_PATH
        )
        np.random.seed(42)
        X = np.random.randn(30, 3)
        y = np.random.randn(30)
        model.fit(X, y)

        preds_before = model.predict(X[:5])

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            model.save_model(f.name)
            path = f.name

        try:
            model2 = CatBoostMLXRegressor(binary_path=BINARY_PATH)
            model2.load_model(path)
            preds_after = model2.predict(X[:5])
            assert np.allclose(preds_before, preds_after, atol=1e-10)
        finally:
            os.unlink(path)
