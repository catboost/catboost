"""
test_qa_round7.py -- QA Round 7: Testing round 6 fixes + multiclass classes_ bugs.

The dev fixed stale cat_features, case-sensitive logloss check, and
approx_dimension fallback for classes_. QA verified those fixes and found
a critical off-by-one in the approx_dimension fallback (CatBoost uses
approx_dimension = num_classes - 1, not num_classes).

Focus areas:
1. Verify round 6 fixes (stale cat_features, logloss case, approx_dim)
2. classes_ off-by-one with approx_dimension fallback (HIGH)
3. predict_proba / classes_ shape consistency
4. Multiclass end-to-end after load_model
"""

import json
import os
import tempfile

import numpy as np

from catboost_mlx import CatBoostMLXClassifier, CatBoostMLXRegressor

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _make_model_json(features, trees, model_info=None, tmpdir=None):
    """Helper: write a minimal model JSON and return its path."""
    if model_info is None:
        model_info = {"loss_type": "RMSE"}
    data = {"model_info": model_info, "features": features, "trees": trees}
    if tmpdir is None:
        tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "model.json")
    with open(path, "w") as f:
        json.dump(data, f)
    return path


def _simple_features(n=2):
    """Helper: create n numeric features."""
    return [{"index": i, "name": f"f{i}", "borders": [0.5]} for i in range(n)]


def _multiclass_trees(approx_dim, depth=1):
    """Helper: single tree with leaf_values for multiclass (K-1 convention)."""
    n_leaves = 2 ** depth
    n_values = n_leaves * approx_dim
    return [{
        "depth": depth,
        "splits": [{"feature_idx": 0, "bin_threshold": i} for i in range(depth)],
        "leaf_values": list(np.linspace(-0.5, 0.5, n_values)),
    }]


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1: Verify round 6 fixes
# ════════════════════════════════════════════════════════════════════════════


class TestRound6FixVerification:
    """Confirm that round 6 bugs are actually fixed."""

    def test_stale_cat_features_cleared(self):
        """Round 6 NEW-1: Stale cat_features cleared on load of numeric model."""
        features = [
            {"index": 0, "name": "f0", "borders": [0.5]},
            {"index": 1, "name": "f1", "borders": [0.5]},
        ]
        path = _make_model_json(features=features, trees=[{
            "depth": 1,
            "splits": [{"feature_idx": 0, "bin_threshold": 0}],
            "leaf_values": [0.1, -0.1],
        }])
        m = CatBoostMLXRegressor(cat_features=[0])
        m.load_model(path)
        assert m.cat_features is None

    def test_capital_logloss_sets_classes(self):
        """Round 6 NEW-2: 'Logloss' (capital L) correctly sets classes_."""
        path = _make_model_json(
            features=_simple_features(2),
            trees=[{
                "depth": 1,
                "splits": [{"feature_idx": 0, "bin_threshold": 0}],
                "leaf_values": [0.5, -0.5],
            }],
            model_info={"loss_type": "Logloss", "num_classes": 0},
        )
        clf = CatBoostMLXClassifier()
        clf.load_model(path)
        assert hasattr(clf, "classes_")
        assert list(clf.classes_) == [0, 1]

    def test_approx_dim_fallback_sets_classes(self):
        """Round 6 NEW-3: approx_dimension fallback sets classes_."""
        path = _make_model_json(
            features=_simple_features(2),
            trees=_multiclass_trees(approx_dim=2),
            model_info={"loss_type": "MultiClass", "num_classes": 0,
                        "approx_dimension": 2},
        )
        clf = CatBoostMLXClassifier()
        clf.load_model(path)
        assert hasattr(clf, "classes_")

    def test_cat_features_persistence_roundtrip(self):
        """cat_features persisted in model_info survives save→load."""
        features = [
            {"index": 0, "name": "f0", "borders": [0.5]},
            {"index": 1, "name": "f1", "is_categorical": True, "borders": [],
             "cat_hash_map": {"a": 0}},
        ]
        path = _make_model_json(
            features=features,
            trees=[{
                "depth": 1,
                "splits": [{"feature_idx": 0, "bin_threshold": 0}],
                "leaf_values": [0.1, -0.1],
            }],
            model_info={"loss_type": "RMSE", "cat_features": [1]},
        )
        m = CatBoostMLXRegressor()
        m.load_model(path)
        assert m.cat_features == [1]

        # Save and reload
        tmpdir = tempfile.mkdtemp()
        save_path = os.path.join(tmpdir, "resaved.json")
        m.save_model(save_path)

        m2 = CatBoostMLXRegressor()
        m2.load_model(save_path)
        assert m2.cat_features == [1]


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2: NEW-1 — classes_ off-by-one with approx_dimension (HIGH)
# ════════════════════════════════════════════════════════════════════════════


class TestClassesOffByOne:
    """CatBoost uses approx_dimension = num_classes - 1 for multiclass.
    The classifier's load_model fallback does:
        self.classes_ = np.arange(approx_dim, dtype=int)
    This gives K-1 classes instead of K.

    Example: 3-class model with approx_dim=2 → classes_=[0,1] instead of [0,1,2]
    This causes classes_/predict_proba shape mismatch."""

    def test_3class_approx_dim_2(self):
        """3-class model: approx_dim=2, classes_ should be [0,1,2] not [0,1]."""
        path = _make_model_json(
            features=_simple_features(2),
            trees=_multiclass_trees(approx_dim=2),
            model_info={"loss_type": "MultiClass", "num_classes": 0,
                        "approx_dimension": 2},
        )
        clf = CatBoostMLXClassifier()
        clf.load_model(path)
        # approx_dim=2 → K-1=2 → K=3 classes
        assert len(clf.classes_) == 3, (
            f"Expected 3 classes (approx_dim+1), got {len(clf.classes_)}: {clf.classes_}"
        )
        assert list(clf.classes_) == [0, 1, 2]

    def test_5class_approx_dim_4(self):
        """5-class model: approx_dim=4, classes_ should be [0,1,2,3,4]."""
        path = _make_model_json(
            features=_simple_features(2),
            trees=_multiclass_trees(approx_dim=4),
            model_info={"loss_type": "MultiClass", "num_classes": 0,
                        "approx_dimension": 4},
        )
        clf = CatBoostMLXClassifier()
        clf.load_model(path)
        assert len(clf.classes_) == 5, (
            f"Expected 5 classes (approx_dim+1), got {len(clf.classes_)}: {clf.classes_}"
        )

    def test_classes_predict_proba_shape_match(self):
        """len(classes_) must equal predict_proba.shape[1]."""
        path = _make_model_json(
            features=_simple_features(2),
            trees=_multiclass_trees(approx_dim=2),
            model_info={"loss_type": "MultiClass", "num_classes": 0,
                        "approx_dimension": 2},
        )
        clf = CatBoostMLXClassifier()
        clf.load_model(path)
        X = np.array([[0.3, 0.5], [0.7, -0.5]])
        proba = clf.predict_proba(X)
        assert len(clf.classes_) == proba.shape[1], (
            f"classes_ has {len(clf.classes_)} entries but predict_proba "
            f"has {proba.shape[1]} columns"
        )

    def test_num_classes_path_correct(self):
        """When num_classes is provided, classes_ is correct (control test)."""
        path = _make_model_json(
            features=_simple_features(2),
            trees=_multiclass_trees(approx_dim=2),
            model_info={"loss_type": "MultiClass", "num_classes": 3,
                        "approx_dimension": 2},
        )
        clf = CatBoostMLXClassifier()
        clf.load_model(path)
        assert list(clf.classes_) == [0, 1, 2]
        X = np.array([[0.3, 0.5], [0.7, -0.5]])
        proba = clf.predict_proba(X)
        assert len(clf.classes_) == proba.shape[1]


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3: Multiclass end-to-end after load_model
# ════════════════════════════════════════════════════════════════════════════


class TestMulticlassEndToEnd:
    """Full multiclass pipeline: load → predict → predict_proba → staged."""

    def test_4class_full_pipeline(self):
        """4-class model with num_classes=4: complete prediction pipeline."""
        path = _make_model_json(
            features=_simple_features(2),
            trees=_multiclass_trees(approx_dim=3),  # K-1=3 → K=4
            model_info={"loss_type": "MultiClass", "num_classes": 4,
                        "approx_dimension": 3},
        )
        clf = CatBoostMLXClassifier()
        clf.load_model(path)

        X = np.array([[0.3, 0.5], [0.7, -0.5], [1.5, 0.0]])

        # predict
        pred = clf.predict(X)
        assert pred.shape == (3,)
        assert all(p in range(4) for p in pred)

        # predict_proba
        proba = clf.predict_proba(X)
        assert proba.shape == (3, 4)
        assert np.allclose(proba.sum(axis=1), 1.0)

        # staged_predict
        staged = list(clf.staged_predict(X))
        assert np.array_equal(pred, staged[-1])

        # staged_predict_proba
        staged_proba = list(clf.staged_predict_proba(X))
        assert np.allclose(proba, staged_proba[-1])

    def test_binary_logloss_full_pipeline(self):
        """Binary model with Logloss: complete prediction pipeline."""
        path = _make_model_json(
            features=_simple_features(2),
            trees=[{
                "depth": 1,
                "splits": [{"feature_idx": 0, "bin_threshold": 0}],
                "leaf_values": [2.0, -2.0],
            }],
            model_info={"loss_type": "Logloss", "num_classes": 0},
        )
        clf = CatBoostMLXClassifier()
        clf.load_model(path)

        assert list(clf.classes_) == [0, 1]

        X = np.array([[0.3, 0.5], [0.7, -0.5]])
        pred = clf.predict(X)
        proba = clf.predict_proba(X)

        assert pred.shape == (2,)
        assert proba.shape == (2, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_predict_proba_sums_to_one(self):
        """Probability rows must sum to 1.0 for multiclass configs (K >= 3)."""
        for n_classes in [3, 5, 10]:
            approx_dim = n_classes - 1
            path = _make_model_json(
                features=_simple_features(2),
                trees=_multiclass_trees(approx_dim=approx_dim),
                model_info={"loss_type": "MultiClass",
                            "num_classes": n_classes,
                            "approx_dimension": approx_dim},
            )
            clf = CatBoostMLXClassifier()
            clf.load_model(path)
            X = np.random.randn(5, 2)
            proba = clf.predict_proba(X)
            assert proba.shape == (5, n_classes), (
                f"n_classes={n_classes}: expected shape (5, {n_classes}), "
                f"got {proba.shape}"
            )
            assert np.allclose(proba.sum(axis=1), 1.0)

    def test_2class_multiclass_predict_proba(self):
        """2-class multiclass (approx_dim=1) crashes in apply_link.

        cursor is 1D (shape (n_samples,)) but apply_link does
        cursor.max(axis=1) which fails for 1D arrays."""
        path = _make_model_json(
            features=_simple_features(2),
            trees=_multiclass_trees(approx_dim=1),
            model_info={"loss_type": "MultiClass",
                        "num_classes": 2,
                        "approx_dimension": 1},
        )
        clf = CatBoostMLXClassifier()
        clf.load_model(path)
        X = np.random.randn(3, 2)
        proba = clf.predict_proba(X)
        assert proba.shape == (3, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4: Regression tests — things that still work
# ════════════════════════════════════════════════════════════════════════════


class TestRound7Regressions:
    """Ensure round 6 fixes didn't break existing functionality."""

    def test_predict_with_zero_trees(self):
        """Model with 0 trees returns zeros (or base prediction)."""
        path = _make_model_json(
            features=_simple_features(2),
            trees=[],
            model_info={"loss_type": "RMSE"},
        )
        m = CatBoostMLXRegressor()
        m.load_model(path)
        pred = m.predict(np.array([[0.3, 0.5]]))
        assert np.allclose(pred, 0.0)

    def test_predict_with_base_prediction(self):
        """Model with base_prediction but 0 trees returns base prediction."""
        path = _make_model_json(
            features=_simple_features(2),
            trees=[],
            model_info={"loss_type": "RMSE", "base_prediction": [3.5]},
        )
        m = CatBoostMLXRegressor()
        m.load_model(path)
        pred = m.predict(np.array([[0.3, 0.5]]))
        assert np.allclose(pred, 3.5)

    def test_predict_staged_consistency(self):
        """predict() and staged_predict() final output must match."""
        path = _make_model_json(
            features=_simple_features(2),
            trees=[{
                "depth": 1,
                "splits": [{"feature_idx": 0, "bin_threshold": 0}],
                "leaf_values": [0.3, -0.3],
            }, {
                "depth": 1,
                "splits": [{"feature_idx": 1, "bin_threshold": 0}],
                "leaf_values": [0.1, -0.1],
            }],
            model_info={"loss_type": "RMSE"},
        )
        m = CatBoostMLXRegressor()
        m.load_model(path)
        X = np.array([[0.3, 0.5], [0.7, -0.5], [1.5, 0.0]])
        pred = m.predict(X)
        staged = list(m.staged_predict(X))
        assert np.allclose(pred, staged[-1])

    def test_feature_importances_after_load(self):
        """feature_importances_ works after load_model."""
        path = _make_model_json(
            features=[
                {"index": 0, "name": "age", "borders": [25.0]},
                {"index": 1, "name": "income", "borders": [30000]},
            ],
            trees=[{
                "depth": 1,
                "splits": [{"feature_idx": 0, "bin_threshold": 0}],
                "leaf_values": [0.1, -0.1],
            }],
            model_info={"loss_type": "RMSE"},
        )
        m = CatBoostMLXRegressor()
        m.load_model(path)
        fi = m.feature_importances_
        assert fi.shape == (2,)
        assert np.isclose(fi.sum(), 0.0) or np.isclose(fi.sum(), 1.0)

    def test_nan_in_predict(self):
        """NaN values in input handled without crash."""
        path = _make_model_json(
            features=[
                {"index": 0, "name": "f0", "borders": [0.5],
                 "has_nan": True, "nan_goes_right": True},
            ],
            trees=[{
                "depth": 1,
                "splits": [{"feature_idx": 0, "bin_threshold": 0}],
                "leaf_values": [0.3, -0.3],
            }],
            model_info={"loss_type": "RMSE"},
        )
        m = CatBoostMLXRegressor()
        m.load_model(path)
        X = np.array([[np.nan], [0.3], [np.nan]])
        pred = m.predict(X)
        assert pred.shape == (3,)
        assert not np.any(np.isnan(pred))
