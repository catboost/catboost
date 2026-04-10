"""
test_qa_round6.py -- QA Round 6: Testing round 5 fixes + load_model state bugs.

The dev fixed load_model to restore cat_features from is_categorical flags,
added classes_ restoration in classifier, and added n_folds validation in
cross_validate. QA verified those fixes and found new bugs in the same
load_model code paths.

Focus areas:
1. Verify round 5 fixes (cat_features, classes_, n_folds)
2. Stale cat_features NOT cleared when loading numeric-only model (HIGH)
3. Case-sensitive logloss check breaks binary classifier classes_ (HIGH)
4. classes_ not set when num_classes=0 for multiclass (HIGH)
"""

import json
import os
import tempfile

import numpy as np
import pytest

from catboost_mlx import CatBoostMLXClassifier, CatBoostMLXRegressor

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BINARY_PATH = REPO_ROOT


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


def _simple_features(n=2, categoricals=None):
    """Helper: create feature list with optional categoricals."""
    categoricals = categoricals or []
    feats = []
    for i in range(n):
        f = {"index": i, "name": f"f{i}", "borders": [0.5]}
        if i in categoricals:
            f["is_categorical"] = True
            f["borders"] = []
        feats.append(f)
    return feats


def _simple_trees():
    """Helper: single tree with one split."""
    return [{"splits": [{"feature_index": 0, "border": 0.5}],
             "leaf_values": [0.1, -0.1]}]


def _multiclass_trees(n_classes=3):
    """Helper: single tree with leaf_values for multiclass."""
    return [{"splits": [{"feature_index": 0, "border": 0.5}],
             "leaf_values": list(np.linspace(-0.1, 0.1, 2 * n_classes))}]


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1: Verify round 5 fixes
# ════════════════════════════════════════════════════════════════════════════


class TestRound5FixVerification:
    """Confirm that round 5 bugs are actually fixed."""

    def test_load_model_restores_cat_features(self):
        """Round 5 NEW-1: load_model should restore cat_features from JSON."""
        path = _make_model_json(
            features=_simple_features(3, categoricals=[1]),
            trees=_simple_trees(),
        )
        m = CatBoostMLXRegressor()
        assert m.cat_features is None
        m.load_model(path)
        assert m.cat_features == [1], f"Expected [1], got {m.cat_features}"

    def test_load_model_restores_classes_multiclass(self):
        """Round 5 NEW-2: classifier load_model should set classes_."""
        path = _make_model_json(
            features=_simple_features(2),
            trees=_multiclass_trees(3),
            model_info={"loss_type": "MultiClass", "num_classes": 3,
                        "approx_dimension": 3},
        )
        clf = CatBoostMLXClassifier()
        clf.load_model(path)
        assert hasattr(clf, "classes_"), "classes_ not set after load"
        assert list(clf.classes_) == [0, 1, 2]

    def test_n_folds_validation_rejects_bad_values(self):
        """Round 5 NEW-3: cross_validate rejects invalid n_folds."""
        m = CatBoostMLXRegressor()
        X = np.random.randn(10, 2)
        y = np.random.randn(10)
        for bad in [0, 1, -1, True, False, 2.5, "3", None]:
            with pytest.raises(ValueError, match="n_folds must be an integer"):
                m.cross_validate(X, y, n_folds=bad)

    def test_n_folds_exceeds_samples(self):
        """Round 5 NEW-3: n_folds > n_samples rejected."""
        m = CatBoostMLXRegressor()
        X = np.array([[1], [2]])
        y = np.array([0, 1])
        with pytest.raises(ValueError, match="cannot exceed"):
            m.cross_validate(X, y, n_folds=5)

    def test_load_model_syncs_loss(self):
        """Observation: load_model correctly syncs self.loss from model_info."""
        path = _make_model_json(
            features=_simple_features(2),
            trees=_simple_trees(),
            model_info={"loss_type": "Poisson"},
        )
        m = CatBoostMLXRegressor()
        assert m.loss == "rmse"  # default
        m.load_model(path)
        assert m.loss == "poisson"


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2: NEW-1 — Stale cat_features not cleared on load (HIGH)
# ════════════════════════════════════════════════════════════════════════════


class TestStaleCatFeatures:
    """load_model only sets cat_features when model HAS categoricals.
    If __init__ set cat_features=[0] and the loaded model has zero
    categoricals, the stale [0] persists → wrong predict dispatch."""

    def test_stale_cat_features_cleared_on_load(self):
        """Instance cat_features=[0] must be cleared after loading numeric-only model."""
        path = _make_model_json(
            features=_simple_features(3),  # no categoricals
            trees=_simple_trees(),
        )
        m = CatBoostMLXRegressor(cat_features=[0])
        assert m.cat_features == [0]
        m.load_model(path)
        # After loading a model with zero categoricals, cat_features must be None
        assert m.cat_features is None, (
            f"Stale cat_features={m.cat_features} persists after loading "
            f"numeric-only model"
        )

    def test_stale_cat_features_wrong_dispatch(self):
        """Stale cat_features routes to subprocess instead of inprocess."""
        path = _make_model_json(
            features=_simple_features(2),  # no categoricals
            trees=_simple_trees(),
        )
        m = CatBoostMLXRegressor(cat_features=[0])
        m.load_model(path)
        # With stale cat_features, _run_predict would try subprocess
        # which is wrong for a numeric-only model
        assert not m.cat_features, (
            f"cat_features={m.cat_features} forces subprocess for numeric model"
        )

    def test_stale_cat_features_sequential_loads(self):
        """Load model with cats, then load model without cats — stale persists."""
        tmpdir = tempfile.mkdtemp()
        # First model: has categorical at index 1
        path1 = _make_model_json(
            features=_simple_features(3, categoricals=[1]),
            trees=_simple_trees(),
            tmpdir=tmpdir,
        )
        # Rename to avoid overwrite
        path1_renamed = os.path.join(tmpdir, "model_cat.json")
        os.rename(path1, path1_renamed)

        # Second model: no categoricals
        path2 = _make_model_json(
            features=_simple_features(3),
            trees=_simple_trees(),
            tmpdir=tmpdir,
        )
        path2_renamed = os.path.join(tmpdir, "model_num.json")
        os.rename(path2, path2_renamed)

        m = CatBoostMLXRegressor()
        m.load_model(path1_renamed)
        assert m.cat_features == [1]  # correctly set from model

        m.load_model(path2_renamed)
        assert m.cat_features is None, (
            f"After loading numeric-only model, cat_features={m.cat_features} "
            f"(stale from previous load)"
        )


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3: NEW-2 — Case-sensitive logloss check in classifier (HIGH)
# ════════════════════════════════════════════════════════════════════════════


class TestClassifierLoglossCase:
    """CatBoostMLXClassifier.load_model checks
    info.get("loss_type", "").startswith("logloss") — lowercase.
    But CatBoost C++ writes "Logloss" (capital L).
    So the fallback path for binary classifiers NEVER triggers."""

    def test_classes_set_for_capital_logloss(self):
        """Binary model with loss_type='Logloss' (C++ casing) must set classes_."""
        path = _make_model_json(
            features=_simple_features(2),
            trees=_simple_trees(),
            model_info={"loss_type": "Logloss", "num_classes": 0},
        )
        clf = CatBoostMLXClassifier()
        clf.load_model(path)
        assert hasattr(clf, "classes_"), (
            "classes_ not set: 'Logloss'.startswith('logloss') is False"
        )
        assert list(clf.classes_) == [0, 1]

    def test_classes_set_for_lowercase_logloss(self):
        """Binary model with loss_type='logloss' does set classes_ (control)."""
        path = _make_model_json(
            features=_simple_features(2),
            trees=_simple_trees(),
            model_info={"loss_type": "logloss", "num_classes": 0},
        )
        clf = CatBoostMLXClassifier()
        clf.load_model(path)
        assert hasattr(clf, "classes_")
        assert list(clf.classes_) == [0, 1]

    def test_stale_classes_not_overwritten_binary(self):
        """Stale classes_ from prior fit persists after loading binary model."""
        path = _make_model_json(
            features=_simple_features(2),
            trees=_simple_trees(),
            model_info={"loss_type": "Logloss", "num_classes": 0},
        )
        clf = CatBoostMLXClassifier()
        clf.classes_ = np.array([10, 20, 30])  # stale from prior fit
        clf.load_model(path)
        # Neither branch triggers: num_classes=0, 'Logloss' != 'logloss'
        # So stale [10, 20, 30] persists for a binary model!
        assert list(clf.classes_) == [0, 1], (
            f"Stale classes_={list(clf.classes_)} persists for binary model"
        )


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4: NEW-3 — classes_ missing when num_classes=0 (HIGH)
# ════════════════════════════════════════════════════════════════════════════


class TestClassesMissingNumClassesZero:
    """When model_info has num_classes=0 (some exports, or non-standard labels),
    neither branch in classifier.load_model triggers, so classes_ is never set.
    approx_dimension could serve as a fallback but isn't used."""

    def test_multiclass_num_classes_zero(self):
        """Multiclass model with num_classes=0 but approx_dimension=3."""
        path = _make_model_json(
            features=_simple_features(2),
            trees=_multiclass_trees(3),
            model_info={"loss_type": "MultiClass", "num_classes": 0,
                        "approx_dimension": 3},
        )
        clf = CatBoostMLXClassifier()
        clf.load_model(path)
        assert hasattr(clf, "classes_"), (
            "classes_ not set: num_classes=0 and loss_type is not 'logloss'"
        )

    def test_approx_dimension_as_fallback(self):
        """approx_dimension should be used as fallback when num_classes=0."""
        path = _make_model_json(
            features=_simple_features(2),
            trees=_multiclass_trees(4),
            model_info={"loss_type": "MultiClass", "num_classes": 0,
                        "approx_dimension": 4},
        )
        clf = CatBoostMLXClassifier()
        clf.load_model(path)
        assert hasattr(clf, "classes_")
        # CatBoost approx_dimension = num_classes - 1, so approx_dim=4 → 5 classes
        assert len(clf.classes_) == 5

    def test_nonstandard_labels_roundtrip_lossy(self):
        """Non-standard labels like [10,20,30] cannot survive JSON roundtrip.

        This is a known limitation since model JSON only stores num_classes,
        not actual label values. classes_ IS set, but to [0,1,2] not [10,20,30].
        Documenting the lossy behavior, not flagging as a bug."""
        path = _make_model_json(
            features=_simple_features(2),
            trees=_multiclass_trees(3),
            model_info={"loss_type": "MultiClass", "num_classes": 3,
                        "approx_dimension": 3},
        )
        clf = CatBoostMLXClassifier()
        clf.load_model(path)
        assert hasattr(clf, "classes_")
        assert len(clf.classes_) == 3
        # Lossy: original labels [10,20,30] become [0,1,2]
        assert list(clf.classes_) == [0, 1, 2]


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5: Regression tests — things that still work
# ════════════════════════════════════════════════════════════════════════════


class TestLoadModelRegressions:
    """Ensure round 5 fixes didn't break any working paths."""

    def test_n_features_in_updates_on_sequential_loads(self):
        """Loading different models updates n_features_in_ correctly."""
        path5 = _make_model_json(features=_simple_features(5), trees=_simple_trees())
        path2 = _make_model_json(features=_simple_features(2), trees=_simple_trees())

        m = CatBoostMLXRegressor()
        m.load_model(path5)
        assert m.n_features_in_ == 5

        m.load_model(path2)
        assert m.n_features_in_ == 2

    def test_cat_features_correctly_set_from_model(self):
        """Loading model with categoricals sets cat_features (not stale)."""
        path = _make_model_json(
            features=_simple_features(3, categoricals=[1, 2]),
            trees=_simple_trees(),
        )
        m = CatBoostMLXRegressor()
        m.load_model(path)
        assert m.cat_features == [1, 2]

    def test_cat_features_overwritten_by_model(self):
        """Loading model with categoricals overwrites __init__ value."""
        path = _make_model_json(
            features=_simple_features(3, categoricals=[2]),
            trees=_simple_trees(),
        )
        m = CatBoostMLXRegressor(cat_features=[0])
        m.load_model(path)
        # Model says cat is at index 2, not 0
        assert m.cat_features == [2], f"Expected [2], got {m.cat_features}"

    def test_feature_names_restored(self):
        """Feature names are correctly restored from model JSON."""
        features = [
            {"index": 0, "name": "age", "borders": [0.5]},
            {"index": 1, "name": "income", "borders": [1.0]},
        ]
        path = _make_model_json(features=features, trees=_simple_trees())
        m = CatBoostMLXRegressor()
        m.load_model(path)
        assert list(m.feature_names_in_) == ["age", "income"]

    def test_multiclass_classes_correctly_set(self):
        """Multiclass with num_classes > 0 correctly sets classes_."""
        path = _make_model_json(
            features=_simple_features(2),
            trees=_multiclass_trees(5),
            model_info={"loss_type": "MultiClass", "num_classes": 5,
                        "approx_dimension": 5},
        )
        clf = CatBoostMLXClassifier()
        clf.load_model(path)
        assert hasattr(clf, "classes_")
        assert list(clf.classes_) == [0, 1, 2, 3, 4]

    def test_load_returns_self(self):
        """load_model returns self for method chaining."""
        path = _make_model_json(features=_simple_features(2), trees=_simple_trees())
        m = CatBoostMLXRegressor()
        result = m.load_model(path)
        assert result is m
