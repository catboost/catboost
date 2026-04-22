"""
test_qa_round13_sprint10.py -- QA Round 13: Sprint 10 feature validation.

Sprint 10 (mlx/sprint-10-lossguide-packaging-hardening) delivered:

  1. Lossguide grow policy: best-first leaf-wise trees, grow_policy="Lossguide",
     --max-leaves N, priority-queue-based structure_searcher.
  2. Model format versioning: format_version=2 written by save_model,
     read and range-checked by load_model.
  3. Benchmark script: python/benchmarks/benchmark_vs_catboost.py
  4. PyPI packaging: version 0.3.0, pyproject.toml, coverage threshold 70%
  5. README: python/README.md

This file validates the Python-layer surface area of items 1, 2, and 4:

  SECTION 1 -- Lossguide: _validate_params acceptance
    1.  grow_policy="Lossguide" accepted (no exception raised)
    2.  grow_policy="Lossguide" with default max_leaves=31 accepted
    3.  grow_policy=None accepted (falls back to SymmetricTree)
    4.  grow_policy="" accepted (falls back to SymmetricTree)
    5.  grow_policy="SymmetricTree" still accepted (regression)
    6.  grow_policy="Depthwise" still accepted (regression)
    7.  grow_policy="FooPolicy" rejected with ValueError
    8.  grow_policy="lossguide" (lowercase) rejected — case-sensitive validation

  SECTION 2 -- max_leaves: _validate_params boundary conditions
    9.  max_leaves=2 accepted (minimum valid value)
    10. max_leaves=1 rejected with ValueError
    11. max_leaves=0 rejected with ValueError
    12. max_leaves=-1 rejected with ValueError
    13. max_leaves=True rejected (bool coercion guard)
    14. max_leaves=False rejected (bool coercion guard)
    15. max_leaves=2.0 rejected (float not accepted)
    16. max_leaves=31 (default) accepted (regression)
    17. max_leaves=1000 accepted (no upper bound documented)

  SECTION 3 -- max_leaves CLI wiring
    18. Lossguide appends --grow-policy + --max-leaves to _build_train_args
    19. --max-leaves value matches max_leaves parameter
    20. SymmetricTree does NOT append --grow-policy or --max-leaves
    21. Depthwise appends --grow-policy Depthwise but NOT --max-leaves
    22. max_leaves=64 is emitted verbatim (not a default-elision issue)

  SECTION 4 -- grow_policy in get_params
    23. get_params() contains "grow_policy" key
    24. get_params() returns correct value for Lossguide
    25. get_params() contains "max_leaves" key
    26. get_params() returns correct max_leaves value
    27. set_params(grow_policy="Lossguide", max_leaves=16) round-trips cleanly

  SECTION 5 -- Model format versioning
    28. save_model writes "format_version": 2 at top level of JSON
    29. load_model accepts format_version=2 without error
    30. load_model accepts format_version=1 without error (older compat)
    31. load_model defaults to v1 compat when format_version key is absent
    32. load_model raises ValueError for format_version=3 (future version)
    33. load_model raises ValueError for format_version=99
    34. format_version is removed from _model_data after load (not leaked to downstream)
    35. Required keys still validated after format_version is stripped

  SECTION 6 -- Package version
    36. catboost_mlx.__version__ == "0.3.0"
    37. __version__ is a str, not None or a non-string

  SECTION 7 -- Adversarial / edge cases
    38. max_leaves=2, grow_policy="SymmetricTree" -- both params valid simultaneously
    39. grow_policy changes via set_params are reflected in _build_train_args
    40. Regressor and Classifier both accept grow_policy="Lossguide"

Bugs found during source review:

  BUG-006: _validate_params validates max_leaves unconditionally, regardless of
           grow_policy. A user who sets grow_policy="SymmetricTree" (default)
           and max_leaves=1 gets a ValueError for a parameter that is documented
           as "Ignored for other grow policies." This is arguably over-eager
           validation and could surprise users who pass max_leaves without
           explicitly using Lossguide.
           File: python/catboost_mlx/core.py:593
           Severity: Low (correct guard fires, but wrong context; max_leaves<2
           is caught even when grow_policy is not Lossguide).
           NOTE: The test suite acknowledges this behavior without endorsing it.

  BUG-007: grow_policy="lossguide" (lowercase) is correctly rejected at
           _validate_params time, but --grow-policy lossguide would also fail
           at the binary layer. The validation message says the accepted values
           are 'SymmetricTree', 'Depthwise', or 'Lossguide' — capital-first
           only. Consistent, but worth documenting.
"""

import json
import os
import tempfile
import unittest.mock as mock

import numpy as np
import pytest

import catboost_mlx
from catboost_mlx import CatBoostMLXClassifier, CatBoostMLXRegressor
from catboost_mlx.core import CatBoostMLX

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_regressor(**kwargs) -> CatBoostMLXRegressor:
    """Return a CatBoostMLXRegressor with fast defaults and the given overrides."""
    defaults = dict(iterations=5, depth=3, learning_rate=0.1)
    defaults.update(kwargs)
    return CatBoostMLXRegressor(**defaults)


def _make_classifier(**kwargs) -> CatBoostMLXClassifier:
    """Return a CatBoostMLXClassifier with fast defaults and the given overrides."""
    defaults = dict(iterations=5, depth=3, learning_rate=0.1)
    defaults.update(kwargs)
    return CatBoostMLXClassifier(**defaults)


def _minimal_model_json(tmpdir: str, format_version=None) -> str:
    """Write a minimal valid model JSON file and return its path.

    Includes format_version in the payload when specified.  When format_version
    is None the key is omitted entirely (simulates a pre-versioning model file).
    """
    features = [{"index": 0, "name": "f0", "borders": [0.5]}]
    trees = [{"depth": 1, "splits": [{"feature_idx": 0, "bin_threshold": 0}],
              "leaf_values": [-0.1, 0.1]}]
    model_info = {"loss_type": "RMSE"}
    payload = {"model_info": model_info, "features": features, "trees": trees}
    if format_version is not None:
        payload = {"format_version": format_version, **payload}
    path = os.path.join(tmpdir, "model.json")
    with open(path, "w") as f:
        json.dump(payload, f)
    return path


def _fitted_regressor_model_json(tmpdir: str) -> str:
    """Simulate what save_model writes: a format_version=2 JSON."""
    features = [{"index": 0, "name": "f0", "borders": [0.5]}]
    trees = [{"depth": 1, "splits": [{"feature_idx": 0, "bin_threshold": 0}],
              "leaf_values": [-0.1, 0.1]}]
    model_info = {"loss_type": "RMSE", "num_classes": 0}
    payload = {
        "format_version": 2,
        "model_info": model_info,
        "features": features,
        "trees": trees,
    }
    path = os.path.join(tmpdir, "model_v2.json")
    with open(path, "w") as f:
        json.dump(payload, f)
    return path


# ============================================================================
# SECTION 1: Lossguide -- _validate_params acceptance
# ============================================================================

class TestLossguideValidation:

    def test_lossguide_accepted_by_validate_params(self):
        """grow_policy='Lossguide' passes _validate_params without error."""
        m = _make_regressor(grow_policy="Lossguide", max_leaves=31)
        # Should not raise
        m._validate_params()

    def test_lossguide_with_default_max_leaves_accepted(self):
        """grow_policy='Lossguide' with max_leaves=31 (default) is valid."""
        m = _make_regressor(grow_policy="Lossguide")
        assert m.max_leaves == 31
        m._validate_params()

    def test_grow_policy_none_accepted(self):
        """grow_policy=None is tolerated (treated as SymmetricTree at CLI layer)."""
        m = _make_regressor(grow_policy=None)
        m._validate_params()

    def test_grow_policy_empty_string_accepted(self):
        """grow_policy='' is tolerated (treated as SymmetricTree at CLI layer)."""
        m = _make_regressor(grow_policy="")
        m._validate_params()

    def test_symmetrictree_still_accepted(self):
        """grow_policy='SymmetricTree' (default) is still valid -- regression."""
        m = _make_regressor(grow_policy="SymmetricTree")
        m._validate_params()

    def test_depthwise_still_accepted(self):
        """grow_policy='Depthwise' from Sprint 9 is still valid -- regression."""
        m = _make_regressor(grow_policy="Depthwise")
        m._validate_params()

    def test_unknown_policy_rejected(self):
        """grow_policy='FooPolicy' raises ValueError."""
        m = _make_regressor(grow_policy="FooPolicy")
        with pytest.raises(ValueError, match="grow_policy"):
            m._validate_params()

    def test_lossguide_lowercase_rejected(self):
        """grow_policy='lossguide' (lowercase) raises ValueError -- case-sensitive."""
        m = _make_regressor(grow_policy="lossguide")
        with pytest.raises(ValueError, match="grow_policy"):
            m._validate_params()


# ============================================================================
# SECTION 2: max_leaves -- boundary conditions
# ============================================================================

class TestMaxLeavesBoundaries:

    def test_max_leaves_minimum_boundary_accepted(self):
        """max_leaves=2 is the documented minimum and must be accepted."""
        m = _make_regressor(grow_policy="Lossguide", max_leaves=2)
        m._validate_params()

    def test_max_leaves_one_rejected(self):
        """max_leaves=1 is below the minimum and must be rejected."""
        m = _make_regressor(grow_policy="Lossguide", max_leaves=1)
        with pytest.raises(ValueError, match="max_leaves"):
            m._validate_params()

    def test_max_leaves_zero_rejected(self):
        """max_leaves=0 is rejected."""
        m = _make_regressor(grow_policy="Lossguide", max_leaves=0)
        with pytest.raises(ValueError, match="max_leaves"):
            m._validate_params()

    def test_max_leaves_negative_rejected(self):
        """max_leaves=-1 is rejected."""
        m = _make_regressor(grow_policy="Lossguide", max_leaves=-1)
        with pytest.raises(ValueError, match="max_leaves"):
            m._validate_params()

    def test_max_leaves_bool_true_rejected(self):
        """max_leaves=True rejected -- bool is int subclass, must be guarded."""
        m = _make_regressor(grow_policy="Lossguide", max_leaves=True)
        with pytest.raises(ValueError, match="max_leaves"):
            m._validate_params()

    def test_max_leaves_bool_false_rejected(self):
        """max_leaves=False rejected -- bool is int subclass, must be guarded."""
        m = _make_regressor(grow_policy="Lossguide", max_leaves=False)
        with pytest.raises(ValueError, match="max_leaves"):
            m._validate_params()

    def test_max_leaves_float_rejected(self):
        """max_leaves=2.0 is rejected -- must be int not float."""
        m = _make_regressor(grow_policy="Lossguide", max_leaves=2.0)
        with pytest.raises(ValueError, match="max_leaves"):
            m._validate_params()

    def test_max_leaves_default_31_accepted(self):
        """max_leaves=31 (constructor default) is accepted -- regression."""
        m = _make_regressor()
        assert m.max_leaves == 31
        m._validate_params()

    def test_max_leaves_large_value_accepted(self):
        """max_leaves=1000 is accepted -- no documented upper bound."""
        m = _make_regressor(grow_policy="Lossguide", max_leaves=1000)
        m._validate_params()

    def test_max_leaves_string_rejected(self):
        """max_leaves='31' (string) is rejected."""
        m = _make_regressor(grow_policy="Lossguide", max_leaves="31")
        with pytest.raises(ValueError, match="max_leaves"):
            m._validate_params()


# ============================================================================
# SECTION 3: max_leaves CLI wiring via _build_train_args
# ============================================================================

class TestMaxLeavesCliWiring:
    """Tests that _build_train_args produces the correct flags.

    We mock _find_binary so no filesystem access is needed.
    """

    def _get_args(self, **kwargs) -> list:
        """Return _build_train_args output for a model with given params.

        Mocks _find_binary to avoid needing the actual binary on disk.
        """
        m = _make_regressor(**kwargs)
        with mock.patch("catboost_mlx.core._find_binary", return_value="/mock/csv_train"):
            return m._build_train_args("/tmp/train.csv", "/tmp/model.json", 0)

    def test_lossguide_appends_grow_policy_flag(self):
        """grow_policy='Lossguide' emits --grow-policy Lossguide."""
        args = self._get_args(grow_policy="Lossguide", max_leaves=31)
        assert "--grow-policy" in args
        idx = args.index("--grow-policy")
        assert args[idx + 1] == "Lossguide"

    def test_lossguide_appends_max_leaves_flag(self):
        """grow_policy='Lossguide' emits --max-leaves N."""
        args = self._get_args(grow_policy="Lossguide", max_leaves=31)
        assert "--max-leaves" in args

    def test_max_leaves_value_is_correct(self):
        """--max-leaves flag carries the actual max_leaves parameter value."""
        args = self._get_args(grow_policy="Lossguide", max_leaves=64)
        idx = args.index("--max-leaves")
        assert args[idx + 1] == "64"

    def test_symmetrictree_does_not_emit_grow_policy_flag(self):
        """SymmetricTree (default) does not emit --grow-policy."""
        args = self._get_args(grow_policy="SymmetricTree")
        assert "--grow-policy" not in args

    def test_symmetrictree_does_not_emit_max_leaves_flag(self):
        """SymmetricTree (default) does not emit --max-leaves."""
        args = self._get_args(grow_policy="SymmetricTree")
        assert "--max-leaves" not in args

    def test_depthwise_emits_grow_policy_but_not_max_leaves(self):
        """Depthwise emits --grow-policy Depthwise but NOT --max-leaves."""
        args = self._get_args(grow_policy="Depthwise")
        assert "--grow-policy" in args
        idx = args.index("--grow-policy")
        assert args[idx + 1] == "Depthwise"
        assert "--max-leaves" not in args

    def test_max_leaves_64_emitted_verbatim(self):
        """max_leaves=64 is emitted as '64', not suppressed as a near-default."""
        args = self._get_args(grow_policy="Lossguide", max_leaves=64)
        assert "--max-leaves" in args
        idx = args.index("--max-leaves")
        assert args[idx + 1] == "64"

    def test_max_leaves_2_emitted_verbatim(self):
        """max_leaves=2 (boundary minimum) is emitted correctly."""
        args = self._get_args(grow_policy="Lossguide", max_leaves=2)
        assert "--max-leaves" in args
        idx = args.index("--max-leaves")
        assert args[idx + 1] == "2"


# ============================================================================
# SECTION 4: grow_policy in get_params / set_params
# ============================================================================

class TestGrowPolicyGetSetParams:

    def test_get_params_contains_grow_policy_key(self):
        """get_params() must expose 'grow_policy'."""
        m = _make_regressor()
        assert "grow_policy" in m.get_params()

    def test_get_params_returns_lossguide(self):
        """get_params() returns 'Lossguide' when that policy was set."""
        m = _make_regressor(grow_policy="Lossguide")
        assert m.get_params()["grow_policy"] == "Lossguide"

    def test_get_params_contains_max_leaves_key(self):
        """get_params() must expose 'max_leaves'."""
        m = _make_regressor()
        assert "max_leaves" in m.get_params()

    def test_get_params_returns_correct_max_leaves(self):
        """get_params() returns the max_leaves value that was set."""
        m = _make_regressor(max_leaves=16)
        assert m.get_params()["max_leaves"] == 16

    def test_set_params_grow_policy_and_max_leaves_round_trip(self):
        """set_params(grow_policy='Lossguide', max_leaves=16) is reflected in get_params."""
        m = _make_regressor()
        m.set_params(grow_policy="Lossguide", max_leaves=16)
        params = m.get_params()
        assert params["grow_policy"] == "Lossguide"
        assert params["max_leaves"] == 16

    def test_set_params_updates_instance_attributes(self):
        """set_params side-effects: m.grow_policy and m.max_leaves are updated."""
        m = _make_regressor()
        m.set_params(grow_policy="Lossguide", max_leaves=8)
        assert m.grow_policy == "Lossguide"
        assert m.max_leaves == 8

    def test_classifier_get_params_contains_grow_policy(self):
        """Classifier's get_params() also exposes 'grow_policy'."""
        m = _make_classifier(grow_policy="Lossguide")
        assert m.get_params()["grow_policy"] == "Lossguide"


# ============================================================================
# SECTION 5: Model format versioning
# ============================================================================

class TestModelFormatVersioning:

    def test_save_model_writes_format_version_2(self, tmp_path):
        """save_model must write format_version=2 at the JSON top level."""
        features = [{"index": 0, "name": "f0", "borders": [0.5]}]
        trees = [{"depth": 1, "splits": [{"feature_idx": 0, "bin_threshold": 0}],
                  "leaf_values": [-0.1, 0.1]}]
        model_info = {"loss_type": "RMSE"}
        m = _make_regressor()
        m._model_data = {"model_info": model_info, "features": features, "trees": trees}
        m._is_fitted = True
        out_path = str(tmp_path / "model.json")
        m.save_model(out_path)
        with open(out_path) as f:
            raw = json.load(f)
        assert raw.get("format_version") == 2

    def test_save_model_format_version_is_integer_not_string(self, tmp_path):
        """format_version must be JSON int 2, not string '2'."""
        features = [{"index": 0, "name": "f0", "borders": [0.5]}]
        trees = [{"depth": 1, "splits": [{"feature_idx": 0, "bin_threshold": 0}],
                  "leaf_values": [-0.1, 0.1]}]
        m = _make_regressor()
        m._model_data = {"model_info": {"loss_type": "RMSE"}, "features": features, "trees": trees}
        m._is_fitted = True
        out_path = str(tmp_path / "model.json")
        m.save_model(out_path)
        with open(out_path) as f:
            raw = json.load(f)
        assert isinstance(raw["format_version"], int)
        assert raw["format_version"] == 2

    def test_load_model_accepts_format_version_2(self, tmp_path):
        """load_model must succeed when format_version=2 is present."""
        path = _minimal_model_json(str(tmp_path), format_version=2)
        m = _make_regressor()
        m.load_model(path)  # must not raise
        assert m._is_fitted is True

    def test_load_model_accepts_format_version_1(self, tmp_path):
        """load_model must accept format_version=1 (prior minor version)."""
        path = _minimal_model_json(str(tmp_path), format_version=1)
        m = _make_regressor()
        m.load_model(path)
        assert m._is_fitted is True

    def test_load_model_absent_format_version_defaults_to_v1_compat(self, tmp_path):
        """load_model with no format_version key must succeed (treats as v1)."""
        path = _minimal_model_json(str(tmp_path), format_version=None)
        # Verify the file genuinely has no format_version key
        with open(path) as f:
            raw = json.load(f)
        assert "format_version" not in raw
        m = _make_regressor()
        m.load_model(path)
        assert m._is_fitted is True

    def test_load_model_raises_for_format_version_3(self, tmp_path):
        """load_model must raise ValueError for format_version=3 (future)."""
        path = _minimal_model_json(str(tmp_path), format_version=3)
        m = _make_regressor()
        with pytest.raises(ValueError, match="format_version"):
            m.load_model(path)

    def test_load_model_raises_for_format_version_99(self, tmp_path):
        """load_model must raise ValueError for format_version=99 (far future)."""
        path = _minimal_model_json(str(tmp_path), format_version=99)
        m = _make_regressor()
        with pytest.raises(ValueError, match="format_version"):
            m.load_model(path)

    def test_format_version_not_leaked_into_model_data(self, tmp_path):
        """After load_model, format_version must not appear in _model_data."""
        path = _minimal_model_json(str(tmp_path), format_version=2)
        m = _make_regressor()
        m.load_model(path)
        assert "format_version" not in m._model_data

    def test_missing_required_key_after_version_strip_raises(self, tmp_path):
        """load_model must still validate required keys after stripping format_version."""
        # Build a file that has format_version but is missing 'trees'
        features = [{"index": 0, "name": "f0", "borders": [0.5]}]
        model_info = {"loss_type": "RMSE"}
        payload = {
            "format_version": 2,
            "model_info": model_info,
            "features": features,
            # deliberately omit 'trees'
        }
        path = str(tmp_path / "bad_model.json")
        with open(path, "w") as f:
            json.dump(payload, f)
        m = _make_regressor()
        with pytest.raises(ValueError, match="trees"):
            m.load_model(path)

    def test_save_then_load_round_trip_preserves_model_data(self, tmp_path):
        """save_model + load_model round-trip: key model fields survive."""
        features = [{"index": 0, "name": "feat_x", "borders": [0.5, 1.5]}]
        trees = [{"depth": 1, "splits": [{"feature_idx": 0, "bin_threshold": 0}],
                  "leaf_values": [-0.2, 0.3]}]
        model_info = {"loss_type": "RMSE", "num_classes": 0}
        m = _make_regressor()
        m._model_data = {"model_info": model_info, "features": features, "trees": trees}
        m._is_fitted = True
        out_path = str(tmp_path / "round_trip.json")
        m.save_model(out_path)

        m2 = _make_regressor()
        m2.load_model(out_path)
        assert m2._is_fitted is True
        assert m2._model_data["features"] == features
        assert m2._model_data["trees"] == trees

    def test_load_class_method_also_strips_format_version(self, tmp_path):
        """CatBoostMLX.load() classmethod must not expose format_version in _model_data."""
        path = _fitted_regressor_model_json(str(tmp_path))
        m = CatBoostMLX.load(path)
        assert "format_version" not in m._model_data
        assert m._is_fitted is True


# ============================================================================
# SECTION 6: Package version
# ============================================================================

class TestPackageVersion:

    def test_version_matches_pyproject(self):
        from importlib.metadata import version as _dist_version
        assert catboost_mlx.__version__ == _dist_version("catboost-mlx")

    def test_version_is_a_string(self):
        """catboost_mlx.__version__ must be a str."""
        assert isinstance(catboost_mlx.__version__, str)

    def test_version_is_not_none(self):
        """catboost_mlx.__version__ must not be None."""
        assert catboost_mlx.__version__ is not None

    def test_version_has_three_components(self):
        """__version__ follows MAJOR.MINOR.PATCH semver structure."""
        parts = catboost_mlx.__version__.split(".")
        assert len(parts) == 3, f"Expected 3 parts, got {parts}"
        for part in parts:
            assert part.isdigit(), f"Non-numeric version component: {part!r}"

    def test_version_major_is_0(self):
        """Major version is 0 (pre-1.0 alpha)."""
        major = int(catboost_mlx.__version__.split(".")[0])
        assert major == 0


# ============================================================================
# SECTION 7: Adversarial / edge cases
# ============================================================================

class TestAdversarialEdgeCases:

    def test_max_leaves_ignored_with_symmetrictree_policy(self):
        """max_leaves is not validated when grow_policy != 'Lossguide' (BUG-006 fixed)."""
        m = _make_regressor(grow_policy="SymmetricTree", max_leaves=2)
        m._validate_params()  # should not raise

    def test_max_leaves_1_with_symmetrictree_policy_accepted(self):
        """max_leaves=1 accepted under SymmetricTree since it's ignored (BUG-006 fixed)."""
        m = _make_regressor(grow_policy="SymmetricTree", max_leaves=1)
        m._validate_params()  # should not raise — only validated for Lossguide

    def test_set_params_then_build_train_args_reflects_update(self):
        """grow_policy changed via set_params is immediately reflected in CLI args."""
        m = _make_regressor(grow_policy="SymmetricTree")
        m.set_params(grow_policy="Lossguide", max_leaves=8)
        with mock.patch("catboost_mlx.core._find_binary", return_value="/mock/csv_train"):
            args = m._build_train_args("/tmp/train.csv", "/tmp/model.json", 0)
        assert "--grow-policy" in args
        assert "Lossguide" in args
        assert "--max-leaves" in args

    def test_regressor_accepts_lossguide(self):
        """CatBoostMLXRegressor accepts grow_policy='Lossguide'."""
        m = CatBoostMLXRegressor(iterations=5, grow_policy="Lossguide", max_leaves=4)
        m._validate_params()

    def test_classifier_accepts_lossguide(self):
        """CatBoostMLXClassifier accepts grow_policy='Lossguide'."""
        m = CatBoostMLXClassifier(iterations=5, grow_policy="Lossguide", max_leaves=4)
        m._validate_params()

    def test_load_model_format_version_0_accepted(self, tmp_path):
        """format_version=0 is <= 2 so load_model must accept it without error.

        Edge case: 0 is a valid integer <= 2. The check is `fmt_version > 2`.
        This should succeed, not raise.
        """
        path = _minimal_model_json(str(tmp_path), format_version=0)
        m = _make_regressor()
        m.load_model(path)
        assert m._is_fitted is True

    def test_load_model_format_version_exact_boundary_2(self, tmp_path):
        """format_version=2 is the current max; must not raise."""
        path = _minimal_model_json(str(tmp_path), format_version=2)
        m = _make_regressor()
        m.load_model(path)

    def test_load_model_format_version_exact_boundary_3_raises(self, tmp_path):
        """format_version=3 is one above current max; must raise."""
        path = _minimal_model_json(str(tmp_path), format_version=3)
        m = _make_regressor()
        with pytest.raises(ValueError):
            m.load_model(path)

    def test_lossguide_max_leaves_none_rejected(self):
        """max_leaves=None is rejected (not an int) -- raises ValueError."""
        m = _make_regressor(grow_policy="Lossguide", max_leaves=None)
        with pytest.raises(ValueError, match="max_leaves"):
            m._validate_params()
