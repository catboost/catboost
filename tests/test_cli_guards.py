"""
DEC-032 closeout guards — Sprint 29 S29-CLI-GUARD-T2, task #83.
G1-CLI gate per DEC-031 Rule 3.

Covers SA-H1 bypass closure: _core.train() and csv_train CLI must reject
Cosine+{Lossguide, SymmetricTree} — both paths previously bypassed the
Python _validate_params guards landed in Sprint 28 commit b9577067ef.

Four cases:
  1. _core.train() direct → ValueError with TODO-S29-LG-COSINE-RCA marker
  2. _core.train() direct → ValueError with TODO-S29-ST-COSINE-KAHAN marker
  3. csv_train CLI       → non-zero exit + TODO-S29-LG-COSINE-RCA in stderr
  4. csv_train CLI       → non-zero exit + TODO-S29-ST-COSINE-KAHAN in stderr

CLI tests assert returncode != 0 (not == 1) so they survive a future
main() try/catch cleanup that replaces SIGABRT(134/-6) with exit(1).
See S29-CR for the deferred cleanup.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Locate the csv_train binary built by #82.
# The post-#82 binary lives at python/catboost_mlx/bin/csv_train (arm64,
# built ~2026-04-23 17:00).  The repo-root csv_train is an older build that
# predates the --score-function flag and cannot test these guards.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CSV_TRAIN_BIN = _REPO_ROOT / "python" / "catboost_mlx" / "bin" / "csv_train"

_CSV_TRAIN_MISSING = not _CSV_TRAIN_BIN.exists()
_SKIP_CLI = pytest.mark.skipif(
    _CSV_TRAIN_MISSING,
    reason=f"csv_train binary not found at {_CSV_TRAIN_BIN} — build first",
)

# ---------------------------------------------------------------------------
# Minimal TrainConfig builder for nanobind direct-train tests.
# Only the fields needed to reach TrainConfigToInternal are set.
# The guard fires at the top of TrainConfigToInternal, before any data load.
# ---------------------------------------------------------------------------


def _make_minimal_config(grow_policy: str, score_function: str):
    """Return a TrainConfig with the forbidden Cosine combo."""
    from catboost_mlx import _core

    cfg = _core.TrainConfig()
    cfg.num_iterations = 1
    cfg.max_depth = 2
    cfg.learning_rate = 0.1
    cfg.l2_reg_lambda = 3.0
    cfg.max_bins = 16
    cfg.loss_type = "RMSE"
    cfg.eval_fraction = 0.0
    cfg.early_stopping_patience = 0
    cfg.subsample_ratio = 1.0
    cfg.colsample_by_tree = 1.0
    cfg.random_seed = 42
    cfg.random_strength = 0.0
    cfg.bootstrap_type = "no"
    cfg.bagging_temperature = 1.0
    cfg.mvs_reg = 0.0
    cfg.nan_mode = "min"
    cfg.use_ctr = False
    cfg.ctr_prior = 0.5
    cfg.max_onehot_size = 10
    cfg.min_data_in_leaf = 1
    cfg.monotone_constraints = []
    cfg.max_leaves = 31
    cfg.snapshot_path = ""
    cfg.snapshot_interval = 1
    cfg.verbose = False
    cfg.compute_feature_importance = False
    cfg.grow_policy = grow_policy
    cfg.score_function = score_function
    return cfg


def _minimal_arrays():
    """Tiny arrays that satisfy the nanobind signature — data never read by guard."""
    X = np.zeros((5, 2), dtype=np.float32)
    y = np.arange(5, dtype=np.float32)
    w = np.ones(5, dtype=np.float32)
    g = np.zeros(5, dtype=np.float32)
    val_X = np.zeros((0, 2), dtype=np.float32)
    val_y = np.zeros(0, dtype=np.float32)
    feature_names = ["f0", "f1"]
    is_categorical = [False, False]
    cat_hash_maps = [{}, {}]
    return X, y, w, g, val_X, val_y, feature_names, is_categorical, cat_hash_maps


# ---------------------------------------------------------------------------
# Nanobind direct-train tests
# ---------------------------------------------------------------------------


def test_core_train_rejects_cosine_lossguide():
    """_core.train() must raise ValueError for score_function='Cosine' + grow_policy='Lossguide'.

    Guard placed in TrainConfigToInternal (train_api.cpp) as C++ defense-in-depth,
    mirroring the Python-layer guard in core.py:628-636.  nanobind auto-translates
    std::invalid_argument → Python ValueError.

    The TODO-S29-LG-COSINE-RCA token is greppable across C++ and Python sources,
    providing a stable marker for future RCA resolution.
    """
    from catboost_mlx import _core

    cfg = _make_minimal_config(grow_policy="Lossguide", score_function="Cosine")
    X, y, w, g, val_X, val_y, fn, ic, chm = _minimal_arrays()

    # covers: C++/nanobind guard, Cosine+Lossguide rejection, SA-H1 bypass closure
    with pytest.raises(ValueError) as exc_info:
        _core.train(
            features=X,
            targets=y,
            feature_names=fn,
            is_categorical=ic,
            weights=w,
            group_ids=g,
            cat_hash_maps=chm,
            val_features=val_X,
            val_targets=val_y,
            config=cfg,
        )

    assert "TODO-S29-LG-COSINE-RCA" in str(exc_info.value), (
        f"Expected TODO-S29-LG-COSINE-RCA marker in ValueError message, got: "
        f"{exc_info.value}"
    )


def test_core_train_rejects_cosine_symmetric_tree():
    """_core.train() must raise ValueError for score_function='Cosine' + grow_policy='SymmetricTree'.

    Guard placed in TrainConfigToInternal (train_api.cpp) as C++ defense-in-depth,
    mirroring the Python-layer guard in core.py:638-647.  nanobind auto-translates
    std::invalid_argument → Python ValueError.

    The TODO-S29-ST-COSINE-KAHAN token is greppable across C++ and Python sources,
    providing a stable marker for when the Kahan/Neumaier compensated-summation fix
    ships in Sprint 29 (at which point this guard — and this test — will be removed).
    """
    from catboost_mlx import _core

    cfg = _make_minimal_config(grow_policy="SymmetricTree", score_function="Cosine")
    X, y, w, g, val_X, val_y, fn, ic, chm = _minimal_arrays()

    # covers: C++/nanobind guard, Cosine+SymmetricTree rejection, SA-H1 bypass closure
    with pytest.raises(ValueError) as exc_info:
        _core.train(
            features=X,
            targets=y,
            feature_names=fn,
            is_categorical=ic,
            weights=w,
            group_ids=g,
            cat_hash_maps=chm,
            val_features=val_X,
            val_targets=val_y,
            config=cfg,
        )

    assert "TODO-S29-ST-COSINE-KAHAN" in str(exc_info.value), (
        f"Expected TODO-S29-ST-COSINE-KAHAN marker in ValueError message, got: "
        f"{exc_info.value}"
    )


# ---------------------------------------------------------------------------
# CLI subprocess tests
# ---------------------------------------------------------------------------


@_SKIP_CLI
def test_csv_train_cli_rejects_cosine_lossguide():
    """csv_train CLI must exit non-zero and emit TODO-S29-LG-COSINE-RCA on stderr.

    Guard fires in ParseArgs (csv_train.cpp:244-252) after flag parsing,
    before file open — so /dev/null as the CSV path argument is safe.

    returncode is asserted != 0 (not == 1): current behavior is SIGABRT (-6)
    because main() has no top-level try/catch.  This assertion survives a future
    main() cleanup that catches std::invalid_argument and returns 1 (S29-CR).

    Binary: python/catboost_mlx/bin/csv_train (the post-#82 build).
    """
    result = subprocess.run(
        [
            str(_CSV_TRAIN_BIN),
            "/dev/null",
            "--grow-policy", "Lossguide",
            "--score-function", "Cosine",
        ],
        capture_output=True,
        text=True,
    )

    # covers: CLI guard, Cosine+Lossguide rejection, non-zero exit, SA-H1 bypass closure
    assert result.returncode != 0, (
        f"csv_train should have exited non-zero for Cosine+Lossguide, "
        f"got returncode={result.returncode}"
    )
    assert "TODO-S29-LG-COSINE-RCA" in result.stderr, (
        f"Expected TODO-S29-LG-COSINE-RCA marker in csv_train stderr, got:\n"
        f"{result.stderr[:500]}"
    )


@_SKIP_CLI
def test_csv_train_cli_rejects_cosine_symmetric_tree():
    """csv_train CLI must exit non-zero and emit TODO-S29-ST-COSINE-KAHAN on stderr.

    Guard fires in ParseArgs (csv_train.cpp:257-266) after flag parsing,
    before file open — so /dev/null as the CSV path argument is safe.

    returncode is asserted != 0 (not == 1) for the same S29-CR reason as the
    Lossguide test above.

    Binary: python/catboost_mlx/bin/csv_train (the post-#82 build).
    """
    result = subprocess.run(
        [
            str(_CSV_TRAIN_BIN),
            "/dev/null",
            "--grow-policy", "SymmetricTree",
            "--score-function", "Cosine",
        ],
        capture_output=True,
        text=True,
    )

    # covers: CLI guard, Cosine+SymmetricTree rejection, non-zero exit, SA-H1 bypass closure
    assert result.returncode != 0, (
        f"csv_train should have exited non-zero for Cosine+SymmetricTree, "
        f"got returncode={result.returncode}"
    )
    assert "TODO-S29-ST-COSINE-KAHAN" in result.stderr, (
        f"Expected TODO-S29-ST-COSINE-KAHAN marker in csv_train stderr, got:\n"
        f"{result.stderr[:500]}"
    )
