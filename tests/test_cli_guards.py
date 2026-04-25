"""
DEC-032 / DEC-042 guard lifecycle — S29-CLI-GUARD-T2 (#83) + S33-L4-FIX Commits 3a/3b (#93/#94).

Original scope (S29): SA-H1 bypass closure — _core.train() and csv_train CLI must reject
Cosine+{Lossguide, SymmetricTree} in all entry paths (Python layer, C++ nanobind, CLI).

Current state after S33-L4-FIX Commits 3a+3b (2026-04-25):
  - S28-ST-GUARD REMOVED (#93): ST+Cosine validated (G4a 0.0001%, G4b 0.027%; DEC-042).
    Tests 2 and 4 below assert ST+Cosine succeeds.
  - S28-LG-GUARD REMOVED (#94): LG+Cosine validated post-fix (iter=1 0.0000%, iter=50 0.382%).
    Tests 1 and 3 below assert LG+Cosine succeeds.

All four cases are now positive acceptance tests (DEC-042 fully closed):
  1. _core.train() direct → succeeds for LG+Cosine (no ValueError)  [INVERTED by #94]
  2. _core.train() direct → succeeds for ST+Cosine (no ValueError)  [INVERTED by #93]
  3. csv_train CLI       → exit 0 for LG+Cosine                     [INVERTED by #94]
  4. csv_train CLI       → exit 0 for ST+Cosine                     [INVERTED by #93]
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


def test_core_train_accepts_cosine_lossguide():
    """_core.train() must succeed for score_function='Cosine' + grow_policy='Lossguide'.

    S28-LG-GUARD removed in S33-L4-FIX Commit 3b (#94) after drift measurement:
      iter=1  drift 0.0000%  (<=0.1% threshold analogous to G4a) PASS
      iter=50 drift 0.382%   (<=2% threshold analogous to G4b)   PASS
      Anchor: rng(42), N=50k, 20 features, LG/Cosine/RMSE, d=6, max_leaves=64,
              bins=128, l2=3, lr=0.03, rs=0

    Guard was in TrainConfigToInternal (train_api.cpp) + Python core.py. Both removed.
    This test verifies the combination trains without raising.

    covers: S28-LG-GUARD removal, DEC-042 full closure, Commit 3b (#94)
    """
    from catboost_mlx import _core

    cfg = _make_minimal_config(grow_policy="Lossguide", score_function="Cosine")
    X, y, w, g, val_X, val_y, fn, ic, chm = _minimal_arrays()

    # Must not raise — LG+Cosine is now a supported combination
    result = _core.train(
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
    assert result is not None, "Expected a non-None result from LG+Cosine train"


def test_core_train_accepts_cosine_symmetric_tree():
    """_core.train() must succeed for score_function='Cosine' + grow_policy='SymmetricTree'.

    S28-ST-GUARD removed in S33-L4-FIX Commit 3a (#93) after DEC-042 validation:
      G4a iter=1 drift 0.0001% (<=0.1% threshold) PASS
      G4b iter=50 drift 0.027%  (<=2% threshold)  PASS

    Guard was in TrainConfigToInternal (train_api.cpp) + Python core.py. Both removed.
    This test verifies the combination trains without raising (correctness validated
    separately by the G4a/G4b gate harness at docs/sprint33/commit2-gates/).

    covers: S28-ST-GUARD removal, DEC-042 closure, Commit 3a (#93)
    """
    from catboost_mlx import _core

    cfg = _make_minimal_config(grow_policy="SymmetricTree", score_function="Cosine")
    X, y, w, g, val_X, val_y, fn, ic, chm = _minimal_arrays()

    # Must not raise — ST+Cosine is now a supported combination
    result = _core.train(
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
    assert result is not None, "Expected a non-None result from ST+Cosine train"


# ---------------------------------------------------------------------------
# CLI subprocess tests
# ---------------------------------------------------------------------------


@_SKIP_CLI
def test_csv_train_cli_accepts_cosine_lossguide():
    """csv_train CLI must exit 0 for score_function='Cosine' + grow_policy='Lossguide'.

    S28-LG-GUARD removed in S33-L4-FIX Commit 3b (#94).  The old guard fired in
    ParseArgs after flag parsing, before file open, and terminated with non-zero exit.
    After guard removal the combination must train successfully.

    Uses a tiny 20-row CSV written to a temp file so the binary sees real data.
    The binary at python/catboost_mlx/bin/csv_train is the production binary
    (post-Commit-3b rebuild).

    LG requires --max-leaves to be set (default 31 is used here).

    covers: S28-LG-GUARD removal from CLI path (csv_train.cpp), DEC-042 Commit 3b
    """
    import csv
    import tempfile

    rng = np.random.default_rng(42)
    X = rng.standard_normal((20, 3)).astype(np.float32)
    y = (X[:, 0] * 0.5 + X[:, 1] * 0.3).astype(np.float32)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        tmp_path = f.name
        writer = csv.writer(f)
        writer.writerow(["f0", "f1", "f2", "target"])
        for i in range(20):
            writer.writerow([X[i, 0], X[i, 1], X[i, 2], y[i]])

    try:
        result = subprocess.run(
            [
                str(_CSV_TRAIN_BIN),
                tmp_path,
                "--iterations", "2",
                "--depth", "4",
                "--bins", "16",
                "--grow-policy", "Lossguide",
                "--max-leaves", "8",
                "--score-function", "Cosine",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
    finally:
        import os
        os.unlink(tmp_path)

    # covers: S28-LG-GUARD removal, CLI LG+Cosine now succeeds
    assert result.returncode == 0, (
        f"csv_train should exit 0 for Cosine+Lossguide after guard removal, "
        f"got returncode={result.returncode}\nstderr: {result.stderr[:500]}"
    )
    assert "TODO-S29-LG-COSINE-RCA" not in result.stderr, (
        f"Old guard error still present in stderr — guard not fully removed:\n"
        f"{result.stderr[:500]}"
    )


@_SKIP_CLI
def test_csv_train_cli_accepts_cosine_symmetric_tree():
    """csv_train CLI must exit 0 for score_function='Cosine' + grow_policy='SymmetricTree'.

    S28-ST-GUARD removed in S33-L4-FIX Commit 3a (#93).  The old guard fired in
    ParseArgs after flag parsing, before file open, and terminated with non-zero exit.
    After guard removal the combination must train successfully.

    Uses a tiny 20-row CSV written to a temp file so the binary sees real data.
    The binary at python/catboost_mlx/bin/csv_train is the production binary
    (post-#82 build, includes --score-function flag).

    covers: S28-ST-GUARD removal from CLI path (csv_train.cpp), DEC-042 Commit 3a
    """
    import csv
    import tempfile

    rng = np.random.default_rng(42)
    X = rng.standard_normal((20, 3)).astype(np.float32)
    y = (X[:, 0] * 0.5 + X[:, 1] * 0.3).astype(np.float32)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        tmp_path = f.name
        writer = csv.writer(f)
        writer.writerow(["f0", "f1", "f2", "target"])
        for i in range(20):
            writer.writerow([X[i, 0], X[i, 1], X[i, 2], y[i]])

    try:
        result = subprocess.run(
            [
                str(_CSV_TRAIN_BIN),
                tmp_path,
                "--iterations", "2",
                "--depth", "2",
                "--bins", "16",
                "--grow-policy", "SymmetricTree",
                "--score-function", "Cosine",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
    finally:
        import os
        os.unlink(tmp_path)

    # covers: S28-ST-GUARD removal, CLI ST+Cosine now succeeds
    assert result.returncode == 0, (
        f"csv_train should exit 0 for Cosine+SymmetricTree after guard removal, "
        f"got returncode={result.returncode}\nstderr: {result.stderr[:500]}"
    )
    assert "TODO-S29-ST-COSINE-KAHAN" not in result.stderr, (
        f"Old guard error still present in stderr — guard not fully removed:\n"
        f"{result.stderr[:500]}"
    )
