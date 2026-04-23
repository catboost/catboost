"""
S26-D0 G3 / S26-FU-2 regression harness for Python-path parity.

Catches the bug class where MLX predictions are systematically shrunken vs CPU
due to any regression in FindBestSplit, RandomStrength scaling, basePred
computation, or nanobind orchestration. bench_boosting's ULP=0 record covers
histogram kernel only — this test catches Python-path regressions bench_boosting
cannot see.

See DEC-028 for the original SymmetricTree bug: the RandomStrength noise formula
used `rs * N` instead of `rs * gradRms`, producing noise ~16895x too large that
effectively randomized all split candidates and collapsed leaf values to near-zero,
yielding predictions with ~0.69x the correct standard deviation.

See DEC-029 for the Depthwise/Lossguide SplitProps bug: `TTreeRecord.SplitProps`
was never populated for non-oblivious trees, so `WriteModelJSON` emitted
`"splits": []` for every Depthwise/Lossguide tree.  Every sample was assigned to
leaf 0, producing constant predictions (pred_std_R ≈ 0, RMSE delta 560–598%).

See S26-FU-2 for the Depthwise/Lossguide RandomStrength fix: `FindBestSplitPerPartition`
had no noise path at all, causing ~10–12% RMSE under-fit vs CPU at rs=1 for
Depthwise/Lossguide even after DEC-029 was resolved.

SymmetricTree bug class signature (DEC-028):
  - MLX train RMSE converges monotonically (training "works")
  - MLX predictions have correct directionality (Pearson > 0.9 with CPU)
  - MLX pred std is ~0.69x CPU pred std (systematic leaf shrinkage)
  - Final RMSE delta: MLX ~68% worse than CPU (0.338 vs 0.201 at N=10k, seed=1337)

Depthwise/Lossguide DEC-029 bug class signature:
  - pred_std_R ≈ 0 (all samples routed to leaf 0 → constant prediction)
  - RMSE delta 560–598% (constant prediction ≈ mean of y ≈ 0, residuals are O(0.2))
  - Training loss history still converges (internal cursor unaffected)

Depthwise/Lossguide FU-2 bug class signature (post-DEC-029, pre-FU-2):
  - pred_std_R near 1.0 (DEC-029 resolved, leaf routing correct)
  - rs=0: RMSE parity within ±5% (deterministic path fine)
  - rs=1: MLX RMSE 10–12% worse than CPU (missing noise in FindBestSplitPerPartition
    causes all partition splits to be evaluated without perturbation, so the
    per-partition split selection diverges from CPU's noisy version)

Tolerance 5% (vs G1 gate 2%): CI test must tolerate stochastic variation and
machine-to-machine float32 differences; 5% catches the DEC-028 class (68% delta)
with an enormous margin while avoiding flakes.

Gate segmentation (per S26-D0 G1 analysis):
  rs=0 (deterministic): symmetric ratio ∈ [0.95, 1.05] — no PRNG divergence.
  rs=1 (stochastic):    one-sided MLX_RMSE ≤ CPU_RMSE × 1.05 AND
                         pred_std_R ∈ [0.90, 1.10].
  The one-sided criterion avoids false-fails when MLX happens to be *better* than
  CPU due to independent RNG realization divergence (confirmed empirically in G1:
  all rs=1 SymmetricTree cells had MLX_RMSE < CPU_RMSE).
"""

import os

import numpy as np
import pytest

os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")


def _make_data(n: int, seed: int):
    """Canonical S26 data: 20 features, signal in f0 and f1, 10% noise."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 20)).astype(np.float32)
    y = (
        X[:, 0] * 0.5 + X[:, 1] * 0.3 + rng.standard_normal(n) * 0.1
    ).astype(np.float32)
    return X, y


def _cpu_rmse(X, y, seed: int, random_strength: float) -> float:
    """Run CPU CatBoost; return final train RMSE from evals_result_.

    score_function='L2' is explicit (DEC-031 Rule 3): CPU SymmetricTree default
    is L2, matching MLX default.  Stated explicitly so the gate tests the same
    algorithm on both sides regardless of future CPU default changes.
    """
    from catboost import CatBoostRegressor

    m = CatBoostRegressor(
        iterations=50,
        depth=6,
        learning_rate=0.03,
        loss_function="RMSE",
        grow_policy="SymmetricTree",
        score_function="L2",  # DEC-031 Rule 3: explicit, not implicit default
        max_bin=128,
        random_seed=seed,
        random_strength=random_strength,
        bootstrap_type="No",
        verbose=0,
        thread_count=1,
    )
    m.fit(X, y)
    return float(m.evals_result_["learn"]["RMSE"][-1])


def _mlx_rmse(X, y, seed: int, random_strength: float) -> float:
    """Run MLX CatBoost via nanobind path; return final train RMSE.

    Uses _train_loss_history[-1] when available (nanobind path populates it).
    Falls back to pred-based RMSE if history is empty (subprocess path).
    Both paths exercise the full Python -> C++ -> GPU pipeline.

    score_function='L2' is explicit (DEC-031 Rule 3): MLX default is L2.
    Stated explicitly so the test documents which algorithm is under test.
    """
    from catboost_mlx import CatBoostMLXRegressor

    m = CatBoostMLXRegressor(
        iterations=50,
        depth=6,
        learning_rate=0.03,
        loss="rmse",
        grow_policy="SymmetricTree",
        score_function="L2",  # DEC-031 Rule 3: explicit, not implicit default
        bins=128,
        random_seed=seed,
        random_strength=random_strength,
        bootstrap_type="no",
        verbose=False,
    )
    m.fit(X, y)

    if m._train_loss_history:
        return float(m._train_loss_history[-1])

    # Fallback: compute RMSE from predictions (subprocess path or nanobind with
    # empty history, which should not happen post-fix but is safe to handle).
    preds = m.predict(X)
    return float(
        np.sqrt(((np.asarray(preds, dtype=np.float64) - y.astype(np.float64)) ** 2).mean())
    )


@pytest.mark.parametrize("n,seed", [(10_000, 1337), (10_000, 42)])
@pytest.mark.parametrize("random_strength", [0.0, 1.0])
def test_symmetrictree_python_path_parity(n: int, seed: int, random_strength: float):
    """MLX Python-path RMSE must be within 5% of CPU CatBoost at identical config.

    Tolerance is looser than the G1 sprint gate (2%) because this is a CI
    regression test that tolerates stochastic variation across machines; the
    bug class we're catching (DEC-028) produced a 68% delta, so 5% catches it
    with large margin while avoiding flakes.

    Parametrization:
      - Two seeds (1337, 42): verifies fix is not seed-specific.
      - rs=0.0 (deterministic branch): no noise injection, split selection
        must be exactly the same formula as CPU.
      - rs=1.0 (stochastic branch): the rs>0 path is precisely where DEC-028
        lived; this is the highest-risk parameter combination.
    """
    X, y = _make_data(n, seed)
    cpu_rmse = _cpu_rmse(X, y, seed, random_strength)
    mlx_rmse = _mlx_rmse(X, y, seed, random_strength)

    ratio = mlx_rmse / cpu_rmse
    # covers: aggregate RMSE, Python-path end-to-end, score_function=L2,
    #         grow_policy=SymmetricTree (DEC-028 regression class)
    assert 0.95 <= ratio <= 1.05, (
        f"Python-path parity regression (DEC-028 class): "
        f"MLX/CPU RMSE ratio = {ratio:.4f} "
        f"(MLX={mlx_rmse:.6f}, CPU={cpu_rmse:.6f}) "
        f"at n={n}, seed={seed}, rs={random_strength}. "
        f"A ratio outside [0.95, 1.05] indicates systematic leaf shrinkage. "
        f"Investigate: FindBestSplit noise formula (csv_train.cpp), "
        f"RandomStrength scaling (DEC-028), basePred computation, "
        f"or nanobind parameter mapping in core.py::_fit_nanobind."
    )


@pytest.mark.parametrize("seed", [1337, 42])
def test_symmetrictree_pred_std_ratio(seed: int):
    """MLX prediction std must be within 10% of CPU prediction std.

    The DEC-028 bug produced MLX pred_std / CPU pred_std ≈ 0.69 (31% shrinkage).
    This test catches leaf-value scaling bugs that the RMSE test may miss at
    high noise settings where both RMSEs are dominated by irreducible noise.

    Uses rs=1.0 (the path where DEC-028 lived) and N=10k for sensitivity.
    """
    from catboost import CatBoostRegressor
    from catboost_mlx import CatBoostMLXRegressor

    n = 10_000
    X, y = _make_data(n, seed)

    cpu = CatBoostRegressor(
        iterations=50, depth=6, learning_rate=0.03,
        loss_function="RMSE", grow_policy="SymmetricTree",
        score_function="L2",  # DEC-031 Rule 3: explicit, not implicit default
        max_bin=128,
        random_seed=seed, random_strength=1.0,
        bootstrap_type="No", verbose=0, thread_count=1,
    )
    cpu.fit(X, y)
    cpu_preds = np.asarray(cpu.predict(X), dtype=np.float64)

    mlx = CatBoostMLXRegressor(
        iterations=50, depth=6, learning_rate=0.03,
        loss="rmse", grow_policy="SymmetricTree",
        score_function="L2",  # DEC-031 Rule 3: explicit, not implicit default
        bins=128,
        random_seed=seed, random_strength=1.0,
        bootstrap_type="no", verbose=False,
    )
    mlx.fit(X, y)
    mlx_preds = np.asarray(mlx.predict(X), dtype=np.float64)

    cpu_std = float(np.std(cpu_preds))
    mlx_std = float(np.std(mlx_preds))

    # covers: Python-path end-to-end, sanity — CPU must produce non-constant preds
    assert cpu_std > 0, "CPU predictions are constant — training failed"
    std_ratio = mlx_std / cpu_std

    # covers: prediction std ratio, Python-path end-to-end, score_function=L2,
    #         grow_policy=SymmetricTree (leaf-magnitude signal, DEC-028 class)
    assert 0.90 <= std_ratio <= 1.10, (
        f"MLX pred std ratio = {std_ratio:.4f} "
        f"(MLX_std={mlx_std:.4f}, CPU_std={cpu_std:.4f}) at seed={seed}. "
        f"DEC-028 produced std_ratio ≈ 0.69; a value outside [0.90, 1.10] "
        f"indicates residual leaf-magnitude shrinkage. "
        f"Investigate: leaf value Newton step denominator (hessian=1 vs hessian=2 confusion)."
    )


@pytest.mark.parametrize("seed", [1337, 42])
def test_symmetrictree_monotone_convergence(seed: int):
    """MLX training loss must be monotonically non-increasing over 50 iters.

    A DEC-028-class bug can cause training loss to stall (constant) or oscillate
    (if noise makes every split candidate equivalent). This test ensures the
    fix did not break the fundamental convergence guarantee.

    Tolerance: allows <5% of consecutive iter pairs to be non-monotone (single
    float32 rounding at early iters is acceptable).
    """
    from catboost_mlx import CatBoostMLXRegressor

    n = 10_000
    X, y = _make_data(n, seed)

    m = CatBoostMLXRegressor(
        iterations=50, depth=6, learning_rate=0.03,
        loss="rmse", grow_policy="SymmetricTree",
        score_function="L2",  # DEC-031 Rule 3: explicit, not implicit default
        bins=128,
        random_seed=seed, random_strength=1.0,
        bootstrap_type="no", verbose=False,
    )
    m.fit(X, y)
    hist = m._train_loss_history

    # covers: Python-path end-to-end, nanobind history population, score_function=L2,
    #         grow_policy=SymmetricTree
    assert len(hist) > 0, "train_loss_history is empty — nanobind path did not populate it"

    first_loss = hist[0]
    last_loss = hist[-1]
    # covers: convergence direction, Python-path end-to-end, score_function=L2,
    #         grow_policy=SymmetricTree
    assert last_loss < first_loss, (
        f"MLX train loss did not decrease: first={first_loss:.6f}, last={last_loss:.6f}. "
        f"Training is not converging — suggests noise or leaf bug prevents any split gain."
    )

    # Count non-monotone steps (loss[i] > loss[i-1])
    hist_arr = np.array(hist, dtype=np.float64)
    non_mono = int(np.sum(hist_arr[1:] > hist_arr[:-1]))
    total_steps = len(hist_arr) - 1
    non_mono_frac = non_mono / total_steps if total_steps > 0 else 0.0

    # covers: convergence monotonicity, Python-path end-to-end, score_function=L2,
    #         grow_policy=SymmetricTree
    assert non_mono_frac <= 0.05, (
        f"MLX train loss has {non_mono}/{total_steps} non-monotone steps "
        f"({non_mono_frac*100:.1f}% > 5% tolerance). "
        f"Expected near-monotone convergence with random_strength=1.0."
    )


# ── Non-oblivious parity block (S26-FU-2 / S27-FU-3) ───────────────────────
#
# Covers Depthwise and Lossguide grow policies under rs={0.0, 1.0}.
#
# Three orthogonal signals, mirroring the SymmetricTree block above:
#   1. test_nonoblivious_python_path_parity  — final train RMSE ratio
#   2. test_nonoblivious_pred_std_ratio      — prediction std ratio (leaf-magnitude)
#   3. test_nonoblivious_monotone_convergence — training loss convergence
#
# Bug classes caught:
#   DEC-029: SplitProps never populated → constant predictions → pred_std_R ≈ 0
#            RMSE delta 560–598%.  Caught by (1) and (2).
#   FU-2:    FindBestSplitPerPartition had no noise path → rs=1 ~10-12% RMSE
#            under-fit.  Caught by (1) with the one-sided rs=1 bound.
#
# S27-FU-3 / DEC-032 note:
#   S28-FU3-REVALIDATE (task #74) confirmed 5/5 DW cells pass with
#   score_function='Cosine' on BOTH sides (ratios [0.9950, 1.0160], seeds 42-46,
#   N=1000).  The DW force-L2 conditional is removed; DW cells now test Cosine
#   parity end-to-end, matching CPU's default.
#
#   LG with Cosine both sides fails 0/5 at ratios [1.1403, 1.1498] — a gap of
#   ~14% analogous to the pre-S28 DW gap.  LG cells retain force-L2 pending a
#   dedicated LG Cosine port (S29-FU-LG candidate).  See t5-gate-report.md for
#   evidence and proposed followup scope.
# ---------------------------------------------------------------------------


def _cpu_fit_nonoblivious(X, y, seed: int, random_strength: float, grow_policy: str):
    """Train CPU CatBoost with a non-oblivious grow policy; return fitted model.

    Lossguide uses max_leaves=31 (CPU default, matching MLX default).

    Depthwise uses score_function='Cosine' (CPU default, now matched by MLX
    after S28-L2-EXPLICIT + S28-FU3-REVALIDATE).  DEC-031 Rule 3: explicit.

    Lossguide uses score_function='L2' — CPU LG default is Cosine but MLX LG
    has a ~14% Cosine/L2 gap at N=1000 (0/5 seeds pass at [0.98, 1.02]).
    Force-L2 retained pending S29-FU-LG port.  DEC-031 Rule 3: explicit.
    """
    from catboost import CatBoostRegressor

    kwargs = dict(
        iterations=50,
        depth=6,
        learning_rate=0.03,
        loss_function="RMSE",
        grow_policy=grow_policy,
        max_bin=128,
        random_seed=seed,
        random_strength=random_strength,
        bootstrap_type="No",
        verbose=0,
        thread_count=1,
    )
    if grow_policy == "Lossguide":
        kwargs["max_leaves"] = 31
        # S28-FU3-REVALIDATE: LG Cosine gap confirmed 0/5 seeds in [0.98, 1.02]
        # (ratios ~1.14); force L2 to match MLX until S29-FU-LG closes the gap.
        # DEC-031 Rule 3: explicit algorithm label.
        kwargs["score_function"] = "L2"
    if grow_policy == "Depthwise":
        # S28-FU3-REVALIDATE: DW Cosine parity confirmed 5/5 in [0.98, 1.02].
        # Use CPU's natural default (Cosine) explicitly per DEC-031 Rule 3.
        kwargs["score_function"] = "Cosine"
    m = CatBoostRegressor(**kwargs)
    m.fit(X, y)
    return m


def _mlx_fit_nonoblivious(X, y, seed: int, random_strength: float, grow_policy: str):
    """Train MLX CatBoost with a non-oblivious grow policy; return fitted model.

    Lossguide uses max_leaves=31 (default), matching the CPU setup above.
    Both nanobind and subprocess paths populate _train_loss_history for all
    grow policies after DEC-029 + FU-2.

    Depthwise uses score_function='Cosine' (S28-FU3-REVALIDATE: 5/5 DW cells
    pass with Cosine both sides; DEC-031 Rule 3: explicit, matching CPU default).

    Lossguide uses score_function='L2' (MLX LG Cosine gap ~14% confirmed in
    S28-FU3-REVALIDATE; force-L2 retained pending S29-FU-LG; DEC-031 Rule 3).
    """
    from catboost_mlx import CatBoostMLXRegressor

    score_function = "Cosine" if grow_policy == "Depthwise" else "L2"
    m = CatBoostMLXRegressor(
        iterations=50,
        depth=6,
        learning_rate=0.03,
        loss="rmse",
        grow_policy=grow_policy,
        score_function=score_function,  # DEC-031 Rule 3: explicit per policy
        bins=128,
        random_seed=seed,
        random_strength=random_strength,
        bootstrap_type="no",
        verbose=False,
    )
    m.fit(X, y)
    return m


def _rmse_from_model(m_mlx, X, y) -> float:
    """Extract final train RMSE from an MLX model.

    Prefers _train_loss_history[-1] (nanobind path); falls back to
    pred-based RMSE computation (subprocess path).
    """
    if m_mlx._train_loss_history:
        return float(m_mlx._train_loss_history[-1])
    preds = m_mlx.predict(X)
    return float(
        np.sqrt(
            ((np.asarray(preds, dtype=np.float64) - y.astype(np.float64)) ** 2).mean()
        )
    )


def _assert_segmented_parity(
    mlx_rmse: float,
    cpu_rmse: float,
    random_strength: float,
    context: str,
) -> None:
    """Assert RMSE parity under the segmented per-branch criterion.

    rs=0 (deterministic): symmetric ratio ∈ [0.95, 1.05].
    rs=1 (stochastic):    one-sided MLX_RMSE ≤ CPU_RMSE × 1.05.

    The one-sided bound for rs=1 avoids false-fails when MLX is better than CPU
    (observed empirically: all G1 rs=1 SymmetricTree cells had MLX_RMSE < CPU_RMSE).
    The upper-bound still catches any DEC-028/FU-2 class regression (which produced
    10–68% degradation) with enormous margin.
    """
    ratio = mlx_rmse / cpu_rmse
    if random_strength == 0.0:
        # covers: aggregate RMSE, Python-path end-to-end,
        #         score_function=Cosine(DW)/L2(LG), non-oblivious grow policy,
        #         rs=0 deterministic branch
        assert 0.95 <= ratio <= 1.05, (
            f"Python-path parity regression ({context}): "
            f"MLX/CPU RMSE ratio = {ratio:.4f} "
            f"(MLX={mlx_rmse:.6f}, CPU={cpu_rmse:.6f}). "
            f"rs=0 (deterministic branch): ratio must be in [0.95, 1.05]. "
            f"A value outside this range indicates a non-stochastic algorithmic "
            f"divergence (histogram accumulation, split formula, leaf Newton step)."
        )
    else:
        # One-sided: MLX may be better; only fail if MLX is much worse than CPU.
        # covers: aggregate RMSE, Python-path end-to-end,
        #         score_function=Cosine(DW)/L2(LG), non-oblivious grow policy,
        #         rs=1 stochastic branch
        assert mlx_rmse <= cpu_rmse * 1.05, (
            f"Python-path parity regression ({context}): "
            f"MLX RMSE = {mlx_rmse:.6f} > CPU RMSE × 1.05 = {cpu_rmse * 1.05:.6f} "
            f"(ratio = {ratio:.4f}, CPU = {cpu_rmse:.6f}). "
            f"rs=1 one-sided bound: MLX must not be more than 5% worse than CPU. "
            f"FU-2 pre-fix produced ~10–12% degradation; DEC-028 pre-fix 68%. "
            f"Investigate: FindBestSplitPerPartition noise path (FU-2 / DEC-028), "
            f"gradRms threading, or nanobind parameter mapping."
        )


@pytest.mark.parametrize("seed", [1337, 42])
@pytest.mark.parametrize("random_strength", [0.0, 1.0])
@pytest.mark.parametrize("grow_policy", ["Depthwise", "Lossguide"])
def test_nonoblivious_python_path_parity(grow_policy: str, random_strength: float, seed: int):
    """MLX Python-path RMSE must be within 5% of CPU CatBoost for non-oblivious policies.

    Segmented per S26-D0 G1 analysis:
      - rs=0 (deterministic): symmetric ratio ∈ [0.95, 1.05].
      - rs=1 (stochastic):    one-sided MLX_RMSE ≤ CPU_RMSE × 1.05.

    Bug classes caught:
      DEC-029: SplitProps never populated → constant predictions → RMSE delta
               560–598%.  This would show ratio ≈ 6.6 under rs=0 and > 5.6 under
               rs=1, failing both branches of the criterion with 100× margin.
      FU-2:    Missing noise in FindBestSplitPerPartition → rs=1 MLX RMSE 10–12%
               worse than CPU.  Fails the one-sided rs=1 bound (MLX_RMSE > CPU × 1.05).

    Parametrization:
      - Two seeds (1337, 42): verifies fix is not seed-specific.
      - Two grow policies: Depthwise and Lossguide are structurally different
        (symmetric-shaped vs priority-queue-based) but share the same bug root.
      - rs=0.0: deterministic branch, no noise injection.
      - rs=1.0: stochastic branch, the path where FU-2 lived.
    """
    n = 10_000
    X, y = _make_data(n, seed)

    cpu_model = _cpu_fit_nonoblivious(X, y, seed, random_strength, grow_policy)
    cpu_rmse = float(cpu_model.evals_result_["learn"]["RMSE"][-1])

    mlx_model = _mlx_fit_nonoblivious(X, y, seed, random_strength, grow_policy)
    mlx_rmse = _rmse_from_model(mlx_model, X, y)

    _assert_segmented_parity(
        mlx_rmse, cpu_rmse, random_strength,
        f"grow_policy={grow_policy}, n={n}, seed={seed}, rs={random_strength}"
    )


@pytest.mark.parametrize("seed", [1337, 42])
@pytest.mark.parametrize("random_strength", [0.0, 1.0])
@pytest.mark.parametrize("grow_policy", ["Depthwise", "Lossguide"])
def test_nonoblivious_pred_std_ratio(grow_policy: str, random_strength: float, seed: int):
    """MLX prediction std must be within 10% of CPU prediction std for non-oblivious policies.

    This is the primary leaf-magnitude signal.  The DEC-029 bug produced
    pred_std_R ≈ 0 (all samples assigned to leaf 0 → constant predictions);
    DEC-028 on SymmetricTree produced pred_std_R ≈ 0.69 (31% leaf shrinkage).
    The FU-2 bug did not produce obvious std collapse (std was correct in magnitude)
    but the RMSE degradation manifested because wrong splits were chosen.

    Tested at both rs=0 and rs=1:
      - rs=0 is a strong signal: if DEC-029 regressed, pred_std_R ≈ 0.
      - rs=1 confirms the FU-2 noise path did not accidentally collapse std.

    The [0.90, 1.10] bound catches the DEC-029 class (pred_std_R ≈ 0) with
    extreme margin (>9× the 10% tolerance).
    """
    n = 10_000
    X, y = _make_data(n, seed)

    cpu_model = _cpu_fit_nonoblivious(X, y, seed, random_strength, grow_policy)
    cpu_preds = np.asarray(cpu_model.predict(X), dtype=np.float64)

    mlx_model = _mlx_fit_nonoblivious(X, y, seed, random_strength, grow_policy)
    mlx_preds = np.asarray(mlx_model.predict(X), dtype=np.float64)

    cpu_std = float(np.std(cpu_preds))
    mlx_std = float(np.std(mlx_preds))

    # covers: Python-path end-to-end, sanity — CPU must produce non-constant preds,
    #         score_function=Cosine(DW)/L2(LG), non-oblivious grow policy
    assert cpu_std > 0, (
        f"CPU predictions are constant (std=0) — CPU training failed. "
        f"grow_policy={grow_policy}, seed={seed}, rs={random_strength}"
    )

    std_ratio = mlx_std / cpu_std

    # covers: prediction std ratio, Python-path end-to-end,
    #         score_function=Cosine(DW)/L2(LG),
    #         non-oblivious grow policy (leaf-magnitude signal, DEC-029/DEC-028 class)
    assert 0.90 <= std_ratio <= 1.10, (
        f"MLX pred std ratio = {std_ratio:.4f} "
        f"(MLX_std={mlx_std:.4f}, CPU_std={cpu_std:.4f}) "
        f"at grow_policy={grow_policy}, seed={seed}, rs={random_strength}. "
        f"DEC-029 produced std_ratio ≈ 0 (all samples in leaf 0 → constant predictions). "
        f"DEC-028 produced std_ratio ≈ 0.69 for SymmetricTree (leaf shrinkage). "
        f"A value outside [0.90, 1.10] indicates leaf-magnitude collapse or shrinkage. "
        f"Investigate: SplitProps population in csv_train.cpp (DEC-029), "
        f"BFS dispatch in _predict_utils.py, or leaf Newton step denominator."
    )


@pytest.mark.parametrize("seed", [1337, 42])
@pytest.mark.parametrize("grow_policy", ["Depthwise", "Lossguide"])
def test_nonoblivious_monotone_convergence(grow_policy: str, seed: int):
    """MLX training loss must be monotonically non-increasing for non-oblivious policies.

    Tested at rs=1.0 (the highest-risk configuration where noise perturbation
    could cause oscillation) — matching the SymmetricTree version's precedent of
    testing one seed per grow policy at rs=1.

    A DEC-029-class or FU-2-class regression can cause:
      - Training loss to stall constant (internal cursor correct but leaf routing
        broken, so every iteration adds zero net improvement per partition).
      - Training loss to oscillate if noise disables all split candidates.

    Tolerance: <5% non-monotone consecutive steps.  The nanobind path must populate
    _train_loss_history; an empty history is itself a failure signal.
    """
    from catboost_mlx import CatBoostMLXRegressor

    n = 10_000
    X, y = _make_data(n, seed)

    # DW uses Cosine (S28-FU3-REVALIDATE closed gap); LG uses L2 (S29-FU-LG pending).
    score_function = "Cosine" if grow_policy == "Depthwise" else "L2"
    m = CatBoostMLXRegressor(
        iterations=50,
        depth=6,
        learning_rate=0.03,
        loss="rmse",
        grow_policy=grow_policy,
        score_function=score_function,  # DEC-031 Rule 3: explicit per policy
        bins=128,
        random_seed=seed,
        random_strength=1.0,
        bootstrap_type="no",
        verbose=False,
    )
    m.fit(X, y)
    hist = m._train_loss_history

    # covers: Python-path end-to-end, nanobind history population,
    #         score_function=Cosine(DW)/L2(LG), non-oblivious grow policy
    assert len(hist) > 0, (
        f"_train_loss_history is empty — nanobind path did not populate it. "
        f"grow_policy={grow_policy}, seed={seed}"
    )

    first_loss = hist[0]
    last_loss = hist[-1]
    # covers: convergence direction, Python-path end-to-end,
    #         score_function=Cosine(DW)/L2(LG), non-oblivious grow policy
    assert last_loss < first_loss, (
        f"MLX train loss did not decrease: first={first_loss:.6f}, last={last_loss:.6f}. "
        f"grow_policy={grow_policy}, seed={seed}. "
        f"Training is not converging — suggests noise or leaf-routing bug prevents "
        f"any split gain in non-oblivious partitions."
    )

    hist_arr = np.array(hist, dtype=np.float64)
    non_mono = int(np.sum(hist_arr[1:] > hist_arr[:-1]))
    total_steps = len(hist_arr) - 1
    non_mono_frac = non_mono / total_steps if total_steps > 0 else 0.0

    # covers: convergence monotonicity, Python-path end-to-end,
    #         score_function=Cosine(DW)/L2(LG), non-oblivious grow policy
    assert non_mono_frac <= 0.05, (
        f"MLX train loss has {non_mono}/{total_steps} non-monotone steps "
        f"({non_mono_frac*100:.1f}% > 5% tolerance). "
        f"grow_policy={grow_policy}, seed={seed}. "
        f"Expected near-monotone convergence with random_strength=1.0."
    )
