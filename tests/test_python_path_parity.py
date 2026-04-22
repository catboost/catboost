"""
S26-D0 G3 regression harness for Python-path parity.

Catches the bug class where MLX predictions are systematically shrunken vs CPU
due to any regression in FindBestSplit, RandomStrength scaling, basePred
computation, or nanobind orchestration. bench_boosting's ULP=0 record covers
histogram kernel only — this test catches Python-path regressions bench_boosting
cannot see.

See DEC-028 for the original bug: the RandomStrength noise formula used
`rs * N` instead of `rs * gradRms`, producing noise ~16895x too large that
effectively randomized all split candidates and collapsed leaf values to near-zero,
yielding predictions with ~0.69x the correct standard deviation.

The bug class signature:
  - MLX train RMSE converges monotonically (training "works")
  - MLX predictions have correct directionality (Pearson > 0.9 with CPU)
  - MLX pred std is ~0.69x CPU pred std (systematic leaf shrinkage)
  - Final RMSE delta: MLX ~68% worse than CPU (0.338 vs 0.201 at N=10k, seed=1337)

Tolerance 5% (vs G1 gate 2%): CI test must tolerate stochastic variation and
machine-to-machine float32 differences; 5% catches the DEC-028 class (68% delta)
with an enormous margin while avoiding flakes.
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
    """Run CPU CatBoost; return final train RMSE from evals_result_."""
    from catboost import CatBoostRegressor

    m = CatBoostRegressor(
        iterations=50,
        depth=6,
        learning_rate=0.03,
        loss_function="RMSE",
        grow_policy="SymmetricTree",
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
    """
    from catboost_mlx import CatBoostMLXRegressor

    m = CatBoostMLXRegressor(
        iterations=50,
        depth=6,
        learning_rate=0.03,
        loss="rmse",
        grow_policy="SymmetricTree",
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
        loss_function="RMSE", grow_policy="SymmetricTree", max_bin=128,
        random_seed=seed, random_strength=1.0,
        bootstrap_type="No", verbose=0, thread_count=1,
    )
    cpu.fit(X, y)
    cpu_preds = np.asarray(cpu.predict(X), dtype=np.float64)

    mlx = CatBoostMLXRegressor(
        iterations=50, depth=6, learning_rate=0.03,
        loss="rmse", grow_policy="SymmetricTree", bins=128,
        random_seed=seed, random_strength=1.0,
        bootstrap_type="no", verbose=False,
    )
    mlx.fit(X, y)
    mlx_preds = np.asarray(mlx.predict(X), dtype=np.float64)

    cpu_std = float(np.std(cpu_preds))
    mlx_std = float(np.std(mlx_preds))

    assert cpu_std > 0, "CPU predictions are constant — training failed"
    std_ratio = mlx_std / cpu_std

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
        loss="rmse", grow_policy="SymmetricTree", bins=128,
        random_seed=seed, random_strength=1.0,
        bootstrap_type="no", verbose=False,
    )
    m.fit(X, y)
    hist = m._train_loss_history

    assert len(hist) > 0, "train_loss_history is empty — nanobind path did not populate it"

    first_loss = hist[0]
    last_loss = hist[-1]
    assert last_loss < first_loss, (
        f"MLX train loss did not decrease: first={first_loss:.6f}, last={last_loss:.6f}. "
        f"Training is not converging — suggests noise or leaf bug prevents any split gain."
    )

    # Count non-monotone steps (loss[i] > loss[i-1])
    hist_arr = np.array(hist, dtype=np.float64)
    non_mono = int(np.sum(hist_arr[1:] > hist_arr[:-1]))
    total_steps = len(hist_arr) - 1
    non_mono_frac = non_mono / total_steps if total_steps > 0 else 0.0

    assert non_mono_frac <= 0.05, (
        f"MLX train loss has {non_mono}/{total_steps} non-monotone steps "
        f"({non_mono_frac*100:.1f}% > 5% tolerance). "
        f"Expected near-monotone convergence with random_strength=1.0."
    )
