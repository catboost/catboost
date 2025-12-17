import numpy as np
import pytest


def _require_cuda():
    try:
        from catboost import utils
    except Exception as e:
        pytest.skip(f"catboost utils not available: {e}")
    try:
        devs = utils.get_gpu_device_count()
    except Exception as e:
        pytest.skip(f"CUDA not available in catboost: {e}")
    if devs < 1:
        pytest.skip("CUDA device not available")
    return devs


def _require_cupy():
    cp = pytest.importorskip("cupy")
    try:
        _ = cp.cuda.runtime.getDeviceCount()
    except Exception as e:
        pytest.skip(f"cupy CUDA runtime unavailable: {e}")
    return cp


def _require_cudf():
    return pytest.importorskip("cudf")


def test_cupy_input_regression_parity():
    _require_cuda()
    cp = _require_cupy()

    from catboost import CatBoostRegressor

    rs = np.random.RandomState(0)
    n = 2000
    f = 10
    x_cpu = rs.rand(n, f).astype(np.float32)
    y_cpu = (x_cpu[:, 0] * 0.3 + x_cpu[:, 1] * -0.2 + 0.1).astype(np.float32)

    x_gpu = cp.asarray(x_cpu)
    y_gpu = cp.asarray(y_cpu)

    params = dict(
        iterations=20,
        depth=6,
        learning_rate=0.1,
        loss_function="RMSE",
        task_type="GPU",
        devices="0",
        random_seed=0,
        verbose=False,
    )

    model_cpu_input = CatBoostRegressor(**params)
    model_cpu_input.fit(x_cpu, y_cpu)

    model_gpu_input = CatBoostRegressor(**params)
    model_gpu_input.fit(x_gpu, y_gpu)

    pred_cpu = model_cpu_input.predict(x_cpu[:200])
    pred_gpu = model_gpu_input.predict(x_cpu[:200])

    assert np.max(np.abs(pred_cpu - pred_gpu)) < 2e-2


def test_cudf_input_cat_ctr_smoke():
    _require_cuda()
    cp = _require_cupy()
    cudf = _require_cudf()

    from catboost import CatBoostClassifier

    cp.random.seed(0)
    n = 5000
    cat = cp.random.randint(0, 1000, size=n, dtype=cp.int32)
    num = cp.random.random(n, dtype=cp.float32)
    logits = (cat % 2).astype(cp.float32) * 1.5 + (num - 0.5) * 2
    p = 1 / (1 + cp.exp(-logits))
    y = (cp.random.random(n, dtype=cp.float32) < p).astype(cp.int32)

    x_df = cudf.DataFrame({"cat0": cat, "num0": num})
    y_s = cudf.Series(y)

    model = CatBoostClassifier(
        iterations=10,
        depth=6,
        learning_rate=0.1,
        loss_function="Logloss",
        task_type="GPU",
        devices="0",
        one_hot_max_size=2,
        max_ctr_complexity=2,
        verbose=False,
        random_seed=0,
    )
    model.fit(x_df, y_s, cat_features=[0])

    # Prediction API does not accept cuDF directly; use host representation here.
    proba = model.predict_proba(x_df.to_pandas())
    assert proba.shape == (n, 2)


def test_gpu_sample_weight_affects_training():
    _require_cuda()
    cp = _require_cupy()

    from catboost import CatBoostRegressor

    cp.random.seed(0)
    n = 5000
    f = 5
    x = cp.random.random((n, f), dtype=cp.float32)
    y = (x[:, 0] * 2.0 - x[:, 1] * 1.0 + 0.5).astype(cp.float32)
    w = cp.ones(n, dtype=cp.float32)
    w[: n // 2] = 0.0

    params = dict(
        iterations=30,
        depth=6,
        learning_rate=0.1,
        loss_function="RMSE",
        task_type="GPU",
        devices="0",
        verbose=False,
        random_seed=0,
    )

    m_unweighted = CatBoostRegressor(**params)
    m_unweighted.fit(x, y)

    m_weighted = CatBoostRegressor(**params)
    m_weighted.fit(x, y, sample_weight=w)

    x_cpu = cp.asnumpy(x[:200])
    pred_unweighted = m_unweighted.predict(x_cpu)
    pred_weighted = m_weighted.predict(x_cpu)

    assert np.max(np.abs(pred_unweighted - pred_weighted)) > 1e-3


def test_memcpy_tracker_strict_mode_enforced(monkeypatch):
    _require_cuda()
    _require_cupy()

    import catboost._catboost as cb

    monkeypatch.setenv("CATBOOST_CUDA_STRICT_NO_D2H", "1")
    monkeypatch.delenv("CATBOOST_CUDA_D2H_BYTES_LIMIT", raising=False)
    monkeypatch.delenv("CATBOOST_CUDA_D2H_SINGLE_BYTES_LIMIT", raising=False)

    cb._cuda_memcpy_tracker_reset_config()
    cb._cuda_memcpy_tracker_reset_stats()
    try:
        with pytest.raises(cb.CatBoostError):
            cb._cuda_memcpy_tracker_test_device_to_host(4)
    finally:
        cb._cuda_memcpy_tracker_reset_config()
        cb._cuda_memcpy_tracker_reset_stats()


def test_strict_no_d2h_limits_allow_training(monkeypatch):
    _require_cuda()
    cp = _require_cupy()

    import catboost._catboost as cb
    from catboost import CatBoostRegressor

    monkeypatch.setenv("CATBOOST_CUDA_STRICT_NO_D2H", "1")
    monkeypatch.setenv("CATBOOST_CUDA_D2H_SINGLE_BYTES_LIMIT", "262144")  # 256 KiB
    monkeypatch.setenv("CATBOOST_CUDA_D2H_BYTES_LIMIT", "1048576")  # 1 MiB

    cb._cuda_memcpy_tracker_reset_config()
    cb._cuda_memcpy_tracker_reset_stats()

    try:
        cp.random.seed(0)
        n = 100000  # ~4 MiB features matrix; dataset-sized D2H would violate limits
        f = 10
        x = cp.random.random((n, f), dtype=cp.float32)
        y = (x[:, 0] * 0.3 + x[:, 1] * -0.2 + 0.1).astype(cp.float32)

        model = CatBoostRegressor(
            iterations=5,
            depth=6,
            learning_rate=0.1,
            loss_function="RMSE",
            task_type="GPU",
            devices="0",
            verbose=False,
            random_seed=0,
        )
        model.fit(x, y)

        stats = cb._cuda_memcpy_tracker_get_stats()
        assert stats["device_to_host_bytes"] <= 1048576
    finally:
        cb._cuda_memcpy_tracker_reset_config()
        cb._cuda_memcpy_tracker_reset_stats()


def test_cupy_input_multigpu_smoke():
    devs = _require_cuda()
    if devs < 2:
        pytest.skip("Need >=2 GPUs for multi-GPU smoke test")
    cp = _require_cupy()

    from catboost import CatBoostRegressor

    cp.random.seed(0)
    n = 20000
    f = 10
    x = cp.random.random((n, f), dtype=cp.float32)
    y = (x[:, 0] * 0.3 + x[:, 1] * -0.2 + 0.1).astype(cp.float32)

    model = CatBoostRegressor(
        iterations=5,
        depth=6,
        learning_rate=0.1,
        loss_function="RMSE",
        task_type="GPU",
        devices="0,1",
        verbose=False,
        random_seed=0,
    )
    model.fit(x, y)
