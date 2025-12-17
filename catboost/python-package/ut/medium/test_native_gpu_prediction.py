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


def test_output_type_requires_gpu_input():
    from catboost import CatBoostError, CatBoostRegressor

    rs = np.random.RandomState(0)
    x = rs.rand(200, 5).astype(np.float32)
    y = (x[:, 0] * 0.3 + x[:, 1] * -0.2 + 0.1).astype(np.float32)

    model = CatBoostRegressor(
        iterations=5,
        depth=4,
        learning_rate=0.1,
        loss_function="RMSE",
        verbose=False,
        random_seed=0,
    )
    model.fit(x, y)

    with pytest.raises(CatBoostError, match="output_type=.*supported only for GPU-resident inputs"):
        model.predict(x[:10], output_type="cupy")


def test_cupy_predict_output_cupy_strict_no_d2h(monkeypatch):
    _require_cuda()
    cp = _require_cupy()

    import catboost._catboost as cb
    from catboost import CatBoostRegressor

    rs = np.random.RandomState(0)
    n = 2000
    f = 10
    x_cpu = rs.rand(n, f).astype(np.float32)
    y_cpu = (x_cpu[:, 0] * 0.3 + x_cpu[:, 1] * -0.2 + 0.1).astype(np.float32)

    x_gpu = cp.asarray(x_cpu)
    y_gpu = cp.asarray(y_cpu)

    model = CatBoostRegressor(
        iterations=10,
        depth=6,
        learning_rate=0.1,
        loss_function="RMSE",
        task_type="GPU",
        devices="0",
        verbose=False,
        random_seed=0,
    )
    model.fit(x_gpu, y_gpu)

    monkeypatch.setenv("CATBOOST_CUDA_STRICT_NO_D2H", "1")
    monkeypatch.delenv("CATBOOST_CUDA_D2H_BYTES_LIMIT", raising=False)
    monkeypatch.delenv("CATBOOST_CUDA_D2H_SINGLE_BYTES_LIMIT", raising=False)
    cb._cuda_memcpy_tracker_reset_config()
    cb._cuda_memcpy_tracker_reset_stats()

    try:
        preds = model.predict(x_gpu[:200], task_type="GPU", output_type="cupy")
        assert isinstance(preds, cp.ndarray)
        assert preds.shape == (200,)

        stats = cb._cuda_memcpy_tracker_get_stats()
        assert stats["device_to_host_bytes"] == 0
    finally:
        cb._cuda_memcpy_tracker_reset_config()
        cb._cuda_memcpy_tracker_reset_stats()


def test_cupy_predict_output_dlpack():
    _require_cuda()
    cp = _require_cupy()

    from catboost import CatBoostRegressor

    rs = np.random.RandomState(0)
    n = 500
    f = 5
    x_cpu = rs.rand(n, f).astype(np.float32)
    y_cpu = (x_cpu[:, 0] * 0.5 + 0.1).astype(np.float32)

    x_gpu = cp.asarray(x_cpu)
    y_gpu = cp.asarray(y_cpu)

    model = CatBoostRegressor(
        iterations=5,
        depth=4,
        learning_rate=0.1,
        loss_function="RMSE",
        task_type="GPU",
        devices="0",
        verbose=False,
        random_seed=0,
    )
    model.fit(x_gpu, y_gpu)

    dlpack_capsule = model.predict(x_gpu[:10], task_type="GPU", output_type="dlpack")
    preds = cp.from_dlpack(dlpack_capsule)
    assert preds.shape == (10,)


def test_cupy_predict_parity_vs_cpu():
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

    model = CatBoostRegressor(
        iterations=20,
        depth=6,
        learning_rate=0.1,
        loss_function="RMSE",
        task_type="GPU",
        devices="0",
        verbose=False,
        random_seed=0,
    )
    model.fit(x_gpu, y_gpu)

    cpu_pred = model.predict(x_cpu[:200])
    gpu_pred = model.predict(x_gpu[:200], task_type="GPU", output_type="cupy")
    np.testing.assert_allclose(cpu_pred, cp.asnumpy(gpu_pred), rtol=1e-6, atol=1e-6)


def test_dlpack_input_predict_output_cupy():
    _require_cuda()
    cp = _require_cupy()

    from catboost import CatBoostRegressor

    class DLPackOnly:
        def __init__(self, arr):
            self._arr = arr

        @property
        def shape(self):
            return self._arr.shape

        def __dlpack__(self, stream=None):
            return self._arr.__dlpack__(stream=stream)

        def __dlpack_device__(self):
            return self._arr.__dlpack_device__()

    rs = np.random.RandomState(0)
    n = 500
    f = 5
    x_cpu = rs.rand(n, f).astype(np.float32)
    y_cpu = (x_cpu[:, 0] * 0.5 + 0.1).astype(np.float32)

    x_gpu = cp.asarray(x_cpu)
    y_gpu = cp.asarray(y_cpu)

    model = CatBoostRegressor(
        iterations=5,
        depth=4,
        learning_rate=0.1,
        loss_function="RMSE",
        task_type="GPU",
        devices="0",
        verbose=False,
        random_seed=0,
    )
    model.fit(x_gpu, y_gpu)

    dlpack_input = DLPackOnly(x_gpu[:10])
    preds = model.predict(dlpack_input, task_type="GPU", output_type="cupy")
    assert isinstance(preds, cp.ndarray)
    assert preds.shape == (10,)


def test_cudf_predict_output_cudf():
    _require_cuda()
    cp = _require_cupy()
    cudf = _require_cudf()

    from catboost import CatBoostClassifier

    rs = np.random.RandomState(0)
    n = 2000
    f = 5
    x_cpu = rs.rand(n, f).astype(np.float32)
    logits = (x_cpu[:, 0] * 1.5 - x_cpu[:, 1] * 1.0 + 0.2).astype(np.float32)
    y_cpu = (rs.rand(n) < (1.0 / (1.0 + np.exp(-logits)))).astype(np.int32)

    x_gpu = cp.asarray(x_cpu)
    y_gpu = cp.asarray(y_cpu)
    x_df = cudf.DataFrame({f"f{i}": x_gpu[:, i] for i in range(f)})
    y_s = cudf.Series(y_gpu)

    model = CatBoostClassifier(
        iterations=5,
        depth=6,
        learning_rate=0.1,
        loss_function="Logloss",
        task_type="GPU",
        devices="0",
        verbose=False,
        random_seed=0,
    )
    model.fit(x_df, y_s)

    proba = model.predict_proba(x_df.head(100), task_type="GPU", output_type="cudf")
    assert isinstance(proba, cudf.DataFrame)
    assert proba.shape == (100, 2)


def test_cudf_predict_onehot_cat_parity_vs_cpu():
    _require_cuda()
    cp = _require_cupy()
    cudf = _require_cudf()

    from catboost import CatBoostClassifier

    cp.random.seed(0)
    n = 5000
    cat = cp.random.randint(0, 3, size=n, dtype=cp.int32)
    num = cp.random.random(n, dtype=cp.float32)
    logits = (cat == 1).astype(cp.float32) * 2.0 + (num - 0.5) * 0.5
    p = 1.0 / (1.0 + cp.exp(-logits))
    y = (cp.random.random(n, dtype=cp.float32) < p).astype(cp.int32)

    x_df = cudf.DataFrame({"cat0": cat, "num0": num})
    y_s = cudf.Series(y)

    model = CatBoostClassifier(
        iterations=50,
        depth=6,
        learning_rate=0.1,
        loss_function="Logloss",
        task_type="GPU",
        devices="0",
        one_hot_max_size=10,
        max_ctr_complexity=0,
        verbose=False,
        random_seed=0,
    )
    model.fit(x_df, y_s, cat_features=[0])

    cpu_proba = model.predict_proba(x_df.to_pandas())
    gpu_proba = model.predict_proba(x_df, task_type="GPU", output_type="cupy")
    np.testing.assert_allclose(cpu_proba, cp.asnumpy(gpu_proba), rtol=1e-6, atol=1e-6)


def test_cudf_predict_onehot_cat_strict_no_d2h(monkeypatch):
    _require_cuda()
    cp = _require_cupy()
    cudf = _require_cudf()

    import catboost._catboost as cb
    from catboost import CatBoostClassifier

    cp.random.seed(0)
    n = 5000
    cat = cp.random.randint(0, 3, size=n, dtype=cp.int32)
    num = cp.random.random(n, dtype=cp.float32)
    logits = (cat == 1).astype(cp.float32) * 2.0 + (num - 0.5) * 0.5
    p = 1.0 / (1.0 + cp.exp(-logits))
    y = (cp.random.random(n, dtype=cp.float32) < p).astype(cp.int32)

    x_df = cudf.DataFrame({"cat0": cat, "num0": num})
    y_s = cudf.Series(y)

    model = CatBoostClassifier(
        iterations=20,
        depth=6,
        learning_rate=0.1,
        loss_function="Logloss",
        task_type="GPU",
        devices="0",
        one_hot_max_size=10,
        max_ctr_complexity=0,
        verbose=False,
        random_seed=0,
    )
    model.fit(x_df, y_s, cat_features=[0])

    monkeypatch.setenv("CATBOOST_CUDA_STRICT_NO_D2H", "1")
    monkeypatch.delenv("CATBOOST_CUDA_D2H_BYTES_LIMIT", raising=False)
    monkeypatch.delenv("CATBOOST_CUDA_D2H_SINGLE_BYTES_LIMIT", raising=False)
    cb._cuda_memcpy_tracker_reset_config()
    cb._cuda_memcpy_tracker_reset_stats()

    try:
        proba = model.predict_proba(x_df.head(200), task_type="GPU", output_type="cupy")
        assert isinstance(proba, cp.ndarray)
        assert proba.shape == (200, 2)
        stats = cb._cuda_memcpy_tracker_get_stats()
        assert stats["device_to_host_bytes"] == 0
    finally:
        cb._cuda_memcpy_tracker_reset_config()
        cb._cuda_memcpy_tracker_reset_stats()


def test_cudf_predict_ctr_parity_vs_cpu():
    _require_cuda()
    cp = _require_cupy()
    cudf = _require_cudf()

    from catboost import CatBoostClassifier

    cp.random.seed(0)
    n = 2000
    cat = cp.random.randint(0, 1000, size=n, dtype=cp.int32)
    y = (cat % 2).astype(cp.int32)

    x_df = cudf.DataFrame({"cat0": cat})
    y_s = cudf.Series(y)

    model = CatBoostClassifier(
        iterations=20,
        depth=4,
        learning_rate=0.1,
        loss_function="Logloss",
        task_type="GPU",
        devices="0",
        one_hot_max_size=0,
        max_ctr_complexity=2,
        verbose=False,
        random_seed=0,
    )
    model.fit(x_df, y_s, cat_features=[0])

    cpu_proba = model.predict_proba(x_df.to_pandas())
    gpu_proba = model.predict_proba(x_df, task_type="GPU", output_type="cupy")
    np.testing.assert_allclose(cpu_proba, cp.asnumpy(gpu_proba), rtol=1e-6, atol=1e-6)


def test_cudf_predict_ctr_strict_no_d2h(monkeypatch):
    _require_cuda()
    cp = _require_cupy()
    cudf = _require_cudf()

    import catboost._catboost as cb
    from catboost import CatBoostClassifier

    cp.random.seed(0)
    n = 2000
    cat = cp.random.randint(0, 1000, size=n, dtype=cp.int32)
    y = (cat % 2).astype(cp.int32)

    x_df = cudf.DataFrame({"cat0": cat})
    y_s = cudf.Series(y)

    model = CatBoostClassifier(
        iterations=10,
        depth=4,
        learning_rate=0.1,
        loss_function="Logloss",
        task_type="GPU",
        devices="0",
        one_hot_max_size=0,
        max_ctr_complexity=2,
        verbose=False,
        random_seed=0,
    )
    model.fit(x_df, y_s, cat_features=[0])

    monkeypatch.setenv("CATBOOST_CUDA_STRICT_NO_D2H", "1")
    monkeypatch.delenv("CATBOOST_CUDA_D2H_BYTES_LIMIT", raising=False)
    monkeypatch.delenv("CATBOOST_CUDA_D2H_SINGLE_BYTES_LIMIT", raising=False)
    cb._cuda_memcpy_tracker_reset_config()
    cb._cuda_memcpy_tracker_reset_stats()

    try:
        proba = model.predict_proba(x_df.head(200), task_type="GPU", output_type="cupy")
        assert isinstance(proba, cp.ndarray)
        assert proba.shape == (200, 2)
        stats = cb._cuda_memcpy_tracker_get_stats()
        assert stats["device_to_host_bytes"] == 0
    finally:
        cb._cuda_memcpy_tracker_reset_config()
        cb._cuda_memcpy_tracker_reset_stats()


def test_cudf_predict_categorical_string_onehot_parity_vs_cpu():
    _require_cuda()
    cp = _require_cupy()
    cudf = _require_cudf()

    pd = pytest.importorskip("pandas")

    from catboost import CatBoostClassifier

    rs = np.random.RandomState(0)
    n = 3000
    cats = rs.choice(["red", "green", "blue"], size=n).astype(object)
    num = rs.rand(n).astype(np.float32)
    y = ((cats == "green") | (num > 0.7)).astype(np.int32)

    x_cpu = pd.DataFrame({"cat0": pd.Series(cats, dtype="category"), "num0": num})
    x_gpu = cudf.DataFrame({"cat0": cudf.Series(cats).astype("category"), "num0": cp.asarray(num)})

    model = CatBoostClassifier(
        iterations=50,
        depth=6,
        learning_rate=0.1,
        loss_function="Logloss",
        one_hot_max_size=10,
        max_ctr_complexity=0,
        verbose=False,
        random_seed=0,
    )
    model.fit(x_cpu, y, cat_features=[0])

    cpu_proba = model.predict_proba(x_cpu.iloc[:200])
    gpu_proba = model.predict_proba(x_gpu.head(200), task_type="GPU", output_type="cupy")
    np.testing.assert_allclose(cpu_proba, cp.asnumpy(gpu_proba), rtol=1e-6, atol=1e-6)


def test_cudf_predict_categorical_string_onehot_strict_no_d2h(monkeypatch):
    _require_cuda()
    cp = _require_cupy()
    cudf = _require_cudf()

    pd = pytest.importorskip("pandas")

    import catboost._catboost as cb
    from catboost import CatBoostClassifier

    rs = np.random.RandomState(0)
    n = 3000
    cats = rs.choice(["red", "green", "blue"], size=n).astype(object)
    num = rs.rand(n).astype(np.float32)
    y = ((cats == "green") | (num > 0.7)).astype(np.int32)

    x_cpu = pd.DataFrame({"cat0": pd.Series(cats, dtype="category"), "num0": num})
    x_gpu = cudf.DataFrame({"cat0": cudf.Series(cats).astype("category"), "num0": cp.asarray(num)})

    model = CatBoostClassifier(
        iterations=30,
        depth=6,
        learning_rate=0.1,
        loss_function="Logloss",
        one_hot_max_size=10,
        max_ctr_complexity=0,
        verbose=False,
        random_seed=0,
    )
    model.fit(x_cpu, y, cat_features=[0])

    monkeypatch.setenv("CATBOOST_CUDA_STRICT_NO_D2H", "1")
    monkeypatch.delenv("CATBOOST_CUDA_D2H_BYTES_LIMIT", raising=False)
    monkeypatch.delenv("CATBOOST_CUDA_D2H_SINGLE_BYTES_LIMIT", raising=False)
    cb._cuda_memcpy_tracker_reset_config()
    cb._cuda_memcpy_tracker_reset_stats()

    try:
        proba = model.predict_proba(x_gpu.head(200), task_type="GPU", output_type="cupy")
        assert isinstance(proba, cp.ndarray)
        assert proba.shape == (200, 2)
        stats = cb._cuda_memcpy_tracker_get_stats()
        assert stats["device_to_host_bytes"] == 0
    finally:
        cb._cuda_memcpy_tracker_reset_config()
        cb._cuda_memcpy_tracker_reset_stats()


def test_cudf_series_categorical_predict_onehot_parity_vs_cpu():
    _require_cuda()
    cp = _require_cupy()
    cudf = _require_cudf()

    pd = pytest.importorskip("pandas")

    from catboost import CatBoostClassifier

    rs = np.random.RandomState(0)
    n = 3000
    cats = rs.choice(["red", "green", "blue"], size=n).astype(object)
    y = (cats == "green").astype(np.int32)

    x_cpu = pd.DataFrame({"cat0": pd.Series(cats, dtype="category")})
    model = CatBoostClassifier(
        iterations=50,
        depth=6,
        learning_rate=0.1,
        loss_function="Logloss",
        one_hot_max_size=10,
        max_ctr_complexity=0,
        verbose=False,
        random_seed=0,
    )
    model.fit(x_cpu, y, cat_features=[0])

    s_gpu = cudf.Series(cats).astype("category")

    cpu_proba = model.predict_proba(x_cpu.iloc[:200])
    gpu_proba = model.predict_proba(s_gpu.head(200), task_type="GPU", output_type="cupy")
    np.testing.assert_allclose(cpu_proba, cp.asnumpy(gpu_proba), rtol=1e-6, atol=1e-6)


def test_cupy_predict_multiclass_raw_parity_vs_cpu():
    _require_cuda()
    cp = _require_cupy()

    from catboost import CatBoostClassifier

    rs = np.random.RandomState(0)
    n = 2000
    f = 10
    x_cpu = rs.rand(n, f).astype(np.float32)
    w = np.array([[1.0, -0.3, 0.2], [-0.1, 0.7, -0.4], [0.3, 0.2, 0.9]], dtype=np.float32)
    logits = x_cpu[:, :3].dot(w)
    y_cpu = logits.argmax(axis=1).astype(np.int32)

    x_gpu = cp.asarray(x_cpu)
    y_gpu = cp.asarray(y_cpu)

    model = CatBoostClassifier(
        iterations=20,
        depth=6,
        learning_rate=0.1,
        loss_function="MultiClass",
        task_type="GPU",
        devices="0",
        verbose=False,
        random_seed=0,
    )
    model.fit(x_gpu, y_gpu)

    cpu_raw = model.predict(x_cpu[:200], prediction_type="RawFormulaVal")
    gpu_raw = model.predict(x_gpu[:200], prediction_type="RawFormulaVal", task_type="GPU", output_type="cupy")
    assert isinstance(gpu_raw, cp.ndarray)
    assert gpu_raw.shape == cpu_raw.shape
    np.testing.assert_allclose(cpu_raw, cp.asnumpy(gpu_raw), rtol=1e-6, atol=1e-6)


def test_cupy_predict_multiclass_proba_parity_vs_cpu():
    _require_cuda()
    cp = _require_cupy()

    from catboost import CatBoostClassifier

    rs = np.random.RandomState(0)
    n = 2000
    f = 10
    x_cpu = rs.rand(n, f).astype(np.float32)
    w = np.array([[1.0, -0.3, 0.2], [-0.1, 0.7, -0.4], [0.3, 0.2, 0.9]], dtype=np.float32)
    logits = x_cpu[:, :3].dot(w)
    y_cpu = logits.argmax(axis=1).astype(np.int32)

    x_gpu = cp.asarray(x_cpu)
    y_gpu = cp.asarray(y_cpu)

    model = CatBoostClassifier(
        iterations=20,
        depth=6,
        learning_rate=0.1,
        loss_function="MultiClass",
        task_type="GPU",
        devices="0",
        verbose=False,
        random_seed=0,
    )
    model.fit(x_gpu, y_gpu)

    cpu_proba = model.predict_proba(x_cpu[:200])
    gpu_proba = model.predict_proba(x_gpu[:200], task_type="GPU", output_type="cupy")
    assert isinstance(gpu_proba, cp.ndarray)
    assert gpu_proba.shape == cpu_proba.shape
    np.testing.assert_allclose(cpu_proba, cp.asnumpy(gpu_proba), rtol=1e-6, atol=1e-6)


def test_cupy_predict_multiclass_log_proba_parity_vs_cpu():
    _require_cuda()
    cp = _require_cupy()

    from catboost import CatBoostClassifier

    rs = np.random.RandomState(0)
    n = 2000
    f = 10
    x_cpu = rs.rand(n, f).astype(np.float32)
    w = np.array([[1.0, -0.3, 0.2], [-0.1, 0.7, -0.4], [0.3, 0.2, 0.9]], dtype=np.float32)
    logits = x_cpu[:, :3].dot(w)
    y_cpu = logits.argmax(axis=1).astype(np.int32)

    x_gpu = cp.asarray(x_cpu)
    y_gpu = cp.asarray(y_cpu)

    model = CatBoostClassifier(
        iterations=20,
        depth=6,
        learning_rate=0.1,
        loss_function="MultiClass",
        task_type="GPU",
        devices="0",
        verbose=False,
        random_seed=0,
    )
    model.fit(x_gpu, y_gpu)

    cpu_log_proba = model.predict(x_cpu[:200], prediction_type="LogProbability")
    gpu_log_proba = model.predict(x_gpu[:200], prediction_type="LogProbability", task_type="GPU", output_type="cupy")
    assert isinstance(gpu_log_proba, cp.ndarray)
    assert gpu_log_proba.shape == cpu_log_proba.shape
    np.testing.assert_allclose(cpu_log_proba, cp.asnumpy(gpu_log_proba), rtol=1e-6, atol=1e-6)


def test_cupy_predict_multiclass_class_parity_vs_cpu():
    _require_cuda()
    cp = _require_cupy()

    from catboost import CatBoostClassifier

    rs = np.random.RandomState(0)
    n = 2000
    f = 10
    x_cpu = rs.rand(n, f).astype(np.float32)
    w = np.array([[1.0, -0.3, 0.2], [-0.1, 0.7, -0.4], [0.3, 0.2, 0.9]], dtype=np.float32)
    logits = x_cpu[:, :3].dot(w)
    y_cpu = logits.argmax(axis=1).astype(np.int32)

    x_gpu = cp.asarray(x_cpu)
    y_gpu = cp.asarray(y_cpu)

    model = CatBoostClassifier(
        iterations=20,
        depth=6,
        learning_rate=0.1,
        loss_function="MultiClass",
        task_type="GPU",
        devices="0",
        verbose=False,
        random_seed=0,
    )
    model.fit(x_gpu, y_gpu)

    cpu_class = model.predict(x_cpu[:200], prediction_type="Class").reshape(-1)
    gpu_class = model.predict(x_gpu[:200], prediction_type="Class", task_type="GPU", output_type="cupy")
    assert isinstance(gpu_class, cp.ndarray)
    np.testing.assert_array_equal(cpu_class, cp.asnumpy(gpu_class).reshape(-1))


def test_cupy_predict_rmse_with_uncertainty_parity_vs_cpu():
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

    model = CatBoostRegressor(
        iterations=20,
        depth=6,
        learning_rate=0.1,
        loss_function="RMSEWithUncertainty",
        task_type="GPU",
        devices="0",
        verbose=False,
        random_seed=0,
    )
    model.fit(x_gpu, y_gpu)

    cpu_pred = model.predict(x_cpu[:200], prediction_type="RMSEWithUncertainty")
    gpu_pred = model.predict(x_gpu[:200], prediction_type="RMSEWithUncertainty", task_type="GPU", output_type="cupy")
    assert isinstance(gpu_pred, cp.ndarray)
    assert gpu_pred.shape == cpu_pred.shape
    np.testing.assert_allclose(cpu_pred, cp.asnumpy(gpu_pred), rtol=1e-6, atol=1e-6)


def test_cupy_predict_exponent_parity_vs_cpu():
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

    model = CatBoostRegressor(
        iterations=20,
        depth=6,
        learning_rate=0.1,
        loss_function="Poisson",
        task_type="GPU",
        devices="0",
        verbose=False,
        random_seed=0,
    )
    model.fit(x_gpu, y_gpu)

    cpu_pred = model.predict(x_cpu[:200], prediction_type="Exponent")
    gpu_pred = model.predict(x_gpu[:200], prediction_type="Exponent", task_type="GPU", output_type="cupy")
    assert isinstance(gpu_pred, cp.ndarray)
    assert gpu_pred.shape == cpu_pred.shape
    np.testing.assert_allclose(cpu_pred, cp.asnumpy(gpu_pred), rtol=1e-6, atol=1e-6)


def test_cupy_predict_fortran_order_parity_vs_cpu():
    _require_cuda()
    cp = _require_cupy()

    from catboost import CatBoostRegressor

    rs = np.random.RandomState(0)
    n = 2000
    f = 16
    x_cpu = rs.rand(n, f).astype(np.float32)
    y_cpu = (x_cpu[:, 0] * 0.3 + x_cpu[:, 1] * -0.2 + 0.1).astype(np.float32)

    model = CatBoostRegressor(
        iterations=20,
        depth=6,
        learning_rate=0.1,
        loss_function="RMSE",
        verbose=False,
        random_seed=0,
    )
    model.fit(x_cpu, y_cpu)

    x_gpu_f = cp.array(x_cpu, order="F")
    cpu_pred = model.predict(x_cpu[:200])
    gpu_pred = model.predict(x_gpu_f[:200], task_type="GPU", output_type="cupy")
    np.testing.assert_allclose(cpu_pred, cp.asnumpy(gpu_pred), rtol=1e-6, atol=1e-6)


def test_cupy_predict_row_strided_parity_vs_cpu():
    _require_cuda()
    cp = _require_cupy()

    from catboost import CatBoostRegressor

    rs = np.random.RandomState(0)
    n = 4000
    f = 16
    x_cpu = rs.rand(n, f).astype(np.float32)
    y_cpu = (x_cpu[:, 0] * 0.3 + x_cpu[:, 1] * -0.2 + 0.1).astype(np.float32)

    model = CatBoostRegressor(
        iterations=30,
        depth=6,
        learning_rate=0.1,
        loss_function="RMSE",
        verbose=False,
        random_seed=0,
    )
    model.fit(x_cpu, y_cpu)

    base = cp.asarray(x_cpu)
    x_gpu = base[::2, :]  # non-contiguous, larger row stride
    cpu_pred = model.predict(x_cpu[::2, :][:200])
    gpu_pred = model.predict(x_gpu[:200], task_type="GPU", output_type="cupy")
    np.testing.assert_allclose(cpu_pred, cp.asnumpy(gpu_pred), rtol=1e-6, atol=1e-6)


def test_cupy_staged_predict_parity_vs_cpu():
    _require_cuda()
    cp = _require_cupy()

    from catboost import CatBoostRegressor

    rs = np.random.RandomState(0)
    n = 1000
    f = 8
    x_cpu = rs.rand(n, f).astype(np.float32)
    y_cpu = (x_cpu[:, 0] * 0.3 + x_cpu[:, 1] * -0.2 + 0.1).astype(np.float32)

    model = CatBoostRegressor(
        iterations=25,
        depth=6,
        learning_rate=0.1,
        loss_function="RMSE",
        verbose=False,
        random_seed=0,
    )
    model.fit(x_cpu, y_cpu)

    eval_period = 5
    cpu_stages = list(model.staged_predict(x_cpu[:200], prediction_type="RawFormulaVal", eval_period=eval_period))

    x_gpu = cp.asarray(x_cpu)
    gpu_stages = list(model.staged_predict(x_gpu[:200], prediction_type="RawFormulaVal", eval_period=eval_period))

    assert len(cpu_stages) == len(gpu_stages)
    for cpu_pred, gpu_pred in zip(cpu_stages, gpu_stages):
        np.testing.assert_allclose(cpu_pred, gpu_pred, rtol=1e-6, atol=1e-6)


def test_cupy_calc_leaf_indexes_parity_vs_cpu():
    _require_cuda()
    cp = _require_cupy()

    from catboost import CatBoostRegressor

    rs = np.random.RandomState(0)
    n = 1000
    f = 8
    x_cpu = rs.rand(n, f).astype(np.float32)
    y_cpu = (x_cpu[:, 0] * 0.3 + x_cpu[:, 1] * -0.2 + 0.1).astype(np.float32)

    model = CatBoostRegressor(
        iterations=20,
        depth=6,
        learning_rate=0.1,
        loss_function="RMSE",
        verbose=False,
        random_seed=0,
    )
    model.fit(x_cpu, y_cpu)

    cpu_leaf = model.calc_leaf_indexes(x_cpu[:200])

    x_gpu = cp.asarray(x_cpu)
    gpu_leaf = model.calc_leaf_indexes(x_gpu[:200])

    assert cpu_leaf.shape == gpu_leaf.shape
    assert cpu_leaf.dtype == gpu_leaf.dtype
    np.testing.assert_array_equal(cpu_leaf, gpu_leaf)


def test_multi_gpu_predict_devices_sharding_parity():
    devs = _require_cuda()
    if devs < 2:
        pytest.skip("Need >=2 CUDA devices for multi-GPU prediction test")
    cp = _require_cupy()

    can01 = cp.cuda.runtime.deviceCanAccessPeer(0, 1)
    can10 = cp.cuda.runtime.deviceCanAccessPeer(1, 0)
    if not (can01 and can10):
        pytest.skip("P2P access between GPU0 and GPU1 is required")

    from catboost import CatBoostRegressor

    rs = np.random.RandomState(0)
    n = 4000
    f = 32
    x_cpu = rs.rand(n, f).astype(np.float32)
    y_cpu = (x_cpu[:, 0] * 0.3 + x_cpu[:, 1] * -0.2 + 0.1).astype(np.float32)

    with cp.cuda.Device(0):
        x_gpu = cp.asarray(x_cpu)
        y_gpu = cp.asarray(y_cpu)

        model = CatBoostRegressor(
            iterations=30,
            depth=6,
            learning_rate=0.1,
            loss_function="RMSE",
            task_type="GPU",
            devices="0",
            verbose=False,
            random_seed=0,
        )
        model.fit(x_gpu, y_gpu)

        out_0 = model.predict(x_gpu[:500], task_type="GPU", output_type="cupy", devices="0")
        out_01 = model.predict(x_gpu[:500], task_type="GPU", output_type="cupy", devices="0:1")
        np.testing.assert_allclose(cp.asnumpy(out_0), cp.asnumpy(out_01), rtol=1e-6, atol=1e-6)


def test_multi_gpu_predict_requires_input_device_in_devices():
    devs = _require_cuda()
    if devs < 2:
        pytest.skip("Need >=2 CUDA devices for multi-GPU prediction test")
    cp = _require_cupy()

    from catboost import CatBoostError, CatBoostRegressor

    rs = np.random.RandomState(0)
    n = 1000
    f = 16
    x_cpu = rs.rand(n, f).astype(np.float32)
    y_cpu = (x_cpu[:, 0] * 0.3 + 0.1).astype(np.float32)

    with cp.cuda.Device(0):
        x_gpu = cp.asarray(x_cpu)
        y_gpu = cp.asarray(y_cpu)

        model = CatBoostRegressor(
            iterations=10,
            depth=6,
            learning_rate=0.1,
            loss_function="RMSE",
            task_type="GPU",
            devices="0",
            verbose=False,
            random_seed=0,
        )
        model.fit(x_gpu, y_gpu)

        with pytest.raises(CatBoostError, match="Input device must be included"):
            model.predict(x_gpu[:10], task_type="GPU", output_type="cupy", devices="1")
