import argparse
import statistics
import time


def _sync(cp):
    try:
        cp.cuda.runtime.deviceSynchronize()
    except Exception:
        pass


def _timeit(fn, sync_fn, repeats):
    times = []
    for _ in range(repeats):
        if sync_fn is not None:
            sync_fn()
        start = time.perf_counter()
        fn()
        if sync_fn is not None:
            sync_fn()
        times.append(time.perf_counter() - start)
    return times


def _summarize(name, times, rows):
    med = statistics.median(times)
    mean = statistics.mean(times)
    rows_per_sec_med = (rows / med) if med > 0 else float("inf")
    rows_per_sec_mean = (rows / mean) if mean > 0 else float("inf")
    return {
        "name": name,
        "median_seconds": med,
        "mean_seconds": mean,
        "median_rows_per_second": rows_per_sec_med,
        "mean_rows_per_second": rows_per_sec_mean,
    }


def main():
    parser = argparse.ArgumentParser(description="CatBoost native GPU prediction benchmark (CuPy/cuDF/DLPack path).")
    parser.add_argument("--rows", type=int, default=200_000)
    parser.add_argument("--cols", type=int, default=200)
    parser.add_argument("--train-iterations", type=int, default=500)
    parser.add_argument("--train-depth", type=int, default=6)
    parser.add_argument("--predict-rows", type=int, default=200_000)
    parser.add_argument("--predict-repeats", type=int, default=20)
    parser.add_argument("--device", type=str, default="0")
    args = parser.parse_args()

    import cupy as cp
    from catboost import CatBoostRegressor

    cp.random.seed(0)
    x_gpu = cp.random.random((args.rows, args.cols), dtype=cp.float32)
    y_gpu = (x_gpu[:, 0] * 0.3 + x_gpu[:, 1] * -0.2 + 0.1).astype(cp.float32)

    x_cpu = cp.asnumpy(x_gpu)
    y_cpu = cp.asnumpy(y_gpu)

    train_params = dict(
        iterations=args.train_iterations,
        depth=args.train_depth,
        learning_rate=0.1,
        loss_function="RMSE",
        task_type="GPU",
        devices=args.device,
        random_seed=0,
        verbose=False,
        allow_writing_files=False,
    )

    model = CatBoostRegressor(**train_params)
    model.fit(x_gpu, y_gpu)

    predict_rows = min(args.predict_rows, args.rows)
    x_gpu_pred = x_gpu[:predict_rows]
    x_cpu_pred = x_cpu[:predict_rows]

    # Warm-up CUDA/CatBoost.
    _sync(cp)
    model.predict(x_cpu_pred[: min(10_000, predict_rows)], task_type="CPU")
    model.predict(x_gpu_pred[: min(10_000, predict_rows)], task_type="GPU", output_type="cupy")
    _sync(cp)

    cpu_times = _timeit(
        lambda: model.predict(x_cpu_pred, task_type="CPU"),
        sync_fn=None,
        repeats=args.predict_repeats,
    )
    gpu_in_cpu_out_times = _timeit(
        lambda: model.predict(x_gpu_pred, task_type="GPU", output_type="numpy"),
        sync_fn=lambda: _sync(cp),
        repeats=args.predict_repeats,
    )
    gpu_in_gpu_out_times = _timeit(
        lambda: model.predict(x_gpu_pred, task_type="GPU", output_type="cupy"),
        sync_fn=lambda: _sync(cp),
        repeats=args.predict_repeats,
    )

    cpu = _summarize("cpu_input->cpu_output (task_type=CPU)", cpu_times, predict_rows)
    gpu_in_cpu_out = _summarize("gpu_input->cpu_output (task_type=GPU, output_type=numpy)", gpu_in_cpu_out_times, predict_rows)
    gpu_in_gpu_out = _summarize("gpu_input->gpu_output (task_type=GPU, output_type=cupy)", gpu_in_gpu_out_times, predict_rows)

    print(f"train_rows={args.rows} train_cols={args.cols} train_iters={args.train_iterations} train_depth={args.train_depth} device={args.device}")
    print(f"predict_rows={predict_rows} repeats={args.predict_repeats}")
    for row in (cpu, gpu_in_cpu_out, gpu_in_gpu_out):
        print(
            f"{row['name']}: "
            f"median_seconds={row['median_seconds']:.6f} mean_seconds={row['mean_seconds']:.6f} "
            f"median_rows_per_second={row['median_rows_per_second']:.1f} mean_rows_per_second={row['mean_rows_per_second']:.1f}"
        )

    speedup_cpu_median = cpu["median_seconds"] / gpu_in_gpu_out["median_seconds"] if gpu_in_gpu_out["median_seconds"] > 0 else float("inf")
    speedup_cpu_mean = cpu["mean_seconds"] / gpu_in_gpu_out["mean_seconds"] if gpu_in_gpu_out["mean_seconds"] > 0 else float("inf")
    print(f"speedup_vs_cpu_output_median={speedup_cpu_median:.2f}x")
    print(f"speedup_vs_cpu_output_mean={speedup_cpu_mean:.2f}x")


if __name__ == "__main__":
    main()
