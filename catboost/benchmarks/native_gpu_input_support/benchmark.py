import argparse
import time


def _sync(cp):
    try:
        cp.cuda.runtime.deviceSynchronize()
    except Exception:
        pass


def _timeit(cp, fn):
    _sync(cp)
    start = time.perf_counter()
    fn()
    _sync(cp)
    return time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description="CatBoost native GPU input benchmark (CuPy).")
    parser.add_argument("--rows", type=int, default=200_000)
    parser.add_argument("--cols", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--device", type=str, default="0")
    args = parser.parse_args()

    import cupy as cp
    from catboost import CatBoostRegressor

    cp.random.seed(0)
    x_gpu = cp.random.random((args.rows, args.cols), dtype=cp.float32)
    y_gpu = (x_gpu[:, 0] * 0.3 + x_gpu[:, 1] * -0.2 + 0.1).astype(cp.float32)

    params = dict(
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=0.1,
        loss_function="RMSE",
        task_type="GPU",
        devices=args.device,
        random_seed=0,
        verbose=False,
        allow_writing_files=False,
    )

    def train_cpu_input():
        x_cpu = cp.asnumpy(x_gpu)
        y_cpu = cp.asnumpy(y_gpu)
        model = CatBoostRegressor(**params)
        model.fit(x_cpu, y_cpu)

    def train_gpu_input():
        model = CatBoostRegressor(**params)
        model.fit(x_gpu, y_gpu)

    # Warm up CUDA/CatBoost.
    _timeit(cp, lambda: CatBoostRegressor(iterations=2, task_type="GPU", devices=args.device, verbose=False, allow_writing_files=False).fit(cp.asnumpy(x_gpu[:10_000]), cp.asnumpy(y_gpu[:10_000])))

    cpu_seconds = _timeit(cp, train_cpu_input)
    gpu_seconds = _timeit(cp, train_gpu_input)

    speedup = cpu_seconds / gpu_seconds if gpu_seconds > 0 else float("inf")
    print(f"rows={args.rows} cols={args.cols} iters={args.iterations} depth={args.depth} device={args.device}")
    print(f"cpu_input_total_seconds={cpu_seconds:.3f}")
    print(f"gpu_input_total_seconds={gpu_seconds:.3f}")
    print(f"speedup={speedup:.2f}x")


if __name__ == "__main__":
    main()
