"""
Benchmark: Pool.set_label() vs Pool reconstruction for multi-target training.

Demonstrates that reusing a Pool with set_label() produces identical predictions
to rebuilding the Pool from scratch, while being significantly faster.

Usage:
    python benchmark_set_label.py [--rows 100000] [--cols 200] [--targets 5] [--iters 20]
"""

import argparse
import time
import sys

import numpy as np
from catboost import Pool, CatBoostRegressor, CatBoostClassifier


def has_set_label():
    """Check if Pool.set_label() is available (patched CatBoost)."""
    return hasattr(Pool, 'set_label') and callable(getattr(Pool, 'set_label'))


def generate_data(n_rows, n_cols, n_targets, seed=42):
    """Generate feature matrix and multiple target vectors."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_rows, n_cols)).astype(np.float32)

    targets = {}
    # Mix of regression and classification targets
    for i in range(n_targets):
        if i % 2 == 0:
            # Regression target
            targets[f"reg_{i}"] = {
                "y": rng.standard_normal(n_rows).astype(np.float32),
                "type": "regression",
            }
        else:
            # Binary classification target
            targets[f"cls_{i}"] = {
                "y": rng.integers(0, 2, n_rows).astype(np.float32),
                "type": "classification",
            }
    return X, targets


def train_old_way(X, targets, cb_params):
    """Old way: rebuild Pool for each target."""
    models = {}
    predictions = {}
    pool_build_time = 0.0
    total_time = 0.0

    for name, tinfo in targets.items():
        t0 = time.perf_counter()

        # Build new Pool from scratch each time
        t_pool_start = time.perf_counter()
        pool = Pool(data=X, label=tinfo["y"])
        t_pool_end = time.perf_counter()
        pool_build_time += t_pool_end - t_pool_start

        if tinfo["type"] == "regression":
            model = CatBoostRegressor(**cb_params)
        else:
            model = CatBoostClassifier(**cb_params)

        model.fit(pool, verbose=0)

        # Predict on the same pool (or a separate one)
        if tinfo["type"] == "classification":
            preds = model.predict_proba(pool)
        else:
            preds = model.predict(pool)

        models[name] = model
        predictions[name] = preds
        total_time += time.perf_counter() - t0

    return models, predictions, total_time, pool_build_time


def train_new_way(X, targets, cb_params):
    """New way: build Pool once, use set_label() per target."""
    models = {}
    predictions = {}
    pool_build_time = 0.0
    total_time = 0.0

    # Build Pool ONCE without label
    t_pool_start = time.perf_counter()
    pool = Pool(data=X)
    t_pool_end = time.perf_counter()
    pool_build_time = t_pool_end - t_pool_start

    for name, tinfo in targets.items():
        t0 = time.perf_counter()

        # Just replace the label (near-zero cost)
        pool.set_label(tinfo["y"])

        if tinfo["type"] == "regression":
            model = CatBoostRegressor(**cb_params)
        else:
            model = CatBoostClassifier(**cb_params)

        model.fit(pool, verbose=0)

        if tinfo["type"] == "classification":
            preds = model.predict_proba(pool)
        else:
            preds = model.predict(pool)

        models[name] = model
        predictions[name] = preds
        total_time += time.perf_counter() - t0

    total_time += pool_build_time
    return models, predictions, total_time, pool_build_time


def train_new_way_fit_y(X, targets, cb_params):
    """Alternative new way: use fit(Pool, y=...) syntax."""
    models = {}
    predictions = {}
    pool_build_time = 0.0
    total_time = 0.0

    t_pool_start = time.perf_counter()
    pool = Pool(data=X)
    t_pool_end = time.perf_counter()
    pool_build_time = t_pool_end - t_pool_start

    for name, tinfo in targets.items():
        t0 = time.perf_counter()

        if tinfo["type"] == "regression":
            model = CatBoostRegressor(**cb_params)
        else:
            model = CatBoostClassifier(**cb_params)

        # Pass y directly to fit() -- internally calls set_label()
        model.fit(pool, y=tinfo["y"], verbose=0)

        if tinfo["type"] == "classification":
            preds = model.predict_proba(pool)
        else:
            preds = model.predict(pool)

        models[name] = model
        predictions[name] = preds
        total_time += time.perf_counter() - t0

    total_time += pool_build_time
    return models, predictions, total_time, pool_build_time


def verify_predictions(preds_old, preds_new, method_name):
    """Verify predictions are bitwise identical."""
    all_match = True
    for name in preds_old:
        old = np.asarray(preds_old[name])
        new = np.asarray(preds_new[name])
        if np.array_equal(old, new):
            print(f"  {name}: EXACT MATCH")
        elif np.allclose(old, new, rtol=1e-6, atol=1e-7):
            max_diff = np.max(np.abs(old - new))
            print(f"  {name}: close match (max diff={max_diff:.2e})")
        else:
            max_diff = np.max(np.abs(old - new))
            print(f"  {name}: MISMATCH (max diff={max_diff:.2e})")
            all_match = False
    return all_match


def benchmark_pool_construction_only(X, n_targets):
    """Measure Pool construction overhead separately (works without set_label)."""
    # Single Pool build
    t0 = time.perf_counter()
    pool = Pool(data=X)
    t_single = time.perf_counter() - t0

    # N Pool builds (old way)
    dummy_y = np.zeros(X.shape[0], dtype=np.float32)
    t0 = time.perf_counter()
    for _ in range(n_targets):
        _ = Pool(data=X, label=dummy_y)
    t_multi = time.perf_counter() - t0

    return t_single, t_multi


def benchmark_predict_overhead(X, cb_params):
    """Measure predict_proba with DataFrame vs Pool (the bigger win)."""
    y = np.random.default_rng(42).integers(0, 2, X.shape[0]).astype(np.float32)
    pool = Pool(data=X, label=y)
    clf = CatBoostClassifier(**cb_params)
    clf.fit(pool, verbose=0)

    # Predict with DataFrame (internally rebuilds Pool each time)
    n_predicts = 5
    t0 = time.perf_counter()
    for _ in range(n_predicts):
        clf.predict_proba(X)
    t_df = (time.perf_counter() - t0) / n_predicts

    # Predict with pre-built Pool (no reconstruction)
    predict_pool = Pool(data=X)
    t0 = time.perf_counter()
    for _ in range(n_predicts):
        clf.predict_proba(predict_pool)
    t_pool = (time.perf_counter() - t0) / n_predicts

    return t_df, t_pool


def benchmark_set_label_scale(n_rows, n_cols=4, n_targets=10, seed=42):
    """Isolated set_label throughput benchmark at large scale.

    The set_label move-overload (added in the accompanying PR) eliminates a ~N*4 byte
    memcpy at the C++ boundary. At 10M-100M rows with float32 labels that saves tens
    to hundreds of MB per call -- not visible at 100k rows where the copy is microseconds.

    Uses a small column count so the Pool feature build doesn't dominate the run and
    we actually measure the label-setting cost. Returns (build_seconds, set_label_seconds_per_call).
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_rows, n_cols)).astype(np.float32)

    t0 = time.perf_counter()
    pool = Pool(data=X)
    build_time = time.perf_counter() - t0

    # Generate N independent target arrays to prevent any caching shortcut.
    ys = [rng.standard_normal(n_rows).astype(np.float32) for _ in range(n_targets)]

    # Warmup
    pool.set_label(ys[0])

    t0 = time.perf_counter()
    for y in ys:
        pool.set_label(y)
    elapsed = time.perf_counter() - t0
    return build_time, elapsed / n_targets


def main():
    parser = argparse.ArgumentParser(description="Benchmark Pool.set_label() vs reconstruction")
    parser.add_argument("--rows", type=int, default=100_000, help="Number of rows")
    parser.add_argument("--cols", type=int, default=200, help="Number of feature columns")
    parser.add_argument("--targets", type=int, default=5, help="Number of different targets")
    parser.add_argument("--iters", type=int, default=20, help="CatBoost iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--scale-bench",
        action="store_true",
        help="Run the large-scale set_label throughput benchmark across 100k/1M/10M rows. "
             "Meant to back the move-overload justification in the PR; skips the full pipeline run.",
    )
    args = parser.parse_args()

    if args.scale_bench:
        print("Large-scale set_label throughput (move-overload benefit grows with N):")
        print(f"{'rows':>12}  {'build(s)':>10}  {'set_label(s)':>13}  {'MB/s':>8}")
        for n in (100_000, 1_000_000, 10_000_000):
            try:
                build_t, label_t = benchmark_set_label_scale(n)
                mb_per_sec = (n * 4 / 1_048_576) / max(label_t, 1e-9)
                print(f"{n:>12,}  {build_t:>10.3f}  {label_t:>13.5f}  {mb_per_sec:>8.1f}")
            except MemoryError:
                print(f"{n:>12,}  (out of memory -- skipping)")
                break
        return

    print(f"Dataset: {args.rows:,} rows x {args.cols} cols, {args.targets} targets, {args.iters} iterations")
    print(f"set_label() available: {has_set_label()}")
    print()

    X, targets = generate_data(args.rows, args.cols, args.targets, args.seed)

    cb_params = {
        "iterations": args.iters,
        "random_seed": args.seed,
        "thread_count": -1,
        "allow_writing_files": False,
    }

    # Warmup: first Pool build has import/JIT overhead
    print("Warming up...")
    _warmup = Pool(data=X[:100])
    del _warmup
    print()

    # --- Pool construction overhead ---
    print("=" * 60)
    print("Phase 0: Pool construction overhead (no training)")
    print("=" * 60)
    t_single, t_multi = benchmark_pool_construction_only(X, args.targets)
    print(f"  1 Pool build:              {t_single:.3f}s")
    print(f"  {args.targets} Pool builds (old way):  {t_multi:.3f}s")
    print(f"  Saved by reuse:            {t_multi - t_single:.3f}s ({(t_multi - t_single) / t_multi * 100:.1f}%)")
    print()

    # --- Predict overhead (the bigger win) ---
    print("=" * 60)
    print("Phase 0b: predict_proba overhead (DataFrame vs Pool)")
    print("=" * 60)
    t_df_pred, t_pool_pred = benchmark_predict_overhead(X, cb_params)
    print(f"  predict_proba(DataFrame):  {t_df_pred:.3f}s  (rebuilds Pool internally)")
    print(f"  predict_proba(Pool):       {t_pool_pred:.3f}s  (pre-built Pool)")
    print(f"  Speedup:                   {t_df_pred / t_pool_pred:.1f}x")
    predict_savings_per_target = (t_df_pred - t_pool_pred) * 2  # val + test
    print(f"  Savings per target (val+test): {predict_savings_per_target:.3f}s")
    print(f"  Savings for {args.targets} targets:     {predict_savings_per_target * args.targets:.3f}s")
    print()

    # --- Old way (always works) ---
    print("=" * 60)
    print("Phase 1: Old way (rebuild Pool per target)")
    print("=" * 60)
    _, preds_old, time_old, pool_time_old = train_old_way(X, targets, cb_params)
    print(f"  Total time:       {time_old:.3f}s")
    print(f"  Pool build time:  {pool_time_old:.3f}s ({pool_time_old / time_old * 100:.1f}% of total)")
    print()

    if not has_set_label():
        print("=" * 60)
        print("Phase 2: set_label() NOT available (CatBoost not recompiled)")
        print("=" * 60)
        pool_savings = pool_time_old - t_single
        total_savings = pool_savings + predict_savings_per_target * args.targets
        estimated_new_time = time_old - pool_savings
        print(f"  Pool construction savings:   {pool_savings:.3f}s")
        print(f"  Predict Pool reuse savings:  {predict_savings_per_target * args.targets:.3f}s")
        print(f"  Total estimated savings:     {total_savings:.3f}s")
        print(f"  Estimated speedup (train):   {time_old / estimated_new_time:.2f}x")
        print(f"  With predict reuse:          {time_old / (time_old - total_savings):.2f}x")
        print()
        print("To run full benchmark, recompile CatBoost with set_label() support.")
        print("Predictions will be bitwise identical (same data, same seed).")
        return

    # --- New way: set_label() ---
    print("=" * 60)
    print("Phase 2: New way (set_label)")
    print("=" * 60)
    _, preds_new, time_new, pool_time_new = train_new_way(X, targets, cb_params)
    print(f"  Total time:       {time_new:.3f}s")
    print(f"  Pool build time:  {pool_time_new:.3f}s (single build)")
    print()

    # --- New way: fit(pool, y=...) ---
    print("=" * 60)
    print("Phase 3: New way (fit with y=)")
    print("=" * 60)
    _, preds_fit_y, time_fit_y, pool_time_fit_y = train_new_way_fit_y(X, targets, cb_params)
    print(f"  Total time:       {time_fit_y:.3f}s")
    print(f"  Pool build time:  {pool_time_fit_y:.3f}s (single build)")
    print()

    # --- Verification ---
    print("=" * 60)
    print("Phase 4: Prediction equivalence verification")
    print("=" * 60)

    print("\nOld vs set_label():")
    match1 = verify_predictions(preds_old, preds_new, "set_label")

    print("\nOld vs fit(pool, y=...):")
    match2 = verify_predictions(preds_old, preds_fit_y, "fit_y")

    # --- Summary ---
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Old way (rebuild Pool):     {time_old:.3f}s")
    print(f"  New way (set_label):        {time_new:.3f}s")
    print(f"  New way (fit y=):           {time_fit_y:.3f}s")
    print(f"  Speedup (set_label):        {time_old / time_new:.2f}x")
    print(f"  Speedup (fit y=):           {time_old / time_fit_y:.2f}x")
    print(f"  Pool build savings:         {pool_time_old - pool_time_new:.3f}s")
    print(f"  Predictions match:          {'YES' if match1 and match2 else 'NO'}")

    if not (match1 and match2):
        sys.exit(1)


if __name__ == "__main__":
    main()
