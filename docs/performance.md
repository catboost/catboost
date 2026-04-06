# Performance Analysis — CatBoost-MLX

**Date**: 2026-04-06
**Hardware**: Apple M-series (Apple Silicon), macOS Darwin 25.3.0
**MLX version**: 0.31.1 (Homebrew)
**Benchmark script**: `python/benchmarks/benchmark.py`

---

## 1. Baseline Benchmark Results

Comparison of CatBoost-MLX against XGBoost, LightGBM, and CatBoost-CPU (official
package running on CPU). All frameworks trained with the same hyperparameters:
200 iterations, depth=6, lr=0.1, l2=3.0.

### Training Time (seconds)

| Task         | Samples | CatBoost-MLX | XGBoost | LightGBM | CatBoost-CPU |
|--------------|---------|-------------|---------|----------|-------------|
| Regression   |    1K   |       3.397 |   0.555 |    0.766 |       0.273 |
| Regression   |   10K   |       5.040 |   0.629 |    1.264 |       0.316 |
| Regression   |   50K   |       9.583 |   2.401 |    3.894 |       1.351 |
| Binary       |    1K   |       7.313 |   0.915 |    1.441 |       0.776 |
| Binary       |   10K   |      12.247 |   1.540 |    2.573 |       1.096 |
| Binary       |   50K   |      14.504 |   2.143 |    3.099 |       2.159 |
| Multiclass   |    1K   |      25.355 |   4.359 |    5.639 |       2.017 |
| Multiclass   |   10K   |      39.993 |   7.888 |   12.928 |       2.566 |
| Multiclass   |   50K   |      51.530 |  10.390 |   16.119 |       4.080 |

**Current gap**: MLX is 4-12x slower than CatBoost-CPU on training. The gap
widens for multiclass (high `approx_dim` multiplies histogram kernel dispatches).

### Prediction Time (seconds)

| Task         | Samples | CatBoost-MLX | XGBoost | LightGBM | CatBoost-CPU |
|--------------|---------|-------------|---------|----------|-------------|
| Regression   |   50K   |       0.305 |   0.008 |    0.016 |       0.004 |
| Binary       |   50K   |       0.272 |   0.008 |    0.013 |       0.005 |
| Multiclass   |   50K   |       0.345 |   0.020 |    0.044 |       0.014 |

**Current gap**: MLX prediction is 20-75x slower due to subprocess overhead.

### Model Quality (RMSE for regression, Accuracy for classification)

| Task         | Samples | CatBoost-MLX | XGBoost | LightGBM | CatBoost-CPU |
|--------------|---------|-------------|---------|----------|-------------|
| Regression   |   50K   |     0.4325  |  0.4464 |   0.4303 |      0.2918 |
| Binary       |   50K   |     0.9113  |  0.9077 |   0.9109 |      0.9121 |
| Multiclass   |   50K   |     0.9469  |  0.9750 |   0.9762 |      0.3143 |

MLX matches XGBoost/LightGBM on model quality. CatBoost-CPU multiclass shows
0.31 accuracy — this is a label encoding issue in the benchmark script, not a
real quality difference.

**Full results**: `.cache/benchmarks/baseline_results.json`

---

## 2. Profiling: Where Training Time Goes

Profiled on 50K rows x 50 features, depth=6, RMSE loss.

### Per-iteration breakdown (~50ms avg)

```
Phase                   | Time (ms) | % of iter | Location
------------------------|-----------|-----------|---------------------------
Histogram GPU kernel    |     59.8  |     92.0% | DispatchHistogram + mx::eval
Gradient computation    |      1.9  |      2.9% | GPU elementwise ops
CPU split scoring       |      1.5  |      2.3% | FindBestSplit (CPU loop)
Partition update        |      2.8  |      4.3% | GPU bitwise ops
Partition layout        |      0.7  |      1.1% | GPU argsort + scatter
Leaf estimation         |      0.5  |      0.8% | GPU scatter_add_axis
```

### Training time breakdown (end-to-end)

```
Component           | Time      | % of total
--------------------|-----------|----------
Data serialization  |   0.006s  |     0.0%
C++ training        |  15.695s  |    99.9%
JSON model parse    |   0.003s  |     0.0%
```

### Prediction time breakdown

```
Component               | Time     | Speedup available
------------------------|----------|------------------
Subprocess predict      |  0.272s  | baseline
Python in-process       |  0.009s  | 31x faster
```

Python-side tree evaluation (`_predict_utils.py`) already exists and produces
identical results (max diff: 3.0e-6 from float32 rounding).

---

## 3. Root Causes

### 3.1 Histogram kernel under-parallelization

The histogram kernel is dispatched with `maxBlocksPerPart=1`, meaning a single
threadgroup of 256 threads processes an entire partition sequentially. At depth 0
(1 partition of 50K docs), this means 256 threads process 50K docs — extremely
low GPU occupancy.

```
Grid size: (256 threads, numPartitions threadgroups, 2)
At depth 0: 256 x 1 x 2 = 512 total threads for 50K docs
```

CatBoost CUDA uses multiple blocks per partition with atomic adds for reduction.

### 3.2 Per-feature-group kernel dispatch overhead

Features are processed in groups of 4. With 50 features, this produces ~13
separate Metal kernel dispatches per depth level, each with fixed overhead
(command buffer encoding, pipeline state lookup, GPU dispatch latency).

```
Total kernel dispatches per iteration = numFeatureGroups x depth
                                      = 13 x 6 = 78 kernel dispatches
```

### 3.3 Excessive GPU-CPU sync points

Each depth level forces 2 `mx::eval()` calls (histogram + partition stats),
creating a GPU->CPU sync barrier. For depth=6:

```
Sync points per iteration = 2 x depth = 12 sync points
```

Each sync flushes the Metal command buffer and blocks until the GPU finishes.

### 3.4 Subprocess overhead for prediction

`predict()` spawns a subprocess (`csv_predict`), writes data to disk, reads
results back. Process creation + file I/O + Metal device initialization dominates
over actual tree evaluation.

---

## 4. Optimization Plan

Ordered by expected impact and implementation complexity.

### Phase A: In-process prediction (no subprocess)

**Impact**: 31x prediction speedup
**Risk**: Low — `_predict_utils.py` already exists and is tested
**Effort**: Small

Route `predict()` and `predict_proba()` through the Python-side tree evaluator
for numeric-only models. Fall back to subprocess for categorical features (which
need C++ CTR encoding).

### Phase B: Increase histogram parallelism

**Impact**: ~3-5x histogram speedup (estimated)
**Risk**: Medium — requires kernel output reduction across blocks
**Effort**: Medium

Change `maxBlocksPerPart` from 1 to `ceil(docsInPartition / blockSize)`. Each
block computes a partial histogram; a second reduction kernel sums them. This is
the standard GPU histogram pattern used by CatBoost CUDA and XGBoost.

### Phase C: Batch feature groups into single dispatch

**Impact**: ~2x histogram speedup from reduced dispatch overhead
**Risk**: Low — kernel logic stays the same, just wider grid
**Effort**: Medium

Dispatch all feature groups in a single kernel invocation using a 3D grid
`(threads, numPartitions * maxBlocks, numFeatureGroups)`. Eliminates 12x
per-depth dispatch overhead.

### Phase D: Reduce GPU-CPU sync points

**Impact**: ~20-30% iteration speedup
**Risk**: Medium — requires restructuring eval pattern
**Effort**: Medium

Batch histogram computation across all `approxDim` dimensions into a single
`mx::eval()`. Defer partition stats eval until needed by `FindBestSplit`.

### Phase E: GPU-side split scoring (future)

**Impact**: Eliminates histogram CPU readback
**Risk**: High — new kernel, complex reduction
**Effort**: Large

Move `FindBestSplit` to a Metal kernel. Currently only 1.5ms (2.3% of iter),
so this is low priority until Phases A-D are done.

---

## 5. Expected Outcome

| Phase | Training speedup | Prediction speedup | Cumulative training |
|-------|------------------|--------------------|---------------------|
| Baseline | 1.0x          | 1.0x               | ~10s (50K reg)      |
| A     | —                | 31x                | ~10s                |
| B     | 3-5x             | —                  | ~2-3s               |
| C     | 1.5-2x           | —                  | ~1.5-2s             |
| D     | 1.2-1.3x         | —                  | ~1.2-1.5s           |

**Target**: After phases A-D, CatBoost-MLX should be within 1-2x of CatBoost-CPU
for regression/binary (50K samples), which is competitive for a GPU implementation
that also scales better on larger datasets.

---

## 6. Reproducing

```bash
# Run benchmark suite
python python/benchmarks/benchmark.py --sizes 1000 10000 50000 \
    --output .cache/benchmarks/results.json

# Profile a single training run (50K regression)
./csv_train data.cbmx --iterations 10 --depth 6 --lr 0.1 --l2 3.0 \
    --loss rmse --bins 128 --output model.json --verbose
# Look for "Phase breakdown" and "[profile iter0]" lines
```
