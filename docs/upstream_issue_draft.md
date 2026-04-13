# Draft: Upstream Issue for catboost/catboost

> **Status:** DRAFT -- for Ramos to review before posting.
> **Target:** https://github.com/catboost/catboost/issues/new

---

## Title

Feature proposal: MLX (Apple Silicon GPU) training backend

## Body

### Summary

We have developed an MLX-based GPU training backend for CatBoost that runs natively on Apple Silicon via Metal compute shaders. The backend implements the `IModelTrainer` interface and registers via `TTrainerFactory` under `ETaskType::GPU`, following the same pattern as the CUDA backend. We would like to discuss upstreaming this work.

**Repository:** https://github.com/RR-AMATOK/catboost-mlx

### What it does

- Gradient boosted decision tree training on Apple Silicon GPU using Apple's MLX framework and Metal shaders
- Custom Metal kernels for histogram computation, split scoring, tree application, and leaf estimation
- 12 loss functions: RMSE, MAE, Quantile, Huber, Poisson, Tweedie, MAPE, Logloss, CrossEntropy, MultiClass, PairLogit, YetiRank
- 3 grow policies: SymmetricTree, Depthwise, Lossguide (best-first)
- Tree depth 1-10 with multi-pass leaf accumulation for depth 7-10
- Categorical features with CTR target encoding
- Early stopping, subsampling, monotone constraints, snapshot save/resume
- Python bindings via nanobind (zero-copy numpy arrays, GIL released during training)
- All 3 grow policies on GPU (SymmetricTree, Depthwise, Lossguide) — the CUDA backend only supports SymmetricTree
- Non-symmetric tree export via `TNonSymmetricTreeModelBuilder`
- CI on GitHub Actions macos-14 (Apple Silicon M1): C++ build gate + full pytest suite
- 1010 tests passing, 5 xfail (known edge cases)

### Platform constraints

- **Apple Silicon only** (darwin-arm64, macOS 14+)
- **Requires MLX** (`pip install mlx` or Homebrew)
- **Mutually exclusive with CUDA** at build time (same `ETaskType::GPU` registration; safe because CUDA is never available on darwin-arm64)

### Current integration points

The backend already follows CatBoost's patterns:

1. **Factory registration** (`catboost/mlx/train_lib/train.cpp`):
   ```cpp
   TTrainerFactory::TRegistrator<NCatboostMlx::TMLXModelTrainer> MLXGPURegistrator(ETaskType::GPU);
   ```

2. **IModelTrainer implementation**: `TMLXModelTrainer::TrainModel()` accepts the standard CatBoost parameter types (`TCatBoostOptions`, `TTrainingDataProviders`, etc.)

3. **Build system**: Would follow the CUDA platform-file pattern:
   - `CMakeLists.darwin-arm64-mlx.txt` files alongside existing `-cuda` variants
   - `HAVE_MLX` flag analogous to `HAVE_CUDA`
   - `find_package(MLX)` analogous to `find_package(CUDAToolkit)`
   - `add_global_library_for` for static registrator linkage

### What we would like to discuss

1. **Is the CatBoost team open to an Apple Silicon GPU backend?** We understand this adds a platform-specific dependency and maintenance surface.

2. **CMake-only or ya.make required?** We can provide CMake integration following the CUDA pattern. We do not have access to YaTool -- would the team handle ya.make porting?

3. **`IModelTrainer` integration**: Our `IModelTrainer` implementation supports all 12 loss functions, all 3 grow policies (including Depthwise and Lossguide on GPU — which the CUDA backend does not offer), and non-symmetric tree export via `TNonSymmetricTreeModelBuilder`. Is the `IModelTrainer` interface the right integration point, or would the team prefer a different registration pattern?

4. **MLX dependency management**: MLX is available via pip and Homebrew. Should it be resolved via `find_package`, vendored, or handled differently?

### Performance

Current benchmarks on MacBook Pro M3 Max (128GB unified memory), 100 iterations, depth=6:

| Dataset   | Loss    | CPU (s) | MLX (s) | CPU iter/s | MLX iter/s |
|-----------|---------|---------|---------|------------|------------|
| 10k × 50  | RMSE    |    0.20 |   32.73 |      506.8 |        3.1 |
| 100k × 50 | RMSE    |    0.41 |   70.43 |      244.3 |        1.4 |
| 500k × 50 | RMSE    |    1.18 |  175.97 |       84.5 |        0.6 |
| 10k × 50  | Logloss |    0.30 |   32.08 |      332.5 |        3.1 |
| 100k × 50 | Logloss |    0.70 |   69.55 |      142.7 |        1.4 |
| 500k × 50 | Logloss |    1.73 |  173.38 |       57.9 |        0.6 |

**Honest assessment:** The MLX backend is currently **~100× slower** than CatBoost CPU on these small-to-medium datasets. This is expected — CatBoost CPU is extremely SIMD-optimized, and per-iteration Metal kernel dispatch overhead dominates at these scales. The same pattern exists with CUDA CatBoost, where GPU only wins at large scale.

**Where we see the path to competitive performance:**
- Kernel fusion (histogram + scoring in a single dispatch)
- Batched iteration dispatch (amortize Metal command buffer overhead across multiple iterations)
- Async CPU-GPU overlap (gradient computation pipelining)
- Larger datasets (1M+ rows, 200+ features) where Metal's parallel bandwidth dominates

This submission is about **correctness and architecture** — proving the full CatBoost algorithm works end-to-end on Metal. Performance optimization is the next phase, and we would welcome guidance from the CatBoost team on where the CUDA backend's key optimizations live.

### Files involved

```
catboost/mlx/
  kernels/          # Metal compute shaders (.metal) + kernel_sources.h
  gpu_data/         # GPU data layout, transfer, and dataset builder
  methods/          # Tree search (histogram, scoring, boosting)
  targets/          # Loss functions: pointwise (10) + pairwise (PairLogit, YetiRank)
  train_lib/        # IModelTrainer + model export (symmetric + non-symmetric trees)
  tests/            # csv_train.cpp, csv_predict.cpp, model_export_test.cpp

python/
  catboost_mlx/     # Python package with nanobind bindings (zero-copy, GIL-free)
  tests/            # 1010+ pytest tests

.github/workflows/
  mlx-build.yaml    # C++ compile gate (macos-14)
  mlx-test.yaml     # Full Python test suite (macos-14, Metal GPU)

benchmarks/         # MLX vs CPU benchmark script + results
```

All code is Apache 2.0 licensed. No modifications to existing CatBoost source files.

---

**Problem:**
No GPU training support on Apple Silicon.

**catboost version:**
Fork based on current master (synced regularly).

**Operating System:**
macOS 14+ (Sonoma and later).

**CPU:**
Apple Silicon (M1/M2/M3/M4 family).

**GPU:**
Apple Silicon integrated Metal GPU.
