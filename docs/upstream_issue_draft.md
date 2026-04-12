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
- 1000+ tests passing

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

3. **Library path feature gap**: Our `IModelTrainer` implementation currently supports 10 pointwise losses. Our standalone engine supports all 12 losses + ranking + Lossguide + all features listed above. We would close this gap before submitting a PR. Is the `IModelTrainer` interface the right integration point?

4. **MLX dependency management**: MLX is available via pip and Homebrew. Should it be resolved via `find_package`, vendored, or handled differently?

### Performance

On a MacBook Pro M3 Max (128GB unified memory), training 100k rows x 50 features x 100 iterations:
- CatBoost-MLX (Metal GPU): competitive with CatBoost CPU on the same hardware
- Primary advantage: GPU acceleration without CUDA, native to Apple Silicon

### Files involved

```
catboost/mlx/
  kernels/          # Metal compute shaders (.metal)
  gpu_data/         # GPU data layout and transfer
  methods/          # Tree search (histogram, scoring, boosting)
  targets/          # Loss functions (pointwise)
  train_lib/        # IModelTrainer implementation + model export
  tests/            # Standalone test binary (csv_train.cpp)
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
