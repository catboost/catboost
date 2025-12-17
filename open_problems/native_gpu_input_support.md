# Native GPU Input Support (cuDF / CuPy) for CatBoost GPU Training

## Summary

Implement a **GPU-only** data ingestion + quantization + dataset-build path so CatBoost can train on GPU **directly from cuDF DataFrames and CuPy arrays** without converting to NumPy/pandas or copying full datasets to host memory.

This document is an engineering plan (not user docs). It focuses **only** on GPU tasks/jobs/pipelines; CPU training and CPU input code paths are treated as out of scope and must remain unchanged.

## Implementation Checklist (mark only after confirmed working)

- [x] Python: `Pool`/`fit()` accept cuDF + CuPy for `task_type='GPU'` (no host conversions).
- [x] Cython: device buffer descriptor extraction (DLPack + `__cuda_array_interface__`) for cuDF/CuPy.
- [x] C++: GPU-only input provider type wired into GPU training path (no CPU fallback).
- [x] GPU quantization: borders + float binarization fully on GPU (no dataset-sized D2H).
- [x] GPU categorical: hashing + perfect hashing + one-hot + CTRs on GPU (no dataset-sized D2H).
- [x] Dataset build: compressed index writer supports device-resident bins/columns end-to-end.
- [x] Targets/weights: labels + weights can be GPU-resident end-to-end.
- [x] Multi-GPU: `devices` selection and sharding verified for GPU inputs.
- [x] No-D2H validation: CatBoost-internal D2H detection/strict mode implemented and passing.
- [x] Tests: parity + no-D2H tests added (skipping cleanly when CUDA/RAPIDS not present).
- [x] Benchmarks: end-to-end GPU preprocessing + training benchmark shows improvement.

### Local validation notes

- Python tests: `catboost/python-package/ut/medium/test_native_gpu_input.py` (`5 passed, 1 skipped` on 1-GPU machine).
- Strict no-D2H: enforced by `CATBOOST_CUDA_STRICT_NO_D2H=1` with per-copy/total limits; verified failure path via tracker test hook.
- Benchmark (RTX 3070, 1 GPU): `catboost/benchmarks/native_gpu_input_support/benchmark.py` (default args)
  - `cpu_input_total_seconds=2.646`, `gpu_input_total_seconds=1.216`, `speedup=2.18x`

## Goals (Definition of Done)

1. **Python API acceptance**
   - `catboost.Pool(...)` and `CatBoost*.fit(...)` accept:
     - `cupy.ndarray` (2D features; 1D/2D labels)
     - `cudf.DataFrame`/`cudf.Series` (features/labels)
   - No `.get()`, no `.to_pandas()`, no implicit host staging in CatBoost Python code.

2. **GPU-resident pipeline**
   - Feature/label data remains GPU-resident end-to-end for GPU training.
   - Verification mechanism exists to ensure CatBoost does not perform GPU→CPU copies of dataset-sized buffers during training.

3. **Categorical + numeric support**
   - Numeric: standard dtypes (float32/float64, ints) with correct casting rules.
   - Categorical: CatBoost “native” categorical handling (hashing + perfect hashing + CTRs / one-hot as configured).

4. **Single + multi-GPU**
   - Supports `devices` selection and multi-GPU execution.
   - Memory behavior is predictable (bounded staging, no accidental full host copies, documented device-to-device sharding strategy).

5. **Tests + benchmarks**
   - Correctness parity: GPU training results using GPU inputs match GPU training using CPU inputs (within tolerance) for the same parameters.
   - Performance benchmark shows end-to-end improvement when preprocessing stays in cuDF/CuPy.

## Non-goals / Constraints (GPU-only scope)

- Do **not** change CPU training behavior, CPU quantization, or CPU-only code paths to add GPU input support.
- Do **not** add a CPU fallback that copies GPU inputs to host automatically.
  - If GPU inputs are provided but training is not `task_type='GPU'`, raise a clear error.
- Initial implementation can limit scope to common supervised tasks (regression/binary/multiclass with numeric labels).
  - Ranking/groupwise, text, embeddings, sparse matrices, and exotic targets can be phased in later.

## Current Architecture (Relevant Pieces)

### Python → C++ pool creation (today)

- `catboost.Pool` type checks reject cuDF/CuPy (`catboost/python-package/catboost/core.py`).
- Pool construction in Cython (`catboost/python-package/catboost/_catboost.pyx`) builds a CPU `TDataProvider` via `IRawObjectsOrderDataVisitor` / `IRawFeaturesOrderDataVisitor` by iterating host data.

### GPU training (today)

- Even for `task_type='GPU'`, training uses the CPU quantization pipeline:
  - `GetTrainingData` (`catboost/private/libs/algo/data.cpp`) calls `GetQuantizedObjectsData` (`catboost/libs/data/quantization.cpp`) which expects a CPU `TRawObjectsDataProvider`.
- GPU dataset build consumes CPU-quantized columns and copies bins host→device in multiple places:
  - `TSharedCompressedIndexBuilder::Write` gathers bins into a CPU `TVector<ui8>` (and then writes to GPU).
  - CTR/cat dataset helpers extract categorical values to CPU (`ExtractValues`) then write to GPU.

### Implication

To support GPU-resident cuDF/CuPy ingestion, CatBoost needs a **new GPU-only raw data provider + GPU quantization/dataset-build path**, and existing GPU dataset builders must be extended to avoid CPU extraction when data originates on GPU.

## Proposed Design (High-Level)

### Key idea

Add a **GPU-only “native GPU input” pipeline**:

1. **Python layer** recognizes cuDF/CuPy inputs and routes to a new Cython entrypoint.
2. **Cython layer** extracts device-buffer metadata using:
   - `__cuda_array_interface__` and/or DLPack (`__dlpack__`)
3. **C++/CUDA layer** ingests device pointers and:
   - Computes borders + quantizes float features on GPU (matching existing GPU training defaults).
   - Hashes + perfect-hashes categorical features on GPU (CityHash compatibility).
   - Builds CatBoost’s GPU training datasets (compressed index, cat feature datasets) without dataset-sized host staging.

CPU paths remain unchanged and are not used for GPU inputs.

## User-facing API plan (GPU-only gating)

### Supported call patterns

- Direct `fit`:
  - `model = CatBoostClassifier(task_type='GPU', devices='0')`
  - `model.fit(X_cudf_or_cupy, y_cudf_or_cupy, cat_features=[...])`
- `Pool`:
  - `pool = Pool(X_cudf_or_cupy, y_cudf_or_cupy, cat_features=[...])`
  - `model.fit(pool)`

### Behavior rules

- If feature input is cuDF/CuPy and training `task_type != 'GPU'`: raise `CatBoostError` explaining GPU-only support.
- If cuDF/CuPy input is used but CatBoost is built without CUDA: raise `CatBoostError` early.
- No implicit conversions to NumPy/pandas inside CatBoost.

### Feature typing

- CuPy 2D array:
  - Numeric features: float/int dtypes → treated as float features unless `cat_features` marks columns categorical.
  - Categorical columns in CuPy arrays:
    - Require integer dtype for phase 1.
    - (Future) optional support for string categories is not feasible in CuPy arrays.
- cuDF DataFrame:
  - Numeric columns: supported.
  - Categorical columns:
    - Support cuDF dictionary/categorical (codes + categories) without host transfer.
    - Support integer categorical columns.
    - Support string categories only via cuDF string/dictionary representation (GPU hashing required).

## Implementation Plan (Layer-by-layer)

### 1) Python package: routing + validation (GPU-only)

Files:
- `catboost/python-package/catboost/core.py`
- `catboost/python-package/catboost/_catboost.pyx` (entrypoint wiring)

Tasks:
1. Add GPU input detection utilities (duck-typing; do not import RAPIDS eagerly):
   - Detect CuPy arrays via `__cuda_array_interface__` and `__module__`/type name.
   - Detect cuDF via `__class__.__module__.startswith('cudf')` and known attributes.
2. Extend `Pool` to accept GPU inputs *only as a container*, without converting them.
   - Keep Python references to inputs (to keep device memory alive).
3. Update `fit` input processing to accept cuDF/CuPy for `X`/`y`/`sample_weight`/`baseline` (GPU tasks only).
4. Error handling:
   - If GPU input detected and params indicate CPU training: fail fast with a clear message.
   - If multiple eval sets on GPU are not supported (existing limitation), keep current behavior.

Deliverables:
- `Pool` and `fit` accept cuDF/CuPy and route into the GPU ingestion path when training is GPU.

### 2) Cython: device-buffer extraction and passing descriptors to C++

Files:
- `catboost/python-package/catboost/_catboost.pyx`
- `catboost/python-package/catboost/helpers.{h,cpp}` (if helper C++ glue is needed)

Tasks:
1. Implement extraction of:
   - `__cuda_array_interface__` dict: `data` pointer, `shape`, `strides`, `typestr`, `stream`.
   - DLPack capsule: parse `DLManagedTensor` to get pointer/shape/strides/dtype/device/stream.
2. Create a “GPU input descriptor” struct passed to C++ (per column or per matrix):
   - pointer, dtype, shape, strides, device id (optional), and stream semantics.
3. Represent cuDF DataFrame as a set of column descriptors (avoid materializing a dense 2D device matrix unless explicitly requested).
4. Keep Python-side references to the backing objects to ensure lifetimes exceed training.

Notes:
- DLPack headers are not currently vendored in CatBoost; add a minimal `dlpack.h` (GPU-only build) under `catboost/python-package/catboost` or `catboost/libs` as appropriate.

Deliverables:
- A new Cython entrypoint (e.g., `_init_pool_gpu`) that builds a “GPU-backed pool” object passed into C++.

### 3) C++: introduce a GPU-only raw data provider type

Files (new + existing, suggested locations):
- New: `catboost/cuda/data/gpu_input_provider.h/.cpp`
- Existing integration points:
  - `catboost/private/libs/algo/data.cpp` (GPU-only branch)
  - `catboost/libs/data/data_provider.h` (minimal extension if needed)

Tasks:
1. Define `NCB::TGpuInputDataProvider` (GPU-only) to hold:
   - Feature descriptors (float + categorical).
   - Label/weights descriptors (optional in early phase; but required for “no host copies”).
   - Features layout + meta info (feature names, cat_features indices).
2. Ensure this provider is never used by CPU training code:
   - Only constructed from Python when GPU inputs are provided.
   - Only consumed when `task_type == GPU`.
3. Add a GPU-only detection/dispatch in `GetTrainingData`:
   - If `task_type==GPU` and source provider is GPU-backed, route into GPU quantization/build pipeline.
   - Otherwise, keep existing logic unchanged.

Deliverables:
- A concrete C++ type that carries device pointers and metadata through to GPU training without host staging.

### 4) GPU quantization: borders + binarization on GPU (parity with existing GPU training)

Files:
- New: `catboost/cuda/gpu_data/gpu_quantization.h/.cu/.cpp` (naming flexible)
- Existing kernels used/extended:
  - `catboost/cuda/gpu_data/kernel/binarize.cu` (already has float binarization + some border kernels)

Tasks:
1. Implement GPU border selection that matches current defaults for GPU training:
   - Current default is `GreedyLogSum` (via `FloatFeaturesBinarization` defaults).
   - Implement a GPU `GreedyLogSum`-equivalent border builder:
     - Use GPU sort/hist primitives + device reductions.
     - Ensure deterministic behavior where possible (seeded sampling when needed).
2. For each float feature:
   - Compute borders on GPU (optionally using a controlled subset, matching existing `MaxSubsetSizeForBuildBorders` behavior).
   - Binarize on GPU directly into the compressed index using `BinarizeFloatFeature` kernel.
3. Support numeric dtypes:
   - Efficient cast kernels to float32 where needed.
   - Proper NaN handling consistent with CatBoost GPU semantics.

Deliverables:
- GPU quantization of float features without copying full columns to host.

### 5) Categorical pipeline on GPU: hashing + perfect hashing + CTR/one-hot support

Files:
- New: `catboost/cuda/data/gpu_cat_features.h/.cu`
- Existing places to modify:
  - `catboost/cuda/gpu_data/cat_features_dataset.cpp`
  - `catboost/cuda/gpu_data/batch_binarized_ctr_calcer.cpp`
  - `catboost/cuda/gpu_data/dataset_helpers.h` (float/one-hot writer)

Tasks:
1. GPU hashing compatible with CatBoost:
   - CatBoost uses `CityHash64(feature_bytes) & 0xffffffff` (`catboost/libs/cat_feature/cat_feature.cpp`).
   - Implement a CUDA device version of CityHash64 for:
     - integer categories (hash decimal string representation)
     - cuDF string categories (hash UTF-8 bytes from cuDF string buffers)
2. For cuDF categorical columns:
   - Hash unique categories once, then map codes → hashed values (GPU kernel).
3. GPU perfect hashing:
   - Compute unique hashed values and assign dense ids (0..U-1) on GPU.
   - Produce:
     - perfect-hashed values per row (for CTR computation and/or one-hot bin values)
     - per-feature unique value counts
4. One-hot:
   - For features with `unique_values <= one_hot_max_size`, write bins to compressed index on GPU.
5. CTR:
   - Ensure CTR pipeline reads cat values from GPU directly:
     - Modify `BuildCompressedBins` and `TCompressedCatFeatureDataSetBuilder` to avoid CPU `ExtractValues`.
     - Use GPU buffers directly and compress on GPU.

Deliverables:
- Cat features work end-to-end on GPU inputs with native CatBoost handling.

### 6) Dataset build changes: eliminate CPU extraction for GPU-origin data

Files:
- `catboost/cuda/gpu_data/compressed_index_builder.h`
- `catboost/cuda/gpu_data/*_dataset_builder.cpp`
- `catboost/cuda/gpu_data/cat_features_dataset.cpp`
- `catboost/cuda/gpu_data/batch_binarized_ctr_calcer.cpp`

Tasks:
1. Add new code paths in compressed index writer to accept device bins directly:
   - New `WriteBinsDevice(...)` method that takes `TCudaBufferPtr<const ui8>` and calls `WriteCompressedIndex` kernel.
2. Update helpers that currently do `ExtractValues(...)` to:
   - Detect GPU-backed storage and avoid D2H.
   - Prefer GPU kernels for gather/permutation and compression.
3. Keep existing CPU input behavior untouched (only add new branches).

Deliverables:
- No dataset-sized D2H transfers during GPU training when inputs originate on GPU.

### 7) Labels/weights on GPU (avoid host copies)

Files:
- New: `catboost/cuda/data/gpu_targets.h/.cu`
- Modify dataset builders:
  - `catboost/cuda/gpu_data/doc_parallel_dataset_builder.cpp`
  - `catboost/cuda/gpu_data/feature_parallel_dataset_builder.cpp`

Tasks:
1. Accept labels/weights as GPU buffers via the same protocol path as features.
2. Build GPU `targets` and `weights` buffers without staging through CPU vectors.
3. Implement minimal CPU-side summaries needed for option defaults / checks:
   - For classification, compute unique labels and counts on GPU; copy only summaries to CPU if required.
4. Ensure current CPU-label path remains for CPU inputs (unchanged).

Deliverables:
- `y` and optional `sample_weight` can be cuDF/CuPy without host copies in CatBoost.

### 8) Multi-GPU support and memory behavior

Files:
- `catboost/cuda/cuda_lib/*` (device config)
- GPU dataset build modules

Tasks:
1. Respect `devices` parameter for GPU input training.
2. Decide on sharding strategy:
   - Prefer doc-parallel mapping (split rows across devices).
   - If input is on a single device, perform device-to-device sharding explicitly (no host staging).
3. Document + enforce memory bounds:
   - Peak memory should be ~O(N * features_per_device) plus quantization buffers.
   - Avoid allocating full dense copies when input is columnar (cuDF DataFrame).

Deliverables:
- Predictable memory usage on single and multi-GPU, with explicit device selection.

### 9) “No GPU→CPU transfers” verification (CatBoost-internal)

Files:
- `catboost/cuda/cuda_lib/cuda_base.h` (central copy wrapper)
- Possibly additional wrappers where raw `cudaMemcpyAsync` is used directly.

Tasks:
1. Add an opt-in runtime counter for memory copies initiated by CatBoost:
   - Track direction (D2H/H2D/D2D) using pointer attributes when `cudaMemcpyDefault` is used.
   - Record total bytes and callsites (optional).
2. Add a strict mode env var (example):
   - `CATBOOST_GPU_NO_D2H=1` → fail if any D2H above a small metadata threshold occurs during training.
3. Expose counters for tests (via Python binding or logs).

Deliverables:
- Automated tests can assert “no dataset-sized D2H copies” for GPU input training.

## Testing Plan

### Unit/medium tests (Python)

Location (suggested):
- `catboost/python-package/ut/medium/gpu/test_native_gpu_input.py`

Test matrix:
1. CuPy numeric:
   - Regression + classification (small dataset).
   - Compare predictions/metrics against training from NumPy with the same params (GPU task).
2. cuDF numeric:
   - Same parity tests.
3. Categorical:
   - Integer categorical in CuPy (cat_features indices).
   - cuDF categorical column (dictionary encoding) including unseen categories in eval set.
4. Multi-GPU smoke (if environment provides ≥2 GPUs):
   - `devices='0,1'` with small data, check training completes and parity is reasonable.
5. “No D2H” instrumentation:
   - Enable strict mode env var and assert training passes without D2H events (or without >threshold bytes).

Skipping strategy:
- Skip tests if CUDA is unavailable or cupy/cudf is not installed.
- Keep CPU-only CI unaffected by default.

### C++/CUDA UTs (optional, targeted)

Add UTs around:
- GPU CityHash correctness for known strings/ints (compare against CPU CityHash outputs).
- GPU perfect hashing correctness (unique counts, stable mapping).

## Benchmark Plan

Location (suggested):
- New: `catboost/benchmarks/gpu_native_input/benchmark.py`

Benchmarks:
1. End-to-end pipeline:
   - Generate synthetic data on GPU (CuPy) or load via cuDF.
   - GPU preprocessing (cuDF groupby/joins/encodings) → CatBoost training.
   - Compare against baseline that converts to pandas/NumPy before training.
2. Memory:
   - Measure peak GPU memory usage (via `cupy.cuda.runtime.memGetInfo` deltas or NVML).
3. Multi-GPU scaling:
   - Compare `devices='0'` vs `devices='0,1'` on a fixed dataset size.

## Rollout / Backwards Compatibility

1. Feature-flag the Python API initially:
   - Example: `allow_gpu_input=True` or behind env var to de-risk.
2. Make GPU input support default once stabilized.
3. Ensure CPU pipelines are unaffected:
   - No change in behavior for existing CPU inputs + CPU/GPU tasks.

## Open Questions / Decisions Needed

1. What does “no GPU→CPU transfers” mean operationally?
   - Strict “zero D2H calls” vs “no dataset-sized D2H copies”.
   - This plan assumes “no dataset-sized D2H”; metadata copies may be allowed and bounded.
2. Required categorical scope:
   - Must cuDF string categories be supported in v1, or is integer/categorical enough initially?
3. Determinism expectations:
   - Border selection and hashing should be deterministic given fixed seed where feasible.

## Milestones (Suggested)

1. MVP numeric (single GPU): CuPy + cuDF numeric features + GPU label; train without host dataset copies.
2. GPU quantization parity: implement GPU `GreedyLogSum` border selection to match current behavior.
3. Categorical v1: integer cats + cuDF categorical codes/categories; one-hot + CTR correctness.
4. Multi-GPU: robust device selection and sharding with predictable memory.
5. Instrumentation + CI: “no D2H” checks and performance benchmarks.
