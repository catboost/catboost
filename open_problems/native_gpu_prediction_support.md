# Native GPU Prediction Support (cuDF / CuPy / DLPack) for CatBoost

## Summary

Implement a **pure GPU** model-application (inference) path so CatBoost can run `predict()` / `predict_proba()` **directly on GPU-resident inputs** (cuDF, CuPy, generic DLPack) **without staging feature matrices on host**. Training-side native GPU inputs are already covered by `open_problems/native_gpu_input_support.md`; this document covers **prediction/apply** only.

“Pure GPU” in this plan means:
- **No device→host copies of features** during prediction (bounded metadata copies are OK).
- Any required reformatting (packing, casting, transpose) is **device→device only**.
- Prediction results can be produced as **GPU outputs** (CuPy / cuDF / DLPack). Host outputs (NumPy) remain available **only when explicitly requested** and should be treated as an intentional D2H copy.

## Current Status (What’s Missing Today)

1. Python `predict*` rejects cuDF/CuPy/DLPack inputs early (`catboost/python-package/catboost/core.py:_is_data_single_object`).
2. Even if a GPU-backed `Pool` exists, the C++ apply pipeline is CPU-centric:
   - `catboost/private/libs/algo/features_data_helpers.cpp:CreateFeaturesBlockIterator()` supports only CPU raw/CPU-quantized providers.
3. Existing “GPU evaluator” (`catboost/libs/model/cuda/evaluator.*`) is **not a GPU-input evaluator**:
   - It expects **host** feature vectors and performs host memcpy + H2D.
   - It currently rejects categorical/text/embedding features and only supports oblivious + 1D, with several prediction types unimplemented.

## Implementation Checklist (mark only after confirmed working)

- [x] Python: `predict`, `predict_proba`, `staged_predict`, `calc_leaf_indexes` accept CuPy/cuDF/DLPack inputs (no implicit `.get()`/`.to_pandas()`/NumPy conversion).
- [x] Python: new `output_type` (or equivalent) supports GPU outputs: `cupy`, `cudf`, `dlpack`, plus explicit `numpy` host output.
- [x] Cython: GPU prediction binding returning GPU outputs without device→host staging.
- [x] C++: GPU apply entrypoint for `NCB::TGpuRawObjectsDataProvider` (no fallback into CPU `ApplyModelMulti` pipeline).
- [x] CUDA: device-pointer input path (zero-copy fast path; otherwise D2D pack/cast only).
- [x] CUDA: multiclass and full `prediction_type` parity (`RawFormulaVal`, `Probability`, `LogProbability`, `MultiProbability`, `Class`, `Exponent`, `RMSEWithUncertainty`).
- [x] CUDA: integer categorical + one-hot splits parity with CPU (GPU inputs).
- [x] CUDA: cuDF categorical/dictionary string categories.
- [x] CUDA: CTR support (model CTR tables uploaded to GPU; CTR features computed on GPU; no CPU CTR provider calls during apply).
- [ ] Multi-GPU inference: `devices` sharding; P2P-safe transfers only; deterministic gather.
- [x] No-D2H validation: strict mode forbids input D2H during prediction; D2H output only when explicitly requested.
- [x] Tests: parity + no-D2H tests for CuPy/cuDF/DLPack prediction; clean skips when CUDA/RAPIDS absent.
- [x] Benchmarks: end-to-end “GPU input → GPU output” throughput and latency vs existing apply paths.

## Current Status (Validated on 1 GPU)

Implemented (pure-GPU apply for GPU-resident inputs):
- Python API: `output_type` plumbed through `predict`/`predict_proba`/`predict_log_proba` (`catboost/python-package/catboost/core.py`).
- Python API: `devices` override accepted by `predict*` and forwarded to the GPU-input apply path.
- Cython routing for GPU-backed pools + GPU outputs (`cupy`/`cudf`/`dlpack`) (`catboost/python-package/catboost/_catboost.pyx`).
- C++ apply entrypoints for `NCB::TGpuRawObjectsDataProvider`, including device-output (`catboost/python-package/catboost/helpers.cpp`).
- CUDA evaluator extensions:
  - Tree range (`ntree_start`/`ntree_end`) support.
  - Leaf index computation for oblivious models.
  - One-hot split XOR-mask parity.
  - GPU-side CTR computation using uploaded CTR tables.
- cuDF categorical (`dtype='category'`) support (dictionary codes → CatBoost cat-hash mapping).

Known limitations:
- Models: oblivious only; no text/embedding; no estimated features.
- Multi-GPU inference requires `utils.get_gpu_device_count() >= 2` (tests skip on 1-GPU machines).
- `output_type` GPU path supports: `RawFormulaVal`, `Probability`, `LogProbability`, `Class`, `Exponent`, `RMSEWithUncertainty`.

Local validation:
- Build: `ninja -C catboost/python-package/build/temp.linux-x86_64-cpython-312 _catboost` (success).
- Tests: `catboost/python-package/ut/medium/test_native_gpu_prediction.py` (`23 passed, 2 skipped` on 1-GPU machine; skips are multi-GPU-only tests).
- Benchmark: `catboost/benchmarks/native_gpu_prediction_support/benchmark.py` (example run: `--rows 50000 --cols 50 --train-iterations 300 --predict-rows 50000 --predict-repeats 10`)
  - `cpu_input->cpu_output` median `0.008139s` (≈`6.14M` rows/s)
  - `gpu_input->gpu_output` median `0.003284s` (≈`15.22M` rows/s)
  - speedup vs CPU apply (median): `2.48x`

## Goals (Definition of Done)

### 1) Inputs (CuPy/cuDF/DLPack)
Support prediction directly from:
- `cupy.ndarray` (2D features; optional 1D for single object).
- `cudf.DataFrame` / `cudf.Series`.
- Any object implementing DLPack (`__dlpack__` + `__dlpack_device__`) with CUDA device type.

Input “advanced” requirements:
- Strided arrays (non-contiguous) supported via **device-side packing**.
- Row-major and column-major supported (choose best layout; avoid host transposes).
- Correct handling of CUDA streams from `__cuda_array_interface__` / DLPack (`stream` field), including mixed-stream inputs.
- Multi-device safety: detect per-column `DeviceId` and either enforce “single device per call” or implement explicit P2P copy/sharding.

### 2) Outputs (GPU-first)
Support returning predictions as:
- CuPy ndarray (`output_type='cupy'`) when CuPy is installed.
- cuDF Series/DataFrame (`output_type='cudf'`) when cuDF is installed.
- DLPack capsule (`output_type='dlpack'`) without requiring CuPy/cuDF imports.
- NumPy (`output_type='numpy'`) only as an explicit opt-in (host copy).

### 3) Model parity (GPU-trained model class)
At minimum, support prediction for the full set of models that CatBoost can train on GPU:
- Oblivious trees.
- Float + integer categorical features.
- One-hot splits + CTR splits.
- Multi-dimensional output (multiclass / multitarget) including probability/softmax family.

Stretch (separate milestone):
- Non-symmetric trees, text features, embeddings (only if/when GPU training + GPU preprocessing are supported end-to-end).

### 4) “Pure GPU” enforcement
With `CATBOOST_CUDA_STRICT_NO_D2H=1`, prediction must:
- Perform **no feature D2H copies** (any size).
- Allow bounded D2H only for metadata (and only if within configured limits).
- Allow D2H of predictions **only** when user explicitly requests `output_type='numpy'`.

## Proposed User-Facing API

### Keep existing entrypoints; add an output selector

Add `output_type` (name can vary) to the prediction APIs:
- `predict(..., task_type='GPU', output_type='cupy'|'cudf'|'dlpack'|'numpy')`
- `predict_proba(..., task_type='GPU', output_type=...)`
- `staged_predict(..., task_type='GPU', output_type=...)` (or a GPU-only iterator variant)

Rules:
- If input is GPU-resident and `task_type != 'GPU'`: raise `CatBoostError` (“GPU input requires task_type='GPU'”).
- If `output_type` is omitted and input is GPU-resident:
  - Prefer GPU output (`dlpack` as the dependency-free default; `cupy`/`cudf` when installed).
- Existing CPU-input behavior remains unchanged (including optional GPU evaluator that stages features).

## Architecture Choice (Recommended)

### Build a dedicated “GPU apply” pipeline for GPU-backed pools

Implement a new C++ path:
- `ApplyModelMultiGpu(...)` (name flexible)
- Accepts `TFullModel` + `TGpuRawObjectsDataProvider` (or a lightweight GPU input descriptor).
- Produces predictions into GPU memory (device buffer), with optional explicit conversion to host.

Why:
- The existing apply stack (`CreateFeaturesBlockIterator` → `MakeQuantizedFeaturesForEvaluator` → evaluator) is CPU-centric and hard-wires host feature vectors and CPU visitors.
- A separate path keeps CPU prediction semantics stable and avoids fragile cross-cutting changes.

## Implementation Plan (Phased, with validation after each phase)

### Phase 0 — Spec & plumbing
Deliverables:
1. Define exact API surface:
   - `output_type` values and defaults.
   - How `prediction_type` maps to output shapes/dtypes on GPU.
2. Define strict no-D2H semantics for prediction:
   - “No feature D2H ever.”
   - “Output D2H allowed only for `output_type='numpy'`.”
3. Add a new tracking mode if needed (callsite allowlist for output copies).

Validation:
- Unit tests for API gating and error messages (no CUDA required).

### Phase 1 — Numeric-only GPU input + GPU output (MVP)
Scope:
- Oblivious, 1D models with float features only (match current GPU evaluator constraints initially).

Core work:
1. Python:
   - Extend `catboost/python-package/catboost/core.py:_is_data_single_object` and `_process_predict_input_data` to accept GPU inputs.
   - Ensure `Pool(...)` creation for GPU inputs does not trigger host conversion.
2. Cython bindings:
   - Add a new `_base_predict_gpu(...)` that routes to C++ `ApplyModelMultiGpu`.
   - Add a lightweight Python return type that can expose output via DLPack / `__cuda_array_interface__` (dependency-free).
3. C++ apply:
   - Add `ApplyModelMultiGpuFloatOnly(...)` that:
     - Validates input/model compatibility.
     - Creates a GPU evaluator context without host staging.
     - Executes in blocks to bound memory.
4. CUDA evaluator update (minimal):
   - Add a device-input entrypoint that accepts an external device pointer (no host memcpy).
   - Add a “write results to device buffer” variant (skip final D2H copy).

Notes on data layout:
- Fast path: if input is CuPy float32 contiguous row-major, evaluate directly via row-first accessor.
- Otherwise: pack/cast on GPU into the evaluator’s preferred padded layout (device→device only), using existing GPU input copy kernels (`catboost/cuda/gpu_data/kernel/gpu_input_utils.cuh`).

Validation:
- New pytest file (suggested): `catboost/python-package/ut/medium/test_native_gpu_prediction.py`
  - Train a numeric GPU model; compare CPU-input predictions vs GPU-input predictions (tolerance).
  - With `CATBOOST_CUDA_STRICT_NO_D2H=1`, assert prediction succeeds when `output_type` is GPU and fails (or counts D2H) when `output_type='numpy'` without allowlist.

### Phase 2 — Multiclass + full `prediction_type` parity for numeric models
Scope:
- Still float-only inputs/models, but enable:
  - `ApproxDimension > 1`
  - `Probability` / `MultiProbability` / `LogProbability`
  - `Exponent`
  - `RMSEWithUncertainty`

Core work:
1. Extend CUDA post-processing kernels in `catboost/libs/model/cuda/evaluator.cu`:
   - Implement softmax for `MultiProbability` / multiclass `Probability`.
   - Implement log-softmax for `LogProbability`.
   - Implement `Exponent` and uncertainty variants with GPU parity vs CPU.
2. Ensure output shaping and dtype match Python expectations for all prediction types.

Validation:
- Parity tests for multiclass raw/class/probabilities vs CPU apply, on the same model and input.

### Phase 3 — Categorical (one-hot) support on GPU
Scope:
- Integer categorical columns from CuPy/cuDF/DLPack.
- One-hot splits parity.

Core work:
1. GPU hashed categorical extraction:
   - Reuse the existing integer categorical hashing kernels introduced for training (`catboost/cuda/gpu_data/kernel/gpu_input_factorize.cu`).
   - Produce `hashedCatFeatures` on GPU for the apply block (no host).
2. Extend evaluator quantization to include one-hot buckets:
   - Use `TModelTrees::TRuntimeData::EffectiveBinFeaturesBucketCount` and the repacked bins mapping (supports >254 borders/values via multiple buckets).
   - Add device-side one-hot “bucket value” computation:
     - For each one-hot feature, map hashed value → index in `TOneHotFeature::Values`, emit per-bucket local index (or 0 when not matched).
   - Upload per-onehot-value tables to GPU once per model (sorted arrays per onehot feature for binary search, or a GPU hash map).

Validation:
- Update existing training test `catboost/python-package/ut/medium/test_native_gpu_input.py:test_cudf_input_cat_ctr_smoke` to predict directly from cuDF once available.
- Add parity tests for one-hot models:
  - Train with `one_hot_max_size` forcing one-hot splits.
  - Compare GPU-input predictions vs CPU-input predictions (tolerance).
- Strict no-D2H tests: ensure no input D2H occurs during hashing/lookup.

### Phase 4 — CTR support on GPU (largest chunk)
Scope:
- Full CTR parity for GPU-trained models with categorical + CTR features.

Core work (high-level steps):
1. GPU CTR table representation:
   - Extract CTR tables from the model’s `CtrProvider` (e.g., `TStaticCtrProvider` / `TCtrValueTable`).
   - Convert each table to a GPU-friendly layout:
     - Option A: sorted `(hash, value)` arrays + binary search.
     - Option B: custom open-addressing GPU hash table (faster; more code).
2. GPU CTR computation kernels:
   - Implement GPU equivalents of the core CTR keying logic:
     - Hashing of projections: combine hashed categorical features and referenced binarized features (float/onehot) using the same hashing scheme as CPU (`CalcHash`).
     - Table lookup and CTR value computation (respect `TCtrConfig` priors/shift/scale).
3. Integrate with the evaluator:
   - Compute CTR float values on GPU for the apply block.
   - Quantize CTR splits into the same bucket representation used by repacked bins.
4. Performance considerations:
   - Cache computed CTR values per apply block and per CTR base.
   - Fuse operations where possible (hash → lookup → border compare).

Validation:
- Parity tests vs CPU apply for CTR-heavy models (binary + multiclass).
- Add randomized fuzz tests (small sizes) ensuring exact hash/key parity with CPU for integer categorical inputs.
- Strict no-D2H tests: ensure no CTR computation calls into CPU providers.

### Phase 5 — Multi-GPU inference
Scope:
- Support `devices='0,1,...'` for prediction with GPU inputs.

Core work:
1. Shard objects across devices (block partitioning).
2. P2P-safe data movement:
   - If input resides on one device but multiple devices are requested, use `cudaMemcpyPeerAsync` (no host bounce).
   - If P2P is unavailable, either:
     - refuse multi-GPU inference for that configuration, or
     - require inputs already present per-device (documented).
3. Gather outputs:
   - GPU output: concatenate on a chosen device (D2D) or return a list-of-device-buffers.
   - NumPy output: explicit gather + D2H.

Validation:
- Multi-GPU smoke tests gated on `utils.get_gpu_device_count() >= 2`.

### Phase 6 — Cleanup, documentation, and benchmarking
Deliverables:
1. Benchmarks:
   - New script: `catboost/benchmarks/native_gpu_prediction_support/benchmark.py`
   - Compare:
     - CPU input → CPU output (baseline)
     - GPU input → CPU output (explicit D2H)
     - GPU input → GPU output (pure GPU)
2. Docs:
   - Update Python docstrings for `predict*` to document GPU inputs and output types.
   - Add a short “GPU inference” section to a suitable doc page (or release notes).

Validation:
- Run benchmarks locally and record before/after results and memory behavior.

## Key Code Touchpoints (Expected)

Python:
- `catboost/python-package/catboost/core.py` (GPU input acceptance for predict; new `output_type`)
- `catboost/python-package/catboost/_catboost.pyx` (new GPU apply binding + GPU output wrapper)

C++ / CUDA:
- `catboost/private/libs/algo/apply.cpp` (new GPU-only apply entrypoint; keep CPU path unchanged)
- `catboost/cuda/data/gpu_input_provider.h` (reuse `TGpuRawObjectsDataProvider` as canonical GPU input container)
- `catboost/libs/model/cuda/evaluator.cpp/.cu/.cuh` (device-input + device-output APIs; multiclass + prediction types; onehot + CTR quantization)
- `catboost/cuda/gpu_data/kernel/gpu_input_utils.cuh` (packing/casting strided GPU input to evaluator layout)
- `catboost/cuda/gpu_data/kernel/gpu_input_factorize.cuh/.cu` (integer categorical hashing utilities)

## Risks / Complexity Notes

- CTR support is the critical-path risk: GPU lookup structures must be fast and memory-efficient to beat CPU apply.
- Feature parity vs CPU must be validated with careful hashing compatibility checks (especially for categorical projections).
- Stream correctness is easy to get subtly wrong; plan requires explicit event/stream handling for mixed-stream inputs.
- Multi-GPU inference adds significant complexity; consider shipping single-GPU apply first.

## Suggested Milestones

1. MVP: float-only GPU input + GPU output for oblivious 1D models.
2. Multiclass + prediction type parity (still float-only).
3. Integer categorical + one-hot parity.
4. CTR parity.
5. Multi-GPU inference.
6. Benchmarks + docs + stabilization.
