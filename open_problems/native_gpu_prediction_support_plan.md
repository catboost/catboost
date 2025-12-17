# Native GPU Prediction Support — Full Parity Plan (cuDF / CuPy / DLPack)

This plan extends the Phase 1 “numeric-only” MVP in `open_problems/native_gpu_prediction_support.md` to full (practical) parity for **GPU-resident inputs → GPU-resident outputs**, including **integer categorical**, **one-hot splits**, and **CTR-based splits**, plus multiclass prediction types and remaining API surface.

“Pure GPU” means: **no device→host copies of feature matrices during prediction**. Small host copies for **model metadata** are allowed. Host outputs are allowed only when explicitly requested (e.g. `output_type='numpy'` or legacy APIs that return NumPy).

## Phase A — One-hot categorical splits (no CTR yet)

### A1) CUDA evaluator correctness: XOR mask parity
- Apply the per-split XOR mask in CUDA tree traversal (required for OneHot equality semantics).

### A2) GPU-side categorical hashing for prediction inputs
- For categorical input columns provided as integers (CuPy/cuDF/DLPack), compute CatBoost categorical hashes on GPU (CityHash64(ToString(int)) → low32), matching CPU.
- Store hashed categorical features in a packed, column-first device buffer for the apply block.

### A3) OneHot bucket quantization on GPU
- Upload OneHot `Values` from the model to GPU once.
- Compute OneHot bin values on GPU (local index + 1 within each MAX_VALUES_PER_BIN bucket group), writing directly into the evaluator’s quantized feature buffer.

### A4) Validation
- New unit tests: OneHot-only model (force via `one_hot_max_size`, disable CTRs) parity vs CPU.
- Strict no-D2H validation: ensure no feature D2H occurs for GPU-input prediction.

## Phase B — CTR inference on GPU (static CTR tables)

### B1) Upload CTR tables to GPU
- Require `TStaticCtrProvider` when CTRs are present.
- Upload dense hash buckets (`NCatboost::TBucket`) and CTR blob data in aligned device arrays (mean-history and int counters).

### B2) GPU CTR computation
- For each CTR feature used by the model:
  - Compute projection hashes on GPU from hashed cats + float splits + one-hot splits (matching `CalcHashes` semantics).
  - Lookup in dense hash table on GPU.
  - Compute CTR value for all supported `ECtrType` variants (mean, counter/freq, buckets, borders).
  - Binarize CTR value against the model borders and write CTR buckets into the quantized feature buffer.

### B3) Validation
- Extend prediction tests to cover models with CTR splits (e.g. `max_ctr_complexity > 0`) and verify CPU parity.
- Strict no-D2H: verify no feature D2H occurs (only optional host output).

## Phase C — Multiclass + full `prediction_type` parity (GPU outputs)

### C1) Multi-dimensional raw evaluation
- Support `ApproxDimension > 1` in CUDA evaluator kernels and GPU output buffers.
- Keep `ntree_start/ntree_end` working for staged evaluation and for staged_predict.

### C2) Prediction-type postprocessing
- Implement or provide GPU-side postprocessing for:
  - `RawFormulaVal`
  - `Probability` / `MultiProbability` (softmax)
  - `LogProbability` (log-softmax)
  - `Class`
  - `Exponent`
  - `RMSEWithUncertainty`

### C3) Validation
- Add multiclass parity tests for raw/class/probabilities vs CPU.
- Add GPU output tests for shapes/dtypes for all supported prediction types.

## Phase D — Remaining API parity

### D1) `staged_predict*` on GPU inputs
- Ensure staged prediction accepts GPU inputs without host-staging features.
- Use GPU evaluator with tree ranges; host output allowed unless an explicit GPU output API is added.

### D2) `calc_leaf_indexes` on GPU inputs
- Implement GPU kernel(s) to compute leaf indices for oblivious trees from quantized GPU features.
- Copy leaf indices to host for NumPy return value (legacy API).

### D3) Validation
- Unit tests for `staged_predict` and `calc_leaf_indexes` on GPU-backed pools.

## Phase E — Multi-GPU inference + benchmarks

### E1) Multi-GPU inference
- Implement `devices` sharding for inference over multiple CUDA devices.
- Ensure deterministic output ordering and P2P-safe transfers.
- Add tests that skip cleanly when `<2` GPUs are available.

### E2) Benchmarks
- Add an in-repo benchmark script for inference comparing:
  - CPU input → CPU output
  - CPU input → GPU evaluator
  - GPU input → GPU output (new path)

### E3) Final validation
- Full rebuild of Python extension and run the prediction/input native GPU test suites.
- Update `open_problems/native_gpu_prediction_support.md` checklist only after each item is verified working.

