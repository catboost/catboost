# ParallelModelCalcerWrapper — 2-Minute Read

## The Problem

CatBoost's existing `ModelCalcerWrapper` is a clean, header-only C++ wrapper
around the C API, but every batch-prediction call is **single-threaded**.
In production systems that score tens of thousands of rows per request — think
ranking, fraud detection, or real-time recommendation — the model evaluation
step becomes the bottleneck even though the machine has dozens of idle CPU
cores.

## The Idea

The CatBoost evaluator is **stateless per call**: multiple threads can invoke
`CalcModelPrediction*` on the same `ModelCalcerHandle*` simultaneously without
any data races. This means we can split a batch of N documents into K
equal-sized chunks, hand each chunk to a separate `std::thread`, and collect
the results — with **zero synchronisation overhead** because each thread writes
into a disjoint slice of the pre-allocated output vector.

### `ParallelModelCalcerWrapper` in a nutshell

```cpp
// Drop-in replacement for ModelCalcerWrapper
ParallelModelCalcerWrapper calcer("model.cbm");

// New parallel methods — just add numThreads
std::vector<double> scores = calcer.CalcFlatParallel(batch, /*numThreads=*/4);

// numThreads=0 → std::thread::hardware_concurrency() automatically
std::vector<double> scores = calcer.CalcFlatParallel(batch);

// Single-threaded path unchanged (numThreads=1, zero thread-spawn overhead)
std::vector<double> scores = calcer.CalcFlat(batch);
```

The class is **header-only** (`parallel_calcer_wrapper.h`), requires only
`c_api.h` and a C++17 compiler, and adds no new dependencies beyond
`<thread>`.

### Key design decisions

| Decision | Rationale |
|---|---|
| Composition over inheritance | `ModelCalcerWrapper::CalcerHolder` is `private`; managing our own handle avoids modifying existing files |
| Pre-allocated output vector | Threads write to disjoint offsets — no mutex, no false sharing on cache-line boundaries |
| `numThreads=0` → `hardware_concurrency` | Sensible default for production; caller can override for latency-sensitive paths |
| Exception propagation | Worker exceptions are captured via `std::exception_ptr` and re-thrown in the calling thread |
| Chunk clamping | `numThreads` is clamped to `docCount` so a 3-document batch never spawns 1000 threads |

## Test Results

Tests were run on a **96-core** machine (Intel Xeon, `std::thread::hardware_concurrency() = 96`)
with a 500-tree, 10-feature CatBoost classifier and a batch of **50,000 documents**.

```
=== Suite 1: Correctness ===
[PASS] 1a: empty batch -> empty result
[PASS] 1b: single-object CalcFlat matches CalcFlatParallel(1 thread)
[PASS] 1c: 4-thread results match 1-thread results exactly
[PASS] 1c: 8-thread results match 1-thread results exactly
[PASS] 1c: auto-thread results match 1-thread results exactly
[PASS] 1d: CalcParallel 4-thread matches 1-thread
[PASS] 1e: numThreads > docCount clamped correctly
[PASS] 1f: InitFromFile loads same model
... (17/17 passed)

=== Suite 2: Performance ===
  Batch size : 50,000 documents
  1 thread   :  13.0 ms
  4 threads  :   3.5 ms   → 3.68x speedup
  96 threads :   5.4 ms   → 2.39x speedup
[PASS] 2a: 4-thread is at least 1.5x faster than 1-thread
[PASS] 2b: auto-thread is not slower than 1-thread
```

**4 threads deliver a 3.68x speedup** — close to the theoretical 4x maximum,
confirming that the evaluator is genuinely CPU-bound and parallelises well.
The 96-thread run is slower than 4 threads because thread-spawn and OS
scheduling overhead dominates for a 50k-document batch; in practice, 4–16
threads is the sweet spot for typical batch sizes.

## When to Use It

- **Batch scoring services** where a single request contains hundreds to
  hundreds-of-thousands of rows.
- **Offline pipelines** that need to maximise throughput on a multi-core node.
- Any place where you already use `ModelCalcerWrapper` and want a drop-in
  upgrade with a single extra parameter.

## Files

| File | Description |
|---|---|
| `parallel_calcer_wrapper.h` | Header-only implementation of `ParallelModelCalcerWrapper` |
| `parallel_calcer_wrapper_test.cpp` | Correctness + performance test suite |
| `PARALLEL_CALCER.md` | This document |

## Build the Test

```bash
g++ -std=c++17 -O2 -pthread \
    parallel_calcer_wrapper_test.cpp \
    -I. \
    -L<path_to_libcatboostmodel_dir> \
    -lcatboostmodel \
    -Wl,-rpath,<path_to_libcatboostmodel_dir> \
    -o parallel_calcer_wrapper_test

./parallel_calcer_wrapper_test path/to/model.cbm
```
