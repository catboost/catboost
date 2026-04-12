# Contributing to CatBoost-MLX

This document covers everything you need to build, test, and contribute to the Metal GPU backend.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Building](#building)
3. [Running Tests](#running-tests)
4. [Branch Convention](#branch-convention)
5. [Commit Format](#commit-format)
6. [Development Protocol](#development-protocol)
7. [Code Style](#code-style)
8. [State Files](#state-files)

---

## Prerequisites

- **Hardware**: Apple Silicon Mac (M1 or later). The Metal GPU backend will not build or run on Intel Macs or non-Apple hardware.
- **OS**: macOS 14 (Sonoma) or later, which ships with Metal 3.
- **Xcode Command Line Tools**: `xcode-select --install`
- **MLX**: Install via Homebrew.
  ```bash
  brew install mlx
  ```
  Verify the install path with `brew --prefix mlx`. The CI uses whatever version Homebrew pins; check `.github/workflows/mlx_test.yaml` for the current pinned version.
- **Python**: 3.9 or later (3.9, 3.10, 3.11, 3.12, and 3.13 are all tested in CI).
- **Compiler**: `clang++` from Xcode Command Line Tools. The `-std=c++17` flag is required; some headers use C++20 extensions that are tolerated with `-Wno-c++20-extensions`.
- **CMake 3.27+** and **nanobind** (for the in-process `_core` extension):
  ```bash
  brew install cmake
  pip install nanobind
  ```

---

## Building

All standalone binaries are compiled directly with `clang++` from the repository root. No CMake is required for the standalone path.

Set the MLX prefix once:

```bash
export MLX_PREFIX=$(brew --prefix mlx)
```

### csv_train — standalone training binary (used by Python)

```bash
clang++ -std=c++17 -O2 -I. \
  -I${MLX_PREFIX}/include \
  -L${MLX_PREFIX}/lib -lmlx \
  -framework Metal -framework Foundation \
  -Wno-c++20-extensions \
  catboost/mlx/tests/csv_train.cpp -o csv_train
```

### csv_predict — standalone prediction binary

```bash
clang++ -std=c++17 -O2 -I. \
  -I${MLX_PREFIX}/include \
  -L${MLX_PREFIX}/lib -lmlx \
  -framework Metal -framework Foundation \
  -Wno-c++20-extensions \
  catboost/mlx/tests/csv_predict.cpp -o csv_predict
```

### bench_boosting — performance benchmark

```bash
clang++ -std=c++17 -O2 -I. \
  -I${MLX_PREFIX}/include \
  -L${MLX_PREFIX}/lib -lmlx \
  -framework Metal -framework Foundation \
  -Wno-c++20-extensions \
  catboost/mlx/tests/bench_boosting.cpp -o bench_boosting
```

### build_verify_test — build correctness smoke test

```bash
clang++ -std=c++17 -O2 -I. \
  -I${MLX_PREFIX}/include \
  -L${MLX_PREFIX}/lib -lmlx \
  -framework Metal -framework Foundation \
  -Wno-c++20-extensions \
  catboost/mlx/tests/build_verify_test.cpp -o build_verify_test
```

### Bundle binaries for the Python package

The Python package expects the training and prediction binaries at `python/catboost_mlx/bin/`:

```bash
mkdir -p python/catboost_mlx/bin
cp csv_train python/catboost_mlx/bin/
cp csv_predict python/catboost_mlx/bin/
```

### _core — nanobind in-process extension (primary Python backend)

The nanobind extension compiles the C++ training engine directly into a Python extension module (`_core.cpython-*.so`). When present, the Python package calls it in-process instead of spawning `csv_train` as a subprocess. This is the recommended path for development.

Build via `pip` (which invokes CMake through `mlx.extension.CMakeBuild`):

```bash
cd python && pip install -e . --no-build-isolation
```

The `--no-build-isolation` flag is required so CMake can locate the MLX installation in the active environment. If `mlx.extension` is unavailable, the install falls back to the pure-Python subprocess backend with no error.

To build the extension directly with CMake (useful for debugging the build):

```bash
cd python/catboost_mlx/_core
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Verify the extension loaded:

```bash
python3 -c "from catboost_mlx.core import _HAS_NANOBIND; print('nanobind:', _HAS_NANOBIND)"
```

---

## Running Tests

### Python tests

Install the package in development mode first:

```bash
cd python && pip install -e . --no-build-isolation
# or, without the nanobind extension:
# pip install -e "python/[dev]"
```

Run the full test suite:

```bash
pytest python/tests/ -v --tb=short
```

Run with coverage:

```bash
pytest python/tests/ -v --tb=short --cov=catboost_mlx --cov-report=term-missing
```

Lint (the CI gate uses `ruff`):

```bash
ruff check python/catboost_mlx/ python/tests/
```

Notable test files:

| File | What it covers |
|---|---|
| `test_basic.py` | Core fit/predict smoke tests |
| `test_nanobind_parity.py` | Verifies subprocess and nanobind paths produce identical results |
| `test_qa_round14_sprint11_nanobind.py` | Sprint 11 nanobind-specific QA |
| `test_qa_round13_sprint10.py` | Lossguide grow policy correctness |
| `test_qa_round11_sprint8_library_losses.py` | Poisson, Tweedie, MAPE on library path |

### Build verification

Runs a smoke test to confirm all kernels compile and dispatch without error:

```bash
./build_verify_test
```

### Benchmarks

The benchmark binary synthesizes in-memory data and measures per-iteration wall time for the full kernel pipeline:

```bash
# Default: 100k rows, 50 features, regression, depth 6, 100 iterations
./bench_boosting

# Binary classification
./bench_boosting --rows 100000 --features 50 --classes 2 --depth 6 --iters 100

# Multiclass (K=3)
./bench_boosting --rows 20000 --features 30 --classes 3 --depth 5 --iters 50
```

Iteration 0 is the Metal cold-start (kernel compilation). Compare warm averages (iterations 1–N) when measuring optimization impact.

---

## Branch Convention

Starting with Sprint 4, all sprint work lives on a dedicated branch. Direct commits to `master` during a sprint are not allowed.

**Branch naming:**

```
mlx/sprint-<N>-<short-topic>
```

Examples:
- `mlx/sprint-4-gpu-partition`
- `mlx/sprint-6-ci-and-infra`

**Creating a sprint branch:**

```bash
git checkout master
git pull origin master
git checkout -b mlx/sprint-7-my-topic
```

**Merging to master** (after QA and MLOps sign-off):

```bash
git checkout master
git merge --no-ff mlx/sprint-7-my-topic
git push origin master
```

The `--no-ff` flag preserves the sprint branch as a distinct unit in the merge history.

**Push target**: Always push sprint branches to `origin` (`RR-AMATOK/catboost-mlx`). Never push to `upstream` (`catboost/catboost`). Two remotes exist in this repository:

```bash
git remote -v
# origin   git@github.com:RR-AMATOK/catboost-mlx.git  (push to this one)
# upstream https://github.com/catboost/catboost.git    (never push here)
```

See DEC-002 in `.claude/state/DECISIONS.md` for the rationale behind this policy.

---

## Commit Format

All commits follow this format:

```
[mlx] component: description
```

The `[mlx]` tag scopes the commit to the Metal backend. The `component` names the subsystem being changed.

**Common components:**

| Component | What it covers |
|---|---|
| `kernels` | `kernel_sources.h` — Metal kernel source strings |
| `histogram` | `methods/histogram.cpp` and dispatch logic |
| `score` | `methods/score_calcer.cpp` — split scoring |
| `leaves` | `methods/leaves/leaf_estimator.cpp` |
| `tree` | `methods/tree_applier.cpp` |
| `searcher` | `methods/structure_searcher.cpp` |
| `boosting` | `methods/mlx_boosting.cpp` — training loop |
| `train` | `train_lib/train.cpp` — CatBoost trainer registration |
| `gpu_data` | `gpu_data/` — data structures and compressed index |
| `python` | `python/` — Python bindings |
| `tests` | `catboost/mlx/tests/` |
| `ci` | `.github/workflows/mlx_test.yaml` |
| `docs` | Documentation files |

**Examples:**

```
[mlx] kernels: fix non-deterministic CAS accumulation in histogram kernel
[mlx] score: add binToFeature lookup table for O(1) feature identification
[mlx] python: normalize loss param syntax before binary call
[mlx] docs: add ARCHITECTURE.md for Metal kernel deep-dive
```

---

## Development Protocol

New implementations follow a five-phase harness to reduce defects and ensure the algorithm matches CatBoost semantics.

### Phase 1: PROPOSE

State the problem and proposed solution clearly:
- What CatBoost behavior are we porting or fixing?
- Which source file is the reference (CPU `catboost/private/libs/algo/` or CUDA `catboost/cuda/`)?
- What are the inputs, outputs, and correctness criteria?

### Phase 2: CRITIQUE

Before writing code, self-critique the proposal:
- **Correctness**: Does the algorithm match CatBoost's semantics exactly? Check edge cases: empty partitions, single-feature datasets, depth 0.
- **Numerical stability**: Float32 is used throughout (ADR-005). Are there precision risks in the accumulation pattern?
- **Memory layout**: Does the data layout match what the Metal kernel expects? Check strides, row-major vs column-major, and buffer reuse.
- **GPU semantics**: Does the parallelization strategy avoid data races? Is the addition order deterministic across dispatches?
- **Performance**: Is the threadgroup size appropriate for the Apple Silicon SIMD width (32 threads)? Are reads coalesced?

### Phase 3: IMPLEMENT

Write the code:
- Match CatBoost's C++ style (see [Code Style](#code-style) below).
- Include inline comments for GPU-specific decisions (threadgroup sizes, barrier placement, atomic vs non-atomic outputs).
- Kernel source strings go in `kernel_sources.h`; dispatch wrappers go in the appropriate `methods/` file.
- If the change affects `csv_train.cpp`, it must also be applied to the library path — and vice versa (see [Two Code Paths](ARCHITECTURE.md#two-code-paths) in `ARCHITECTURE.md`).

### Phase 4: VERIFY

Write or run tests that cover:
- Correctness against the CPU reference (identical outputs for identical inputs).
- Numerical precision: compare GPU output to CPU reference with an appropriate tolerance.
- Edge cases identified in Phase 2.
- Performance: run `bench_boosting` before and after if the change touches a hot path.

### Phase 5: REFLECT

Before opening a PR, document:
- What could go wrong in production or at scale?
- What assumptions might break with large datasets (many features, many partitions)?
- Any technical debt or known limitations introduced by the change.

---

## Code Style

### C++ conventions

- **Classes**: `PascalCase` — `TMLXDataSet`, `THistogramResult`
- **Methods and functions**: `camelCase` — `ComputeHistograms`, `findBestSplit`
- **Type names**: `T` prefix — `TCFeature`, `TBoostingConfig`
- **Header guards**: `#pragma once`
- **Macros**: Use CatBoost's `Y_` macros where appropriate (`Y_UNUSED`, `CB_ENSURE`)
- **Namespaces**: All MLX backend code lives in `namespace NCatboostMlx`

### Metal shader conventions

- Use `[[kernel]]` attribute for compute kernels.
- Name kernels descriptively after what they compute: `histogram_one_byte_features`, `suffix_sum_histogram`, `score_splits_lookup`, `apply_oblivious_tree`, `leaf_accumulate`.
- Use `threadgroup` memory only for shared reduction staging — private per-thread arrays go on the thread's register file (or spill to device memory).
- Prefer `float` over `half` for all accumulation. Histogram and gradient values require float32 precision (ADR-005).
- Document atomics with a comment explaining why they are necessary at each use site.
- Place `threadgroup_barrier(mem_flags::mem_threadgroup)` between every write-then-read pair across threads.
- Kernel source strings live in `kernel_sources.h` inside `namespace NCatboostMlx::KernelSources`. Header strings (constants and includes) and body strings (the kernel body) are separate so MLX can insert the generated function signature between them.

### File organization

```
catboost/mlx/
  kernels/              # kernel_sources.h — all Metal kernel source strings
  gpu_data/             # data structures, compressed index, dataset wrapper
  methods/              # training algorithm components
    leaves/             # leaf value estimation
  targets/              # loss functions (RMSE, Logloss, MultiClass, MAE, etc.)
  train_lib/            # CatBoost trainer registration, model export
  tests/                # standalone binaries and unit tests
```

---

## State Files

The agent team uses a set of shared state files in `.claude/state/` to coordinate across sessions and maintain project memory. These files are committed to version control.

| File | Purpose | Who writes |
|---|---|---|
| `HANDOFF.md` | Current session state: what was just done, what comes next. First file read at session start; last file updated at session end. | Last active agent |
| `TODOS.md` | Active task tracker with status flags and acceptance criteria. Tracks in-progress, blocked, and completed tasks across sprints. | ML Product Owner creates; all agents update |
| `MEMORY.md` | Accumulated project knowledge: gotchas, algorithm discoveries, non-obvious behavior, lessons from bugs. | Any agent that discovers something worth persisting |
| `DECISIONS.md` | Operational decisions (DEC series): process decisions, parameter conventions, data format choices. For architecture decisions see `docs/decisions.md` (ADR series). | Research Scientist, Architect, ML Product Owner |
| `CHANGELOG-DEV.md` | What changed each session: sprint summaries, merged PRs, test results. | All agents after completing work |

When working in a new session:

1. Read `HANDOFF.md` to understand where the previous session ended.
2. Read `TODOS.md` to find the current active task.
3. Check `MEMORY.md` for relevant gotchas before starting implementation.
4. Update `TODOS.md` and `CHANGELOG-DEV.md` after completing work.
5. Update `HANDOFF.md` last, describing the current state for the next session.
