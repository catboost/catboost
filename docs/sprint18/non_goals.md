# Sprint 18 Non-Goals

Items explicitly out of scope for Sprint 18. Each has a designated sprint or condition for pickup. Do not add these to Sprint 18 acceptance criteria or the Sprint 18 PR.

---

## L2 — Pre-permute stats + compressedIndex (gather removal)

**What it is:** A preprocessing pass that writes stats and packed features in `docIndices`-permuted order before histogram dispatch, removing the gather indirection at `kernel_sources.h:131–136`. Standalone headroom: approximately 2–4 ms on `histogram_ms` at depth ≥ 1.

**Why deferred:** L2's savings partially overlap L1's target — a coalesced-gather win inside an already-non-spilling accumulator is a second-order effect. The natural pairing is Sprint 19 alongside multiclass fusion: the reshuffled layout composes cleanly with per-dim fusion's grid geometry change, and their combined effect is worth a dedicated sprint rather than a drive-by.

**Target:** Sprint 19, as a sibling lever to L3.

---

## L3 — Multiclass per-dim dispatch fusion

**What it is:** Fuse the `approxDim` serial loop at `catboost/mlx/tests/csv_train.cpp:3185–3204`, which serialises 3 `DispatchHistogram()` calls for MultiClass, into a single kernel dispatch using the Z grid dimension. Expected gain: 15–25 ms on multiclass configs.

**Why deferred:** L3 moves multiclass wall-time only. It has **zero effect on the gate config** (N=10k, RMSE, depth=6, 128 bins). Per DEC-010, L3 is the Sprint 19 headline candidate once L1 lands and multiclass becomes the next plurality cost.

**Target:** Sprint 19. See `docs/sprint19/plan_prior.md` (drafted from S18 findings).

---

## `maxBlocksPerPart` retuning

**What it is:** `catboost/mlx/methods/histogram.cpp:105` hardcodes `maxBlocksPerPart = 1`. The production path (`csv_train.cpp:891–894`) already computes this dynamically. Deferred in Sprint 17 non-goals as the library path is dead code for `csv_train`.

**Why deferred:** Sprint 18 does not touch the library path. No performance impact on current production kernel. Cleanup and retuning belong with the broader library-path unification work.

**Target:** Sprint 19 (or as a standalone cleanup issue before that). Unchanged from Sprint 17 deferral.

---

## M1/M2 `simd_shuffle_xor` validation

**What it is:** Confirm tree-reduction correctness and performance on M1 and M2 chips. The Sprint 17 D1c design makes assumptions about SIMD width and `simd_shuffle_xor` behaviour that may differ subtly across Apple Silicon generations.

**Why deferred:** Sprint 17 R9 deferred this validation. Sprint 18 inherits the same deferral. Sprint 18 targets M3 exclusively; M1/M2 testing adds hardware dependency and test matrix complexity not warranted while the accumulator rewrite is still in progress. The 32 KB threadgroup-memory limit is uniform across M1/M2/M3 (Apple Metal Feature Set Tables), so L1a's ceiling risk is not M-chip-specific.

**Target:** Sprint 18 non-goal continues. Revisit after L1 ships on M3.

---

## Library-path histogram kernel cleanup

**What it is:** `catboost/mlx/methods/histogram.cpp` is a parallel histogram implementation on the library path. It is dead code for `csv_train` (the current production path) but exists as a second source of truth alongside `kernel_sources.h`.

**Why deferred:** Touching dead code adds blast radius with no measurable performance gain for Sprint 18. Unification is a separate task requiring coordination across the training loop, data-loading path, and Python bindings.

**Target:** Sprint 22+, as part of the library-path restoration or deletion decision.

---

## `derivatives_ms` iter-0 JIT warmup

**What it is:** The first iteration incurs a ~7.6 ms JIT compilation cost for the derivative kernel, visible as a spike in `derivatives_ms` at iter=0. Subsequent iterations are <1 ms.

**Why deferred:** The cost amortises to <1 ms per iteration over 50 iterations, which is below the threshold for Sprint 18 gate impact. `derivatives_ms` is 0.18 ms at steady state — not a Sprint 18 lever regardless of JIT behaviour.

**Target:** Not tracked for a specific sprint. Address if JIT warmup becomes material at fewer than 20 iterations or with a faster histogram making the derivative cost visible.

---

## CPU fallback threshold finalization

**What it is:** DEC-007 defers the threshold at which the MLX backend falls back to the CatBoost CPU path (small N, narrow feature counts). A principled threshold requires stable MLX kernel timing across the full N/feature/bin space.

**Why deferred:** Sprint 18 does not change the kernel's scaling properties; the fallback threshold re-evaluation requires Sprint 22–23 data per DEC-007.

**Target:** Sprint 22–23 per DEC-007.

---

## New benchmark infrastructure

**What it is:** Any new benchmarking scripts, CI jobs, or profiling harnesses beyond what was established in Sprint 17.

**Why out of scope:** `benchmarks/check_histogram_gate.py --18config` and `.github/workflows/mlx-perf-regression.yaml` are sufficient for Sprint 18 gates (S18-G5). The Sprint 18 CI gate change is a config update (baseline path, threshold), not new infrastructure. Introducing new infrastructure this sprint adds scope without moving the gate.

**Target:** Out of scope per S18-G5. Revisit if Sprint 19 introduces a new gate type (e.g., multiclass-only regression gate for L3).

---

## Upstream push to catboost/catboost

**What it is:** Any PR or push directed at the `catboost/catboost` upstream repository.

**Why out of scope:** DEC-004. All Sprint 18 work lands on `RR-AMATOK/catboost-mlx` only. The MLX backend is Apple Silicon-specific and not appropriate for the upstream CatBoost repository in its current form.

**Target:** Not planned. DEC-004 is standing policy for the duration of Operation Verstappen.

---

## "Document as limitation" or "revert and defer" framings

If L1 misses the S18-G1 gate, the path is: renegotiate the gate threshold with evidence, expand the ablation space, or extend the sprint by one cycle. Per standing project feedback ("fix properly always"), deferring an unresolved structural problem as a documented limitation is not an acceptable outcome for Sprint 18.
