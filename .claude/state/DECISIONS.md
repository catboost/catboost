# Decision Log

<!-- Agents record design decisions here. Append-only — to reverse a decision, add a new entry. -->
<!-- Format: DEC-NNN sequential IDs. Status: Active / Superseded by DEC-XXX / Deprecated -->

> See also: `docs/decisions.md` for architecture decisions (ADR series).

## DEC-001 — Loss parameter syntax: normalize to positional form before binary call
- **Date:** 2026-04-09
- **Status:** Active
- **Decided by:** ml-engineer (Sprint 3 follow-up)
- **Context:** CatBoost's canonical loss string syntax uses named params (`Quantile:alpha=0.7`). The Python layer needs to accept both the named form (user-facing) and the positional form that `csv_train` expects. BUG-002 revealed that passing the named form directly to the binary caused a parse failure.
- **Decision:** Python `_build_train_args` always normalizes to positional form before constructing the CLI invocation. The named form is only used in `_validate_params` for range checking and user error messages. The binary never sees `param=value` keys.
- **Alternatives:** Teach `csv_train` to accept named params. Rejected: that would require C++ changes to the binary's arg parser and create a second divergence surface.
- **Trade-offs:** Slightly more logic in the Python layer; binary interface stays simple and consistent.

---

## DEC-003 — float32 bucket-count limit in GPU partition layout (RESOLVED)
- **Date:** 2026-04-09
- **Status:** Resolved (Sprint 9)
- **Decided by:** ml-engineer (Sprint 4 follow-up, Sprint 5 documented, Sprint 9 fixed)

**Original issue:** `ComputePartitionLayout` used `mx::scatter_add_axis` with
float32 accumulators to count docs per partition. float32 represents integers
exactly only up to 2^24 = 16,777,216.

**Resolution (Sprint 9):** Switched to int32 scatter_add_axis accumulator.
Int32 is exact for values up to 2^31 (~2.1B docs) — effectively unlimited
for any dataset that fits in Apple Silicon unified memory. The `CB_ENSURE`
guard was removed. MLX's `scatter_add_axis` works correctly with int32.

**Commits:** `8717ddd` (original float32 guard, Sprint 5), `TODO-026` commit (int32 fix, Sprint 9)

---

## DEC-004 — Depthwise grow policy: BFS node ordering and per-node split search
- **Date:** 2026-04-11
- **Status:** Active
- **Decided by:** ml-engineer (Sprint 9, TODO-031)
- **Context:** CatBoost's default "SymmetricTree" (oblivious tree) applies one split condition shared across all nodes at the same depth level, giving 2^d leaves where every leaf is reached by the same sequence of split predicates. This is efficient (one `FindBestSplitGPU` call per depth level) but expressively limited — a single bad split at level d forces every node at that level to split on the same feature. "Depthwise" (XGBoost-style) gives each node at level d its own best split, producing more expressive trees at the same depth.
- **Decision:** Implement `EGrowPolicy::Depthwise` as follows:
  1. **BFS node ordering.** Internal nodes are stored in a flat array indexed in BFS order: node 0 is the root; children of node n are at 2n+1 (left) and 2n+2 (right). This matches XGBoost's convention and makes BFS traversal in the Metal kernel a simple index computation with no pointer chasing.
  2. **Per-node split search.** `SearchDepthwiseTreeStructure` calls `FindBestSplitGPU` once per live node at each depth level (2^d calls at level d, totalling 2^depth − 1 calls per tree). Each call restricts its partition to the documents in that node. This reuses the existing split-search infrastructure without modification.
  3. **Leaf index compatibility.** The leaf index output from `kTreeApplyDepthwiseSource` uses the same BFS position convention: after traversing `depth` levels, `leafIdx = nodeIdx − numNodes` where `numNodes = 2^depth − 1`. The leaf range `[0, 2^depth)` matches `kLeafAccumSource` / `kLeafAccumChunkedSource` — leaf values are interchangeable between oblivious and depthwise paths.
  4. **Shared leaf estimation.** `ComputeLeafSumsGPU` and `ComputeLeafValues` are reused unchanged — they operate on `Partitions_` (the per-doc leaf assignment array), which both tree paths populate in the same format.
- **Alternatives:** BFS with a node→split pointer table (avoids per-node GPU dispatch). Rejected for Sprint 9 as premature — at depth ≤ 10, 2^d ≤ 1024 `FindBestSplitGPU` calls per iteration are acceptable, and the simpler design matches the existing split-search API exactly.
- **Trade-offs:** 2^d split-search dispatches per depth level vs 1 for SymmetricTree. At depth 6 that is 63 dispatches; at depth 10 it is 1023. Mitigated by histogram EvalNow deferral (Item G) which reduces per-call overhead.

---

## DEC-005 — Multi-pass leaf accumulation: 64-leaf chunks to cap private array at 5 KB
- **Date:** 2026-04-11
- **Status:** Active
- **Decided by:** ml-engineer (Sprint 9, TODO-030)
- **Context:** `kLeafAccumSource` pre-allocates a per-thread private array `float privSums[LEAF_PRIV_SIZE]` where `LEAF_PRIV_SIZE = MAX_APPROX_DIM * MAX_LEAVES * 2 = 10 * 64 * 2 = 1280 floats = 5 KB`. This is the maximum that avoids register spill on Apple Silicon's GPU. With `MAX_LEAVES = 64`, the kernel cannot support depth > 6 (2^6 = 64 leaves) without increasing the private array size and spilling to threadgroup memory — which would serialize access and eliminate the performance benefit.
- **Decision:** Keep `LEAF_PRIV_SIZE` fixed at 1280 floats (5 KB) by introducing a chunked multi-pass variant:
  1. `LEAF_CHUNK_SIZE = MAX_LEAVES = 64`. Each pass processes exactly 64 leaves (or fewer for the last chunk).
  2. `kLeafAccumChunkedSource` kernel accepts `chunkBase` and `chunkSize` scalars and accumulates only documents whose leaf index falls in `[chunkBase, chunkBase + chunkSize)`.
  3. `ComputeLeafSumsGPUMultiPass` in `leaf_estimator.cpp` issues `ceil(numLeaves / 64)` kernel dispatches, each with a different `chunkBase`, then copies chunk outputs into the full `[approxDim * numLeaves]` result arrays.
  4. Depth range extended from 1–6 to 1–10 (1024 leaves max).
- **Alternatives:** Increase `MAX_LEAVES` at the cost of register spill. Rejected: spilled registers on Apple Silicon map to threadgroup memory, serializing the accumulation loop and eliminating the performance advantage of per-thread private arrays. Chunked passes are cheap relative to the histogram build (which dominates iteration time), so the multi-pass overhead is negligible.
- **Trade-offs:** `ceil(numLeaves / 64)` passes vs 1 for depth ≤ 6. At depth 10 (1024 leaves) that is 16 passes. Each pass is a full Metal kernel dispatch, so there is kernel-launch overhead per pass. Measured to be acceptable relative to histogram build time.

---

## DEC-002 — Sprint branch policy: dedicated branches per sprint, push to origin (RR-AMATOK) only
- **Date:** 2026-04-09
- **Status:** Active
- **Decided by:** ml-product-owner (process decision, Sprint 3 retrospective); push-target constraint added 2026-04-09
- **Context:** Sprints 1–3 committed directly to master, which meant QA-found bugs (BUG-001, BUG-002) had to be fixed with follow-up commits on master rather than before merge. This creates noisy master history and risks shipping untested states. Two remotes exist: `origin` = `RR-AMATOK/catboost-mlx` (our fork), `upstream` = `catboost/catboost` (upstream). Sprint branches must never be pushed to `upstream`.
- **Decision:** Starting Sprint 4:
  1. Each sprint opens a branch named `mlx/sprint-<N>-<short-topic>` (e.g., `mlx/sprint-4-gpu-partition`).
  2. All sprint commits land on that branch. No direct commits to master during the sprint.
  3. Push target is `origin` (`RR-AMATOK/catboost-mlx`) exclusively. Never push sprint branches or force-push to `upstream`.
  4. Merge to `origin/master` happens only after QA sign-off (test suite green) and MLOps sign-off (no new unbounded sync regression), via a PR on `RR-AMATOK/catboost-mlx`.
  5. Sprints 1–3 are already on master and will not be rewritten.
- **Alternatives:** Keep committing directly to master with required CI passing. Rejected: CI does not currently include the full MLOps sync audit, so bugs like BUG-001/BUG-002 can still reach master before QA review.
- **Trade-offs:** Slightly more branch management overhead; master history is cleaner and every merged state is verified. PRs provide a natural review checkpoint for the agent team. The `origin`-only push rule prevents accidental pollution of the upstream CatBoost repo.
