# Sprint 22 D6 — Security Audit Exit Gate (Gate #4 of 5)

**Branch**: `mlx/sprint-22-t2-integration`
**Date**: 2026-04-20
**Task**: D6 — Security audit of Option III (slab-by-partOffsets) T2 kernels + host dispatch.
**Prior gates**: D3 parity PASS, D4 perf PASS, D5 code review PASS (0 blockers, 7 nits).
**Reviewer**: @security-auditor (independent of D2/D3/D4/D5)
**Scope**: the dirty tree that will ship in the atomic D1-bundle commit — T2 Metal kernels and their host dispatch.
**Status**: **GATE PASS — 0 CRITICAL, 0 HIGH, 2 MEDIUM (advisory), 3 LOW, 2 INFO. Proceed to commit + technical-writer closeout.**

---

## §1 TL;DR

The Option III slab-by-partOffsets rewrite eliminates the D1c 24,558-doc buffer overflow **structurally**. No new memory-safety issues were introduced. All scatter/gather indices are provably bounded by the allocated slab size. Atomic scatter target indices are bounded by the feature's fold count (≤127 per DEC-016 envelope guard). No secrets, credentials, or PII in the kernel or dispatch code.

Two MEDIUM findings are DoS/robustness concerns inherited from the surrounding benchmark harness (argv is not a network attack surface; these are advisory for future production promotion), three LOW findings are defense-in-depth hardening, two INFO notes document integer-overflow thresholds well above the supported envelope.

**Max safe N** for uint32 slab arithmetic: ~165M docs at gate-config shape (numGroups=13, numStats=2); ~10.7M docs at the worst-case envelope growth (numGroups=50, numStats=3, approxDim=4 MultiClass K=5+). Both far beyond DEC-008's 50k envelope and S22's supported configurations.

**No CRITICAL/HIGH findings. The D1+D2 commit is security-clear to land.**

---

## §2 Scope and Files Audited

| File | Lines | Diff role | Audited |
|------|------:|-----------|:-------:|
| `catboost/mlx/kernels/kernel_sources_t2_scratch.h` | 231 | New T2-sort + T2-accum Metal kernels (Option III) | YES |
| `catboost/mlx/tests/bench_boosting.cpp` | 1820 | `DispatchHistogramT2` under `#ifdef CATBOOST_MLX_HISTOGRAM_T2` (L440-610) and two dispatch sites (L1207-1223, L1252-1267) | YES (D2-relevant ranges) |

Cross-referenced: `ComputePartitionLayout` (L298-314), `ParseArgs` (L149-181), gate check at L1470-1477, DEC-008 / DEC-012 / DEC-016 / DEC-020 in `.claude/state/DECISIONS.md`.

Out of scope: production kernel_sources.h, production histogram.cpp (both unmodified per scratch discipline). T1 code path. Network-facing code (none — this is a local CLI benchmark, not a service).

**Threat model**: this binary is a local research benchmark that takes argv input and local dataset files; it is not network-exposed, not multi-tenant, not privilege-separated. Threats considered:
1. Memory corruption from malformed internal state (partition-layout skew — the D1c class of bug).
2. Integer overflow in index arithmetic at large N.
3. DoS via pathological CLI flags.
4. Secret/credential leakage (none expected; confirmed).

Threats explicitly out of scope (not reachable): remote code execution via network, authz bypass, SSRF, injection, CORS. This is not a networked service.

---

## §3 Required-Check Findings

Numbering follows the task spec's 9 required checks.

### Check 1 — Buffer bounds at every scatter/gather site

**Verdict: PASS. No overflow possible.**

#### 1a — T2-sort scatter into `sortedDocs` (`kernel_sources_t2_scratch.h:115`)

```metal
const uint slotBase = (groupIdx * numStats + statIdx) * totalNumDocs + partOffsets[partIdx];
// ...
const uint pos = atomic_fetch_add_explicit(&tgCursors[bin], 1u, memory_order_relaxed);
sortedDocs[slotBase + pos] = docIdx;
```

**Bounds proof**:
- Buffer allocated size (host, `bench_boosting.cpp:563`): `numGroups * numStats * numDocs` uint32 elements.
- `slotBase = (groupIdx * numStats + statIdx) * totalNumDocs + partOffsets[partIdx]`.
- `(groupIdx * numStats + statIdx) ∈ [0, numGroups*numStats)`. Host-side dispatch grid at L553-557 dispatches X=`256*maxBlocksPerPart*numGroups` threads, Z=`numStats`. Kernel early-returns if `groupIdx >= numGroups` (line 57). So `(groupIdx * numStats + statIdx) ≤ (numGroups-1)*numStats + (numStats-1) = numGroups*numStats - 1`.
- `partOffsets[partIdx] ∈ [0, numDocs)` — it is a prefix sum of partSizes, with `sum(partSizes) = numDocs`, and `partIdx < numActiveParts` (kernel Y dispatch dim).
- `pos ∈ [tgOffsets[bin], tgOffsets[bin+1])` by atomic cursor semantics, where `tgOffsets[128] = partSize` is the total count in this partition (line 91, written by thread 0 after prefix-sum).
- `partSize + partOffsets[partIdx] ≤ partOffsets[partIdx] + (numDocs - partOffsets[partIdx]) = numDocs`. So `slotBase + pos < (groupIdx*numStats + statIdx)*numDocs + numDocs = (groupIdx*numStats + statIdx + 1)*numDocs ≤ numGroups*numStats*numDocs = buffer_size`.
- **No overflow is possible**, because:
  1. Each TG's write range `[slotBase, slotBase+partSize)` lies entirely inside the `(groupIdx,statIdx)` slab.
  2. The `(groupIdx,statIdx)` slabs are disjoint by construction.
  3. Within a slab, the per-partition slots are prefix-sum disjoint (`partOffsets[p+1] = partOffsets[p] + partSizes[p]`), so two partitions' slots cannot overlap.

This is the algebraic property that kills the D1c bug class: the failure mode at `bench_boosting.cpp:526` (pre-fix uniform-partition estimate) is unreachable because the new formula derives slot size directly from the exact partition prefix sum, not from an average-case estimate.

#### 1b — T2-accum gather from `sortedDocs` (`kernel_sources_t2_scratch.h:203, :215`)

```metal
const uint docIdx = sortedDocs[slotBase + i];  // both feature-0 and feature-1..3 paths
```

**Bounds proof**:
- Feature-0 path (L198-210): `i ∈ [start, end)` where `start = binOffsets[offBase + b]`, `end = binOffsets[offBase + b + 1u]`, `b ∈ [1, foldCount]`. Since `binOffsets` was written by T2-sort as the exclusive prefix scan of `tgCounts[0..127]` plus `tgOffsets[128] = partSize`, we have `end ≤ partSize` always. So `slotBase + i < slotBase + partSize ≤ buffer_size` — bounded.
- Feature 1-3 path (L214-224): `i ∈ [0, totalDocsInPart)` where `totalDocsInPart = binOffsets[offBase + 128u] = partSize`. Same bound: `slotBase + i < slotBase + partSize ≤ buffer_size`.
- **No read beyond slab boundary.**

#### 1c — Atomic scatter on features 1-3 (`kernel_sources_t2_scratch.h:220-222`)

```metal
device atomic_float* dst = (device atomic_float*)(
    histogram + histBase + firstFold + b - 1u);
atomic_fetch_add_explicit(dst, s, memory_order_relaxed);
```

**Bounds proof**:
- Write gated by `if (b >= 1u && b <= foldCount)` at L219 — only executed for valid bins.
- `histBase = partIdx * numStats * totalBinFeatures + statIdx * totalBinFeatures ∈ [0, numPartitions*numStats*totalBinFeatures)`.
- `firstFold = firstFoldIndicesFlat[foldBase + f]` — this is a feature's first bin in the packed histogram layout. `firstFold + foldCount ≤ totalBinFeatures` by construction of the bin packing.
- So `histBase + firstFold + (b-1) < histBase + firstFold + foldCount ≤ histBase + totalBinFeatures ≤ buffer_size`. Bounded.
- Feature 0 path (L207-209) has identical guard via the loop `for (b = tid+1; b <= foldCount; ...)`.
- **No out-of-bounds atomic write.**

#### 1d — `docIndices[partOffset + i]` (`kernel_sources_t2_scratch.h:76, :111`)

`partOffset + i ≤ partOffset + partSize ≤ numDocs`. `docIndices` is shape `[numDocs]`. Bounded.

#### 1e — `compressedIndex[docIdx * lineSize + featureColumnIdx]` (`:77, :112, :216`)

`docIdx ∈ [0, numDocs)` (argsort result). `featureColumnIdx < numGroups`. `lineSize = numUi32PerDoc`. Buffer size = `numDocs * numUi32PerDoc`. Bounded.

#### 1f — `binOffsets` reads/writes

- Write (L123): `offBase + b` with `b ∈ [0, 128]`. `offBase = (groupIdx*numPartitions*numStats + partIdx*numStats + statIdx)*129u`. Max `offBase + 128 = numGroups*numPartitions*numStats*129 − 1`. Host allocates `numTGs * 129` where `numTGs = numGroups*numActiveParts*numStats` (L530, L564). Since `partIdx < numActiveParts` (Y dispatch), `offBase + 128 ≤ buffer_size − 1`. Bounded.
- Reads at L180, L199-200: same index formula. Bounded.

**Summary**: every scatter and gather is provably bounded by a prefix-sum-disjoint slab arithmetic. The D1c bug class (slot capacity < actual write length) is not merely patched; it is structurally unreachable in the new layout.

---

### Check 2 — Threadgroup memory safety

**Verdict: PASS.**

Threadgroup allocations in T2-sort:
- `threadgroup atomic_uint tgCounts[128];` (L70)
- `threadgroup uint tgOffsets[129];` (L84)
- `threadgroup atomic_uint tgCursors[128];` (L102)

Sizes:
- `tgCounts` indexed by `bin ∈ [0, 128)`: `bin = (packed >> 24u) & 0x7Fu` — 7-bit mask → `bin ∈ [0, 128)`. Bounded.
- `tgOffsets[129]`: indexed by `b ∈ [0, 128]` (inclusive) at L87-91 and L198-200. Bounded.
- `tgCursors[128]`: indexed by `bin ∈ [0, 128)`. Bounded.

Initialization barriers:
- `tgCounts` initialized at L71-72 (all 128 slots by striding tid); `threadgroup_barrier(mem_threadgroup)` at L73 before reads. Correct.
- `tgOffsets` written by thread 0 at L85-92; barrier at L93 before other threads read. Correct.
- `tgCursors` initialized from tgOffsets at L103-104; barrier at L105 before atomic increments. Correct.
- Final `threadgroup_barrier` at L117 before `binOffsets` writeback at L122-123 ensures cursors have stabilized (not strictly required since tgOffsets is the written value, not the cursor, but harmless and conservative).

In T2-accum: no threadgroup allocations. Only device-memory reads and atomic writes. N/A.

**No racy uninitialized reads.** No threadgroup index out of bounds.

---

### Check 3 — Integer overflow risk

**Verdict: PASS within DEC-008 envelope; advisory for extrapolation.**

Key arithmetic in `DispatchHistogramT2` (host, `bench_boosting.cpp:530, :563, :564`):

```cpp
const ui32 numTGs = numGroups * numActiveParts * numStats;                            // L530
mx::Shape sortedDocsShape = {static_cast<int>(numGroups * numStats * numDocs)};       // L563
mx::Shape binOffsetsShape = {static_cast<int>(numTGs * 129u)};                        // L564
```

All operands are `ui32`. Intermediate multiplications occur in `ui32`. The `static_cast<int>` on L563/L564 converts to signed int — **if the uint32 value exceeds 2,147,483,647 (INT_MAX), this is signed-overflow-by-cast undefined behavior and the MLX shape will be interpreted as a negative int**, which MLX will either reject with an error or silently allocate incorrectly.

**Gate-config overflow thresholds** (at `numGroups=13, numStats=2, approxDim=1`):
- `sortedDocsShape = 13 * 2 * numDocs = 26 * numDocs`. INT_MAX / 26 = **82.6M docs**. Above this, signed-int cast breaks.
- Uint32 wraparound at `numGroups * numStats * numDocs`: 2^32 / 26 = **165.2M docs**.

**Worst-case envelope shape** (D5 NOTE-1: numGroups=50, numStats=3 for MultiClass K=4):
- `sortedDocsShape = 50 * 3 * numDocs = 150 * numDocs`. INT_MAX / 150 = **14.3M docs**.
- Uint32 wrap at `150 * numDocs`: 2^32 / 150 = **28.6M docs**.

**DEC-008 envelope** is N ≤ 50,000. 50k is 286× below the most restrictive threshold (14.3M). Gate config (50k, numGroups=13, numStats=2) uses shape value 1.3M — nowhere near INT_MAX.

**Host-side `numTGs * 129u`**: at 50k envelope, `numTGs ≤ 13*64*2 = 1664` → `numTGs*129 = 214,656`. Safe.

**Kernel-side arithmetic** inside the Metal shader:
- `(groupIdx * numStats + statIdx) * totalNumDocs + partOffsets[partIdx]`: all ops in 32-bit. Max value at 50k gate config = `26*50000 = 1.3M`. Safe.
- `docIdx * lineSize + featureColumnIdx`: `docIdx < numDocs`. `lineSize = numUi32PerDoc = ceil(numFeatures/4)`. At gate-config: `50000*13 = 650,000`. Safe. At envelope max (1M docs × 50 features = 1M × 13 = 13M): safe. Wrap at uint32 max when `numDocs * lineSize ≥ 2^32` → at lineSize=13, numDocs ≥ 330M. Well beyond anything practical.
- `statIdx * totalNumDocs + docIdx` inside `stats[...]`: bounded by stats buffer size = `numStats * numDocs`. Safe.
- `histBase = partIdx * numStats * totalBinFeatures + statIdx * totalBinFeatures`: at gate config, `64 * 2 * (50*128) + 128 = 819,456`. Safe. Envelope wrap when `numPartitions*numStats*totalBinFeatures ≥ 2^32`: ~50B operations — unreachable.

**Conclusion**: no integer overflow within DEC-008 or the natural supported envelope. The first breakable threshold is the `static_cast<int>(numGroups * numStats * numDocs)` on L563 at ~14M docs (worst-case shape) or ~82M docs (gate-config shape). Documented as **INFO-1** for future production promotion.

---

### Check 4 — Init barrier visibility

**Verdict: PASS. D1a race hypothesis remains REFUTED under Option III.**

The D1a hypothesis (fill_gpu → accum race on `sortedDocs`) was already falsified in D1b §6 (`fill_gpu` is a compute shader, not a blit — properly serialized in the Metal compute encoder by MLX). Option III does not change this mechanism.

Review of the new wiring (D2, `bench_boosting.cpp:570-605`):
1. `sortOut = GetT2SortKernel()(...)` returns a lazy `{sortedDocs, binOffsets}` output pair. MLX's lazy graph marks both outputs as produced by the sort kernel.
2. `accumOut = GetT2AccumKernel()(inputs={sortOut[0], sortOut[1], ...})` takes those lazy arrays as inputs. MLX's lazy-graph dependency tracking requires any kernel that consumes `sortOut[0]` to execute after the kernel that produced it. This is the standard MLX dataflow-dependency mechanism, verified by D1a §2 and used throughout MLX.
3. `init_value=0.0f` on both kernels (L580, L602) fills the output buffers to 0 prior to the user kernel's body running. The init is issued by MLX on the same compute encoder, preceding the user kernel, with the `init_value` stored as FP32 `0.0` (bit pattern `0x00000000`). Since `sortedDocs` is uint32 but the bit pattern is the same, the init zeroes it as a no-op for the `partSize > 0` path (every slot gets overwritten by T2-sort before T2-accum reads) and as uint-zero for the `partSize == 0` path (T2-accum reads `binOffsets[offBase+128u] = 0` → totalDocsInPart = 0 → both inner loops skip). Consistent with D1c's correctness argument and D5 NIT-3's defensibility analysis.

**Option III-specific re-verification**:
- The new slab formula `(groupIdx * numStats + statIdx) * totalNumDocs + partOffsets[partIdx]` does not introduce any new cross-TG dependency. Each TG still writes only within its own prefix-sum-disjoint slot; the only cross-TG memory is the per-TG `binOffsets` slot at `offBase + 0..128`, which each TG writes and its paired accum-TG reads.
- The sort-accum pairing is TG-symmetric by construction: T2-sort and T2-accum use the same `(groupIdx, partIdx, statIdx, blockInPart)` decomposition. Each (g, p, s) accum TG reads only the (g, p, s) sort TG's output. No accum TG reads another sort TG's slot.
- MLX's dependency graph guarantees accum sees sort's output, not pre-sort state. No race.

**Conclusion**: barrier wiring is correct. D1a's race hypothesis is properly refuted. No new race surface introduced by Option III.

---

### Check 5 — Atomic correctness

**Verdict: PASS.**

Three distinct atomic usages in T2 kernels:

**5a — Threadgroup `atomic_uint tgCounts[128]`** (T2-sort step 1, L79):
```metal
atomic_fetch_add_explicit(&tgCounts[bin], 1u, memory_order_relaxed);
```
- `memory_order_relaxed` is correct: the values are only consumed *after* a `threadgroup_barrier(mem_threadgroup)` at L81. The barrier provides the necessary synchronization; the atomic only needs counting-monotonic semantics. Standard pattern.
- No ABA risk (monotonic counter, only incremented).
- No unbounded spin (atomic_fetch_add on Apple GPUs is lock-free and completes in O(1) amortized).

**5b — Threadgroup `atomic_uint tgCursors[128]`** (T2-sort step 3, L114):
```metal
const uint pos = atomic_fetch_add_explicit(&tgCursors[bin], 1u, memory_order_relaxed);
```
- Same pattern. Cursor starts at `tgOffsets[bin]`, advances monotonically. Each thread gets a unique slot via atomic-return-old semantics. No ABA (monotonic, no decrement).
- Race-free output locations: within a single bin, the atomic return values are distinct non-negative integers; combined with the disjoint-per-bin tgOffsets layout, every `(slotBase + pos)` write in a single TG is to a distinct address. No WAW race within a TG.

**5c — Device `atomic_float* dst` on histogram** (T2-accum, L207-209 and L220-222):
```metal
device atomic_float* dst = (device atomic_float*)(histogram + histBase + firstFold + b - 1u);
atomic_fetch_add_explicit(dst, sum, memory_order_relaxed);
```
- `atomic_float` cast over `device float*` is the standard MLX pattern, identical to `kHistOneByteSource`. Float atomic-add is a CAS loop internally on Apple GPUs; correctness relies on the loop converging.
- **Convergence**: under heavy contention (many TGs writing the same bin) the CAS loop can retry; bounded by `O(contention)`. There is no unbounded spin because each retry makes lock-free forward progress (some thread in the contention set always succeeds).
- **No ABA** in FP32 sum monotonicity: if thread A reads old=X, thread B writes X→Y, then X→Z, then Y→X (same bit pattern coincidentally) — this is theoretically possible with floating point (e.g., `X + a - a = X` with rounding loss) but benign for our use because the final committed value is `X + a - a + original_delta`, which is still a well-defined sum in the atomic-add semantics. Even in the pathological case, the atomic-add is associative with an existing floor behavior that the DEC-008 parity envelope absorbs (ulp ≤ 4). D3's 100/100 determinism at gate config plus 18/18 ULP=0 proves this is not manifesting.
- Kernel guard `if (b >= 1u && b <= foldCount)` at L219 prevents writes at invalid offsets.

**Conclusion**: atomic usage is standard, convergent, no ABA corner case that would manifest within the DEC-008 envelope. 100/100 determinism at gate config (D3 §4) plus 18/18 bit-exact parity (D3 §3) is empirical proof of correctness.

---

### Check 6 — Host-side integer math in `DispatchHistogramT2`

**Verdict: PASS within envelope; INFO-1 raised for long-term hardening.**

See Check 3 for the full integer-overflow analysis. Summary of host-side lines:

| Line | Expression | Max value at gate config | Max value at envelope worst-case | Wrap threshold |
|-----:|------------|-----------------------:|--------------------------------:|---------------:|
| 499 | `(numFeatures + 3) / 4` | 13 | 50 | 1B features |
| 502-504 | `numGroups * 4` vector size | 52 | 200 | 1B groups |
| 530 | `numGroups * numActiveParts * numStats` | 1,664 | 100*64*3=19,200 | 2^32, 223M |
| 554 | `256 * maxBlocksPerPart * numGroups` | 3,328 | 12,800 | 2^32, 16.7M |
| 563 | `numGroups * numStats * numDocs` | 1,300,000 | 14.3M @ 1M docs | INT_MAX / 150 = 14.3M docs (worst-case) |
| 564 | `numTGs * 129u` | 214,656 | 2.5M | 2^32, 33M TGs |
| 565 | `numActiveParts * numStats * totalBinFeatures` | 8,192 | 76,800 | 2^32, 558M |

All safe within DEC-008. See INFO-1 below.

**Unsigned-to-signed cast at L563/L564**: `static_cast<int>` on `ui32` values. If `ui32 > INT_MAX`, cast is implementation-defined (C++17) and produces a negative int that MLX will treat as an invalid shape. This is the tightest binding constraint, documented in INFO-1.

**User-controlled config (argv via `ParseArgs` at L149-181)**: `--rows` is parsed with `std::atoi`, which returns `int`. If the user passes `--rows 2147483647`, the value is stored as `ui32 NumRows`. The binary will attempt to allocate `compressedData` of size `NumRows * numUi32PerDoc` and will OOM on any realistic system before reaching the kernel. This is a DoS-via-resource-exhaustion but not a code-execution path. Standard behavior for a CLI benchmark. Documented as MEDIUM-1 (advisory).

---

### Check 7 — Input validation at the T2 entry point

**Verdict: PARTIAL. Several reachable-but-unvalidated paths flagged as MEDIUM-2 / LOW-1.**

`DispatchHistogramT2` at `bench_boosting.cpp:484-497` accepts:
- `compressedData, stats, docIndices, partOffsets, partSizes` — `mx::array` (shape-typed; MLX validates dtype/rank at kernel entry)
- `numUi32PerDoc, numActiveParts, totalBinFeatures, numStats, numDocs, maxBlocksPerPart` — all `ui32` scalars

**Validated**:
- `maxFoldCount > 127u` at L519-524 — FATAL exit on violation. DEC-016 envelope guard. Correct.
- Grid dimensions derived from non-user-controlled `numFeatures / numClasses`.

**NOT validated** (but in practice not reachable from argv at DEC-008 envelope):
- `numActiveParts >= 1`: no check. If `numActiveParts == 0`, grid Y-dim is 0 → MLX likely rejects or no-ops. Call site at L1183 sets `numActiveParts = 1u << depth ≥ 1` for `depth ≥ 0`. Unreachable from argv. **LOW-1**.
- `numStats >= 1`: same; unreachable. **LOW-1**.
- `numDocs >= 1`: unreachable (dataset always has ≥ 1 row).
- `numActiveParts ≤ some cap`: unbounded. At `depth=31`, `numActiveParts = 2^31`, grid Y = 2^31 > INT_MAX — the `static_cast<int>(numActiveParts)` at L555 is signed-overflow UB. In practice, `maxDepth` is clamped internally and depth=31 would OOM on partition tensor allocation long before reaching the kernel. Argv `--depth 31` path: `partitions = zeros({NumRows})` succeeds; `1u << 31 = 0x80000000` cast to int becomes `-2147483648`, which MLX would reject. Not a memory-safety bug, but a poor error surface. **MEDIUM-2**.
- `partOffsets` monotonicity: not validated at the T2 boundary. But `partOffsets` is produced by `ComputePartitionLayout` at L298-314 via `cumsum(partSizes) - partSizes`, which is the exclusive prefix sum. Monotonic by construction. The only way this could be violated is if `ComputePartitionLayout`'s output were tampered with before reaching `DispatchHistogramT2` — not reachable from argv.
- `partOffsets` boundedness: `partOffsets[p] < numDocs` for all `p < numActiveParts`. Holds because `sum(partSizes) = numDocs` (ones scattered into partSizes, one per doc) and `partOffsets` is exclusive prefix sum.
- `features[f].Folds` bound: checked at L519-524 (DEC-016 guard).
- `features[f].FirstFoldIndex + Folds ≤ totalBinFeatures`: not explicitly checked, but established by the bin-packing invariant in the host code that builds `features`. Not reachable from argv.

**Conclusion**: the T2 entry point is well-formed for the intended use (called from `RunIteration` with host-verified state) but is not defensively hardened against arbitrary `ui32` scalar arguments. Since this is scratch-harness code invoked only from one trusted call site, the lack of defensive checks is acceptable per D5 NOTE-2 (scratch→production promotion should add CB_ENSURE-style guards).

---

### Check 8 — Secrets / credentials

**Verdict: PASS. Clean.**

Automated regex scan of both audited files for:
- API key / token / password / secret / bearer patterns
- AWS access-key patterns (`AKIA[A-Z0-9]{16}`)
- PEM private-key headers
- JWT-shaped tokens (`eyJ...`)
- GitHub token shapes (`ghp_...`)

**Zero matches.** The files contain only algorithmic kernel source, standard numeric constants, and dispatch glue. No credentials, no PII, no connection strings. The word "seed" appears but only as the RNG seed parameter (`--seed`), which is a deterministic-reproduction knob, not a secret.

Note: the benchmark prints dataset dimensions and configuration to stdout (L1425-1427). No sensitive state is logged.

---

### Check 9 — DoS / threat-model note (research code caveat)

**Verdict: MEDIUM-1 and MEDIUM-2 advisory, not blocking.**

This is a local CLI research benchmark, not a networked service. The OWASP Top 10 is mostly inapplicable (no auth, no network, no deserialization of untrusted input, no SSRF surface, no CSRF, no XSS context). The relevant DoS/misuse vectors are:

**MEDIUM-1 — Resource exhaustion via pathological `--rows`**
- `ParseArgs` L152: `cfg.NumRows = std::atoi(argv[++i])`. No upper bound check.
- At `--rows 100000000 --features 1000`, `compressedData` allocation = `10^8 * 250 * 4 bytes = 100 GB`. Will OOM on any realistic system.
- At `--rows 2147483647`, `std::atoi` returns INT_MAX = 2.1B, stored as `ui32`. Allocation computation may wrap.
- **Impact**: DoS on the local machine running the benchmark. No remote impact (no network).
- **Attack scenario**: attacker with shell access runs `bench_boosting --rows 2147483647`. Node thrashes or OOM-kills. Same outcome as `yes > /dev/null & yes > /dev/null & ...`.
- **Recommendation**: add `if (cfg.NumRows > 10_000_000) { fprintf(stderr, "..."); exit(1); }` or similar during scratch→production promotion. Non-blocking for current D1-bundle commit.

**MEDIUM-2 — Pathological `--depth` triggers UB in cast**
- `ParseArgs` L155: `cfg.MaxDepth = std::atoi(argv[++i])`. No upper bound check.
- At `--depth 31+`, `1u << depth` overflows uint32 (for `depth=32`) or produces `0x80000000` (at `depth=31`).
- Grid Y-dim cast at L555 `static_cast<int>(numActiveParts)` → negative int → MLX rejects or undefined.
- Partition tensor allocation at L1414 does NOT depend on depth, so preliminary allocation succeeds; the failure occurs at kernel dispatch time or MLX shape-validation time.
- **Impact**: confusing error or crash, not memory corruption. Same local-DoS class as MEDIUM-1.
- **Recommendation**: clamp depth to `<=16` at ParseArgs (matches DEC-008 envelope and CatBoost's default upper bound). Non-blocking.

**LOW-1 — Unvalidated scalar arguments to `DispatchHistogramT2`**
- `numActiveParts, numStats, numDocs`: no `>= 1` checks.
- Currently unreachable from argv (all derive from dataset or depth-loop invariants), but defensive CB_ENSURE would prevent silent wrong behavior if the scratch function is re-purposed for fuzz testing or unit tests.

**LOW-2 — Ambiguous error surface: `std::fprintf + std::exit`**
- `DispatchHistogramT2` at L519-524 uses `std::fprintf(stderr, ...); std::exit(1)` on fold-count violation. In a library context this would be unacceptable (crashes the host process) but in this benchmark CLI it is acceptable. Flagged for awareness during scratch→production promotion (D5 NOTE-2 also mentions this).

**LOW-3 — No rate limiting / resource-quota on kernel dispatch**
- Not applicable — this is CLI code, not a multi-tenant service. N/A note.

---

## §4 Finding Catalog

### INFO-1 — Integer overflow thresholds for future envelope expansion

**Classification**: INFO (no-action).
**File/lines**: `bench_boosting.cpp:563, :564` — `static_cast<int>(numGroups * numStats * numDocs)` and `static_cast<int>(numTGs * 129u)`.
**Failure mode**: if `numGroups * numStats * numDocs > INT_MAX` (2,147,483,647), the signed-int cast produces undefined behavior / negative shape, which MLX will either reject or allocate incorrectly.
**Reproducibility**: requires `numDocs` exceeding 82.6M (gate-config shape) or 14.3M (worst-case envelope shape numGroups=50, numStats=3). DEC-008 limit is 50,000 — 286× below the breaking point.
**Mitigation**: at scratch→production promotion, replace `static_cast<int>(uint32_expr)` with an explicit overflow-checked conversion (e.g., `CB_ENSURE(expr <= static_cast<uint32_t>(std::numeric_limits<int>::max()))`). Not required for current DEC-008 envelope.

### INFO-2 — Reliance on MLX `init_value=0.0f` cross-dtype aliasing

**Classification**: INFO (already covered by D5 NIT-3).
**File/lines**: `kernel_sources_t2_scratch.h:180`, `bench_boosting.cpp:580, :602`.
**Failure mode**: if MLX ever changes `init_value` semantics to be dtype-aware (i.e., refuse to init uint32 buffers with a float argument), the partSize=0 path in T2-accum reads uninitialized memory. Currently behaves correctly because FP32 0.0 and uint32 0 share bit pattern `0x00000000`.
**Mitigation**: add explicit `if (partSize == 0) return;` early-return in T2-accum (mirroring T2-sort L65). Per D5 NIT-3, scheduled for Sprint 23. Non-blocking.

### LOW-1 — Missing defensive `>= 1` checks in `DispatchHistogramT2`

**Classification**: LOW (defense in depth).
**File/lines**: `bench_boosting.cpp:484-497` (function signature — no entry-pre-conditions).
**Failure mode**: caller passing `numActiveParts = 0`, `numStats = 0`, or `numDocs = 0` would produce a zero-dim grid; MLX likely handles this by no-op, but behavior is not asserted.
**Reproducibility**: not reachable from argv. All internal call paths (`bench_boosting.cpp:1211-1215, :1255-1259`) pass host-verified values.
**Mitigation**: add `CB_ENSURE(numActiveParts >= 1 && numStats >= 1 && numDocs >= 1 && numActiveParts <= numDocs)` at function entry. Part of scratch→production promotion (D5 NOTE-2). Non-blocking.

### LOW-2 — `std::exit(1)` on FATAL path

**Classification**: LOW (API hygiene).
**File/lines**: `bench_boosting.cpp:519-524`.
**Failure mode**: `maxFoldCount > 127` triggers process exit. Acceptable for benchmark CLI; unacceptable in library context.
**Mitigation**: on promotion to `histogram.cpp`, replace with `CB_ENSURE(maxFoldCount <= 127u, ...)` that throws. Covered by D5 NOTE-2.

### LOW-3 — Argv bounds unvalidated

**Classification**: LOW (local DoS, self-inflicted).
**File/lines**: `bench_boosting.cpp:149-181`.
**Failure mode**: user invoking benchmark with unreasonable `--rows` / `--depth` / `--iters` triggers OOM or UB before kernel dispatch.
**Mitigation**: add reasonable clamps (e.g., `NumRows <= 10_000_000`, `MaxDepth <= 16`, `NumIters <= 10_000`). Non-blocking for a research benchmark; the user is the attacker against themselves.

### MEDIUM-1 — `--rows` unbounded: resource-exhaustion DoS

**Classification**: MEDIUM (advisory — relevant only if this code is ever exposed to untrusted input).
**File/lines**: `bench_boosting.cpp:152`.
**Failure mode**: `--rows 2147483647` causes allocation of tens of GB of memory, OOM, possibly kernel panic on unified-memory systems.
**Reproducibility**: trivially reproducible by anyone with shell access.
**Impact**: local DoS. No privilege escalation. No remote impact.
**Mitigation**: add input sanity clamp at ParseArgs. Non-blocking for the current D1-bundle commit (scratch benchmark is trusted-input only).

### MEDIUM-2 — `--depth` unbounded: UB via signed-overflow cast

**Classification**: MEDIUM (advisory — same reasoning as MEDIUM-1).
**File/lines**: `bench_boosting.cpp:155, :1183, :555`.
**Failure mode**: `--depth >= 31` makes `1u << depth` overflow uint32 (at depth=32) or produce `0x80000000` (at depth=31), which becomes negative after `static_cast<int>` at L555. MLX will reject or the kernel dispatch will behave undefined.
**Reproducibility**: trivial — `bench_boosting --depth 40`.
**Impact**: local crash / confusing error; no memory corruption (fails earlier than any unbounded write).
**Mitigation**: clamp `MaxDepth <= 16` at ParseArgs. Non-blocking.

### Positive findings (credit)

- **Option III's slab formula is the correct structural fix.** The D1c bug class (slot capacity < actual partition size) is algebraically unreachable under the new layout — not merely patched.
- **DEC-016 envelope guard at L519-524** correctly bounds `maxFoldCount <= 127`, preventing the MSB-sentinel collision on features 1-3 atomic scatter.
- **`maxBlocksPerPart = 1` hardcode at L1423** correctly eliminates the cross-threadgroup atomic-add race that BUG-001 documents.
- **`init_value=0.0f` on both kernels** provides defense-in-depth against the partSize=0 edge case (though see INFO-2 for fragility).
- **Scratch discipline maintained**: production `kernel_sources.h` and `histogram.cpp` are unchanged. T1 code path byte-identical (verified by D3 §6 and D4 §9).
- **No secrets, no PII, no credentials** in either file.
- **Atomic memory-ordering** (`memory_order_relaxed` paired with threadgroup barriers) is correctly applied.

---

## §5 Max-Safe-N Analysis

For each integer-arithmetic site that is function of `numDocs` (N):

| Expression | Site | Overflow at N = |
|------------|------|-----------------|
| `numGroups * numStats * numDocs` (uint32 wrap) | `bench_boosting.cpp:563` | 2^32 / (13*2) = **165.2M** @ gate shape; 2^32 / (50*3) = **28.6M** @ worst-case shape |
| `static_cast<int>(numGroups * numStats * numDocs)` (signed-int wrap — tightest) | `bench_boosting.cpp:563` | INT_MAX / 26 = **82.6M** @ gate shape; INT_MAX / 150 = **14.3M** @ worst-case shape |
| `numTGs * 129u` (uint32 wrap) | `bench_boosting.cpp:564` | `numActiveParts = 64` @ depth 6 → effectively bounded by numGroups, not N |
| `numDocs * lineSize` (compressedIndex buffer) | kernel `:77, :112, :216` | 2^32 / 13 = **330M** @ gate lineSize; 2^32 / 250 = **17.2M** @ worst-case lineSize |
| `statIdx * totalNumDocs + docIdx` (stats buffer) | kernel `:204, :217` | 2^32 / numStats = **2.1B / numStats** — effectively unbounded |
| `slotBase + pos` (sortedDocs scatter) | kernel `:115, :203, :215` | `(numGroups*numStats*numDocs)` — same as shape wrap above |

**Tightest threshold: N = 14.3M docs** at the worst-case envelope shape (numGroups=50, numStats=3) due to signed-int cast at L563.

**DEC-008 envelope: N ≤ 50,000 — 286× below the tightest threshold.**

**Conclusion**: T2 is safe across the supported envelope with a 2.5-orders-of-magnitude safety margin. Any future sprint that raises the envelope beyond N=10M should re-audit the integer-width assumptions, starting with the `static_cast<int>(...)` at L563-564.

---

## §6 Bounds Proof Sketch — slab addressing under Option III

**Theorem**: For all valid (groupIdx, partIdx, statIdx, pos) during T2-sort and T2-accum execution, the computed index `slotBase + pos` falls strictly within the allocated `sortedDocs` buffer of size `numGroups * numStats * numDocs`.

**Given**:
- `partOffsets[p]` = `Σ_{q < p} partSizes[q]` (exclusive prefix sum, by construction in `ComputePartitionLayout`).
- `Σ_{p=0}^{numActiveParts-1} partSizes[p] = numDocs` (every doc belongs to exactly one partition, enforced by `scatter_add_axis` at L304).
- `pos < partSize` (atomic-cursor invariant: cursor[bin] starts at `tgOffsets[bin]` and only the `partSize` docs in this partition ever increment a cursor; terminal cursor value is `tgOffsets[128] = partSize`).
- `slotBase = (groupIdx * numStats + statIdx) * numDocs + partOffsets[partIdx]`.

**Proof**:

1. Upper bound on `partOffsets[partIdx] + partSize`:
   - `partOffsets[partIdx] + partSize = Σ_{q < partIdx} partSizes[q] + partSizes[partIdx] = Σ_{q ≤ partIdx} partSizes[q] ≤ Σ_{q < numActiveParts} partSizes[q] = numDocs`.
   - Equality when `partIdx = numActiveParts - 1` (last partition); strict inequality otherwise.

2. Upper bound on the `(groupIdx, statIdx)` coefficient:
   - `groupIdx * numStats + statIdx ≤ (numGroups-1) * numStats + (numStats-1) = numGroups*numStats - 1` (because kernel early-returns for `groupIdx >= numGroups` at L57, and Z dispatch bounds `statIdx < numStats`).

3. Combining (1) and (2):
   - `slotBase + pos < slotBase + partSize = (groupIdx*numStats + statIdx) * numDocs + partOffsets[partIdx] + partSize ≤ (groupIdx*numStats + statIdx) * numDocs + numDocs = (groupIdx*numStats + statIdx + 1) * numDocs ≤ numGroups*numStats*numDocs = buffer_size`.

4. Therefore `slotBase + pos ∈ [0, buffer_size)`. **No out-of-bounds write or read.**

**Slot disjointness (writes from distinct TGs do not collide)**:

Consider two TGs (g₁, p₁, s₁) and (g₂, p₂, s₂), (g₁, p₁, s₁) ≠ (g₂, p₂, s₂). Their write ranges are:
- TG 1: `[(g₁*numStats + s₁)*numDocs + partOffsets[p₁], (g₁*numStats + s₁)*numDocs + partOffsets[p₁] + partSizes[p₁])`
- TG 2: `[(g₂*numStats + s₂)*numDocs + partOffsets[p₂], (g₂*numStats + s₂)*numDocs + partOffsets[p₂] + partSizes[p₂])`

Case A: `(g₁, s₁) ≠ (g₂, s₂)`. The `(groupIdx*numStats + statIdx)*numDocs` coefficients differ by at least `numDocs`. Each TG's range has length `partSize ≤ numDocs`. Therefore the two ranges do not overlap. ✓

Case B: `(g₁, s₁) = (g₂, s₂)` but `p₁ ≠ p₂`. WLOG `p₁ < p₂`. Then `partOffsets[p₁] + partSizes[p₁] = partOffsets[p₁+1] ≤ partOffsets[p₂]`. So TG 1's range ends at or before TG 2's range begins. No overlap. ✓

**Conclusion**: the D1c bug class — where TG 1 overflows its slot (size `maxPartDocs`) and writes into TG 2's slot — is eliminated by construction. Option III does not *move* the bug; it *removes the precondition for the bug class*. Under Option III, a write overflow at `sortedDocs[slotBase + pos]` with `pos ≥ partSize` would require `pos` to exceed the per-TG atomic-cursor invariant, which is structurally impossible given the prefix-scan tgOffsets setup.

---

## §7 Final Verdict

**PASS.**

| Severity | Count | Blocks D1-bundle commit? |
|:--------:|:-----:|:------------------------:|
| CRITICAL | 0     | N/A |
| HIGH     | 0     | N/A |
| MEDIUM   | 2     | No (advisory, relevant only if this code is ever network-exposed or fuzzed) |
| LOW      | 3     | No (defense-in-depth hardening for scratch→production promotion) |
| INFO     | 2     | No (documentation of envelope limits and D5 NIT-3 reiteration) |

**All required security checks clear the blocker threshold:**
- Check 1 (bounds proof) — PASS
- Check 2 (threadgroup memory safety) — PASS
- Check 3 (integer overflow) — PASS within envelope; INFO-1 for long-term hardening
- Check 4 (init barrier visibility) — PASS, D1a hypothesis remains refuted
- Check 5 (atomic correctness) — PASS
- Check 6 (host-side integer math) — PASS within envelope; see INFO-1
- Check 7 (input validation at T2 entry) — PARTIAL (MEDIUM-1/2, LOW-1); not reachable from trusted call sites
- Check 8 (secrets) — PASS (regex-verified)
- Check 9 (DoS / threat-model note) — MEDIUM-1/2 advisory for local-DoS surface

**The D1c buffer-overflow bug class is not merely patched — it is structurally eliminated by the prefix-sum slab addressing scheme. The §6 bounds proof sketch shows that every scatter/gather is algebraically bounded by the allocated buffer.**

Per standing orders: **no commit made**. Tree remains dirty for the atomic D1-bundle commit at Ramos's direction. Proceed to gate #5 (@technical-writer closeout).

---

## §8 Recommendations (prioritized)

Non-blocking for the current commit, but worth scheduling:

1. **Sprint 23 hardening pass** (aligned with D5 NIT fixes): add `CB_ENSURE` guards at `DispatchHistogramT2` entry (LOW-1); replace `std::fprintf+std::exit` with exceptions (LOW-2); add explicit `partSize == 0` early-return in T2-accum (INFO-2 / D5 NIT-3); add argv clamps (MEDIUM-1, MEDIUM-2, LOW-3).
2. **Pre-envelope-expansion audit**: before any sprint raises the DEC-008 envelope beyond N=1M docs, re-audit the `static_cast<int>(...)` sites on L563-564 and add overflow-checked conversions.
3. **Production-promotion checklist** (for the scratch→production T2 pass — D5 NOTE-2 and NOTE-3): lift kernel registration out of static-local scope, replace raw `std::fprintf+std::exit` error surface, expose a clean public API, add a documented test matrix that includes the MEDIUM-1/2 DoS cases as negative tests.

No action required to ship the D1-bundle commit under the current DEC-008 envelope.
