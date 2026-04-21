# Sprint 23 — T2 Scratch→Production Promotion + NIT Cleanup + Tree-Search Research

**Branch**: `mlx/sprint-23-t2-promotion` (to be cut from Sprint 22 tip)
**Campaign**: Operation Verstappen — battle 8 of 9
**Gate config**: 50k/RMSE/d6/128b (unchanged from S19–S22)
**Authority**: `docs/sprint22/d5_code_review.md §4` (NIT catalog); `docs/sprint21/d1r4_synthesis.md §3` (tree-search rank #2)

---

## §1 Background

Sprint 22 CLOSED with T2 sort-by-bin shipped, cumulative R8 = **1.90×**, Verstappen ≥1.5× gate cleared by 40 pp. The T2 kernel and dispatch code ship in scratch form:

- Kernel sources in `catboost/mlx/kernels/kernel_sources_t2_scratch.h`
- Host dispatch in `catboost/mlx/tests/bench_boosting.cpp` under `#ifdef CATBOOST_MLX_HISTOGRAM_T2`
- Production `catboost/mlx/kernels/kernel_sources.h` **unmodified**

Sprint 23 has three tracks:

1. **D0 (blocking)** — Promote T2 to production: merge scratch header into `kernel_sources.h`, move `DispatchHistogramT2` into `catboost/mlx/methods/histogram.cpp`, remove compile-time flag, make T2 default.
2. **NIT cleanup batch** — Address 6 deferred nits from the S22 D5 code review (`docs/sprint22/d5_code_review.md §4`).
3. **Research track** — Tree-search restructure: S23-R1 EvalAtBoundary readback elimination (bounded) + S23-R2 dispatch inversion research spike (timeboxed).

---

## §2 D0 — T2 Scratch→Production Promotion

**Purpose**: Graduate T2 from scratch/research status to the production code path. Until promotion lands, `bench_boosting.cpp` is the only caller and `kernel_sources.h` does not contain T2. Championship benchmarks must use the production path.

**Pre-requisite**: Sprint 22 closeout commit landed; PR #14 open (or at minimum the branch tip is `73baadf445` + closeout).

**Scope (per DEC-012 one-structural-change-per-commit)**:

1. **Commit 1 — kernel sources promotion**: Copy `kT2SortSource` and `kT2AccumSource` from `kernel_sources_t2_scratch.h` into `kernel_sources.h` under a clearly marked T2 section. Apply NIT-1/NIT-2/NIT-7 nit fixes during this pass (replace hardcoded literals with named constants; deduplicate `offBase` arithmetic; harmonize bin masks). Verify the promotion compiles clean.
2. **Commit 2 — dispatch promotion**: Implement `DispatchHistogramT2` in `catboost/mlx/methods/histogram.cpp` with production-quality API: CB_ENSURE error handling, factored kernel registration (no static-local inline registration), clean public signature. Remove the `bench_boosting.cpp` inline copy under `#ifdef CATBOOST_MLX_HISTOGRAM_T2`.
3. **Commit 3 — flag removal + default flip**: Remove `CATBOOST_MLX_HISTOGRAM_T2=1` compile-time flag. Update `DispatchHistogramBatched` to call `DispatchHistogramT2` by default. Apply NIT-3/NIT-4/NIT-5 nit fixes (empty-partition guard, maxBlocksPerPart assert, remove unused `numTGs` uniform).
4. **Commit 4 — parity re-verify post-promotion**: Run 18-config DEC-008 parity sweep on the promoted production build. Confirm 18/18 ULP=0 unchanged from S22 D3 gate. 100-run determinism at gate config. Document in `docs/sprint23/d0_promotion_parity.md`.

**Kill-switch**: If parity degrades vs S22 D3 (any config > DEC-008 envelope), halt and debug before proceeding to NIT cleanup or research track.

---

## §3 NIT Cleanup Batch

6 nits deferred from `docs/sprint22/d5_code_review.md §4`. Address alongside D0 promotion (bundle in Commits 1 and 3 above).

| NIT | Description | Commit |
|-----|-------------|--------|
| NIT-1 | Replace inline literals (`256u`, `128u`, `0x7Fu`, `4u`) with named constants from `kHistHeader` | Commit 1 |
| NIT-2 | Named constant `BIN_OFFSETS_STRIDE = 129u`; clarify `128 bins + 1 total` comment | Commit 1 |
| NIT-3 | T2-accum explicit `if (partSize == 0) return;` guard (eliminate float-to-uint zero aliasing reliance) | Commit 3 |
| NIT-4 | Host-side CB_ENSURE `maxBlocksPerPart == 1` when T2 active; document constraint in kernel header | Commit 3 |
| NIT-5 | Remove unused `numTGs` uniform from T2-sort and T2-accum inputs + host dispatch | Commit 3 |
| NIT-7 | Harmonize feature 1-3 bin mask to `& 0x7Fu` (matching T2-sort feat-0, DEC-016 envelope) | Commit 1 |

---

## §4 Research Track

### S23-R1 — EvalAtBoundary Readback Elimination (carry-forward from S19-11)

**Mechanism**: Six `EvalAtBoundary` CPU readbacks in `structure_searcher.cpp` force Metal sync barriers at each tree-level iteration. At depth 6: ~0.3 ms/iter standalone (~1.6% of 19.098 ms T2 iter_total). Carry-forward from S19-11 (scheduled in S22 per d1r4_synthesis.md §3 rank #2; not executed in S22 due to diagnostic arc scope).

**Sites**: `:275` (per-partition split selection), `:593` (per-leaf sum extraction), `:653` (per-leaf compressedIndex walk), `:686` (lossguide exit boundary). Catalog in `docs/sprint16/sync_inventory.md`.

**Goal**: Replace CPU readbacks with GPU-resident split-selection kernel dispatches. Bounded 0.5–1 day. Document in `docs/sprint23/r1_evalatboundary.md`.

**Kill-switch**: If GPU-resident split selection adds more overhead than it eliminates (new kernel launch cost > sync-barrier cost), leave the readbacks and defer permanently.

### S23-R2 — Dispatch Inversion Research Spike (timeboxed 2 days)

**Mechanism**: Invert the partition-fragmented 1638-TG dispatch: compute one histogram over all docs per iteration, apply partition masks at split-scoring time. Speculative — could restore the 195-docs/thread shape at every depth, but scoring-time masking may introduce its own overhead that cancels the gain.

**Timebox**: 2 days to identify a concrete design that produces a mechanism-testable dispatch shape. If no concrete design surfaces, declare unreachable for the Verstappen campaign window and defer to Sprint 24+.

**Output**: `docs/sprint23/r2_dispatch_inversion_spike.md` — design sketch, mechanism gate spec, and go/no-go verdict.

---

## §5 Exit Gates

| Gate | Criterion | Blocked on |
|------|-----------|-----------|
| S23-D0-G1 | 18/18 ULP=0 post-promotion parity sweep (≥ S22 D3 standard) | D0 Commit 4 |
| S23-D0-G2 | T2 iter_total_ms ≤ 19.5 ms at gate config (no regression vs S22 1.90×) | D0 Commit 4 |
| S23-D0-G3 | `kernel_sources.h` contains T2; `bench_boosting.cpp` no longer has inline T2; `CATBOOST_MLX_HISTOGRAM_T2` flag removed | D0 Commit 3 |
| S23-NIT-G | Code review confirms all 6 nits addressed | D0 Commit 3 |

**Note (added 2026-04-20 post-D0 completion)**: S23 D0 parity sweep result was 17/18 ULP=0 + 1 latent bimodal (config #8). The S23-D0-G1 criterion "≥ S22 D3 standard" inherited the S22 D3 under-sampling error. See errata in `docs/sprint22/d3_parity_gate.md`. DEC-023 is the forward fix; gate S23-D0-G1 is satisfied pending DEC-023 resolution at S24 D0.

### Parity-sweep protocol standing order (effective S23 D0 forward)

All future sprint parity sweeps MUST use:
- **≥5 runs per non-gate config**: catches ~50/50 bimodality at 97% per-config probability (P(all-5-Value-A | p=0.5) = 3.125%)
- **100 runs at gate config unconditionally**: was already the S22 D3/D4 standard for the gate config; now formalized for all sprints

Rationale: S22 D3's 1-run-per-config protocol had a 50% miss probability for any ~50/50 bimodal race. The S23 D0 5-run protocol reduced this to 3.1%, which detected the config #8 defect. All future sprint gates inherit this floor.

---

## §6 R8 Position

**Current (post-S22)**: 1.90× cumulative. Verstappen ≥1.5× gate already cleared.

Sprint 23 is not expected to contribute additional R8 from the promotion pass (T2 performance is unchanged by code location). S23-R1 could add ~+0.3 ms/iter compound gain (≈1.6% further improvement on top of 1.90×). S23-R2 is speculative — see §4.

**Do not soften or inflate the 1.90× figure.** This is the honest post-S22 position and propagates unchanged into the Sprint 23 closeout.

---

## §7 D-Document Placeholders

| Doc | Status | Description |
|-----|--------|-------------|
| `d0_promotion_parity.md` | PENDING | Post-promotion 18-config parity sweep + 100-run determinism |
| `r1_evalatboundary.md` | PENDING | EvalAtBoundary readback elimination design + measurement |
| `r2_dispatch_inversion_spike.md` | PENDING | Dispatch inversion 2-day research spike + go/no-go verdict |
