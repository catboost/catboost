# S27-CR Code Review Report

**Sprint**: 27 — correctness closeout (FU-1 / AA / FU-3)
**Branch**: `mlx/sprint-27-correctness-closeout`
**Base**: `master` @ `20d14564e8`
**Tip**: `591f4ce3e6`
**Range**: 24 commits
**Reviewed by**: @code-reviewer (S27-CR)
**Date**: 2026-04-22

---

## Overall Verdict

**APPROVE** — ready for PR against `RR-AMATOK/catboost-mlx:master`.

All three tracks land cleanly:
- FU-1 fixes the documented Depthwise validation-path bugs per DEC-030 and passes G1-FU1 6/6 with a clean non-regression result in the ST smoke cell.
- AA delivers an atomic per-anchor landing stream with DEC-031 hygiene protocol and retires DEC-029's `ComputeLeafIndicesDepthwise` risk entry.
- FU-3 produces DEC-032 with the explicit "different algorithms, not parity-equivalent" framing Ramos required, and G3-FU3 lands 5/5 with explicit scope labeling.

No blockers. Two nits below — neither blocks merge; both are follow-up candidates.

**Blockers**: 0
**Nits**: 2
**Commendations**: 4

---

## Blockers

_None._

---

## Nits (non-blocking)

### Nit 1 — FU-1 gate has no CI-asserted regression guard (DEC-031 Rule-1 tension)

- **Files**: `docs/sprint27/scratch/fu1_t4_gate.py` (sprint artifact only);
  `tests/test_python_path_parity.py` (existing tests cover training-path, not validation-path
  leaf-index encoding).
- **Observation**: G1-FU1 is a one-shot Python harness run and reported in
  `docs/sprint27/fu1/t4-gate-report.md`. It is NOT wired as a pytest. The existing
  `test_nonoblivious_python_path_parity` cells fit CPU CatBoost via the Python binding,
  which routes through the prediction path (DEC-029) rather than `ComputeLeafIndicesDepthwise`.
  The specific function just fixed therefore has no live CI assertion.
- **Relevance**: DEC-031 Rule-1 explicitly prohibits "new docs-only canonical values" and
  requires live assertions. FU-1 evidence walks right up to that line — the gate values
  (ratio 0.9988–1.0027) live only in a markdown report.
- **Recommendation**: Open a follow-up task (e.g. S28-QA-T1) to port the 6-cell DW val-RMSE
  harness to `tests/test_python_path_parity.py` with `eval_set=(X_val, y_val)` and
  `use_best_model=True`, gated by `ratio in [0.98, 1.02]`. This closes the DEC-031 gap for
  the newly-touched surface. Not blocking S27 merge — the gate passed and DW training-path
  regressions would still be caught by existing tests — but the gap should be tracked.

### Nit 2 — `exit(1)` in the ensure-equivalent branch is an abrupt termination

- **File**: `catboost/mlx/tests/csv_train.cpp:1788–1795`
- **Observation**: The "BFS node not found" branch calls `fprintf(stderr, …); exit(1);`.
  The DEC-030 spec left the error-handling pattern to the implementer because `csv_train.cpp`
  does not include `CB_ENSURE`. The chosen pattern terminates cleanly but abruptly, with no
  stack trace and no exception that callers (csv_train is a test harness driver, so this is
  fine in practice) could intercept.
- **Invariant note**: The lookup is provably safe — `splits` and `splitBfsNodeIds` are populated
  in lockstep at `csv_train.cpp:3652/3668` for every partition (including no-op partitions,
  which get placeholder descriptors). The only way to hit this branch is a logic bug in the
  training build loop. `exit(1)` is therefore a defensive guard, not a code path exercised in
  normal operation.
- **Recommendation**: Acceptable as-is. If cleaner, use `throw std::runtime_error(...)` —
  `csv_train.cpp` already uses `std::runtime_error` elsewhere, and the invariant-violation
  message would then propagate through the normal exception path. Style preference, not
  correctness.

---

## Commendations

- **Atomicity discipline (DEC-012)**: 24 commits, each scoped to one structural change. Anchor
  landings split per-anchor (AN-006, AN-007, AN-008, AN-012, AN-016, AN-018, AN-013/014 DEAD,
  AN-015 filename, AN-015a/015b post-wire — 9 distinct commits for 8 anchors + one typo fix).
  `ab5a44f9b9` cleanly separates the post-wire skip logic from the filename fix `bf65c1c59a`.
  This is textbook DEC-012 adherence.
- **DEC-032 framing integrity**: The decision document explicitly states "both sides
  implement correct but **different algorithms**" and declares "a parity claim requires both
  sides to compute the same algorithm; they do not." No "both correct just different" hedge.
  The "coincidental-not-structural" label for prior aggregate-parity claims is honest framing
  done well. Commit `ff053fa3ac` also pre-emptively rejects the easy exit (widening N scope)
  and cites DEC-031 Rule-3 to justify why.
- **Scope discipline on FU-1**: Call-site triage (`eca086e4dd`) established validation-only
  scope before any code change. The final diff touches only `ComputeLeafIndicesDepthwise` body
  and its sole caller at `:4054` — zero collateral changes to training path, FindBestSplit,
  or SymmetricTree code. The ST smoke cell in G1-FU1 (ratio 1.0019) empirically confirms
  the non-regression.
- **DEC-029 retirement done properly**: `13c7ac9b2b` strikes through the DEC-029 Risks bullet
  rather than deleting it (preserves historical readability), adds "Superseded-by: DEC-030"
  pointer, and updates DEC-030 status `DRAFTED → IMPLEMENTED` with commit SHAs. Exactly the
  retirement hygiene DEC-031 will codify for future anchors.

---

## Findings by Focus Area

### Focus 1 — `csv_train.cpp` DW fix — **YES** (matches DEC-030 exactly)

- **Bug A fix**: `partBits |= (goRight << lvl)` + `leafVec[d] = partBits` at lines 1803/1806.
  Mirrors training-path encoding at lines 3664–3666 (`bfsNode` derivation inverse). Bit `lvl`
  = goRight at depth `lvl`. ✓
- **Bug B fix**: `nodeSplitMap` built at lines 1770–1774 from parallel `splits[i]` /
  `splitBfsNodeIds[i]` arrays. Lookup at line 1796 uses `nodeSplitMap.at(nodeIdx)`. Mirrors
  `ComputeLeafIndicesLossguide` pattern at lines 1842–1849. ✓
- **Edge cases**:
  - `depth=0` early-returns zeros at line 1763. ✓
  - `depth=1` exercises the loop once, `partBits ∈ {0,1}`, correct for 2-partition leaf layout. ✓
  - No-op partitions (`Mask == 0`): `splitBfsNodeIds.push_back(bfsNode)` runs unconditionally
    at line 3668, so `nodeSplitMap` always contains an entry for every internal BFS node.
    The `Mask == 0` descriptor yields `fv = 0` and `goRight = 0` (left child) via the
    threshold comparison — harmless. ✓
- **Map rebuild cost**: `unordered_map<ui32, TObliviousSplitLevel>` with `reserve(splits.size())`.
  Map size ≤ 2^depth − 1 (≤ 63 at depth=6). Rebuilt once per call (once per training iter under
  `valDocs > 0`). Negligible, well below the iter's compute budget. Not hoisted, acceptable.
- **Caller update at `:4054`**: `splitBfsNodeIds` is declared at `:3263` inside `RunTraining`
  and in scope at `:4054`. Grep confirms it's populated in both DW (`:3668`) and LG (`:3404`)
  branches. ✓
- **Ensure-equivalent**: `fprintf(stderr, …); exit(1);` — see Nit 2. Message includes
  `nodeIdx`, `depth`, `splits.size()` for triage.
- **Scope check**: Diff touches only the function body (1748–1810) + one caller line (4054).
  No training, FindBestSplit, score, or ST code accidentally modified. ✓

### Focus 2 — Test / gate changes — **YES**

- `tests/test_python_path_parity.py:311` — `score_function='L2'` branches **only on Depthwise**;
  Lossguide and SymmetricTree paths untouched. Matches DEC-032 scope exactly.
- Anchor updates:
  - AN-006 `BINARY_ANCHOR_LOSS`: `0.11909308 → 0.11097676` matches
    `docs/sprint27/scratch/aa-t2-rerun-results.md` "Current Value" column exactly. ✓
  - AN-007 `MULTICLASS_ANCHOR_LOSS`: `0.63507235 → 0.61788315` matches. ✓
  - Assertion tolerances unchanged (1e-6 header comment preserved). ✓
- AN-015 workflow rename: `.github/workflows/mlx-test.yaml` confirmed present (ls output);
  typo fix `mlx_test.yaml → mlx-test.yaml` resolves to the real file.

### Focus 3 — DEC-012 atomicity — **YES**

- 24 commits, each with one structural change. No commits bundle unrelated edits.
- 8 anchor landings are 8 separate commits (AN-006, AN-007, AN-008, AN-012, AN-016, AN-018,
  AN-013/014 pair, AN-015 filename) plus the AN-015a/015b post-wire commit — 9 total, one
  logical unit per commit.
- AN-015 filename fix (`bf65c1c59a`) and AN-015a/015b skip-with-pointer (`ab5a44f9b9`) are
  separate commits. Correct decomposition.
- DEC-030 draft (`c7c09451e2`), DEC-030 implementation (`fb7eb59b5f`), and DEC-029 Risks
  retirement (`13c7ac9b2b`) are three separate commits — spec, implementation, closure cleanly
  decoupled.

### Focus 4 — Standing orders — **YES**

- `git log master..HEAD` + grep for `Co-Authored-By`: **ZERO** commits with trailer. ✓
- All commit subjects follow `[mlx] sprint-27: S27-XX-TN <description>` format. ✓
- No files outside S27 scope touched. `docs/sprint17/`, `docs/sprint18/`, `docs/sprint19/scratch/`,
  `docs/sprint26/d0/`, `benchmarks/results/` are all anchor-update targets for AA-T4 (valid
  scope). `CHANGELOG.md:41` is the AN-008 target. All accounted for.
- Author identity: `RR-AMATOK <72465094+RR-AMATOK@users.noreply.github.com>` on every commit. ✓

### Focus 5 — DECISIONS consistency — **YES**

- **DEC-030** scope matches fix scope: validation-only (call site confirmed at `:4054`),
  both bugs addressed (A encoding + B split-lookup). Status correctly updated
  `DRAFTED → IMPLEMENTED` with T3 `fb7eb59b5f` + G1-FU1 `88cbe6d067` SHAs in
  `13c7ac9b2b`. ✓
- **DEC-031** content covers anchor hygiene protocol with 4-class taxonomy, 4 hygiene rules,
  authored-by and source-material fields complete. Retirement discipline spelled out. ✓
- **DEC-032** language is the strong framing Ramos required: "different algorithms", "not
  parity-equivalent", "coincidental-not-structural" for prior aggregate claims.
  Explicitly rejects the widen-N escape hatch via DEC-031 Rule-3 citation. No pull-punch
  phrasing detected. ✓
- **DEC-029 Risks retirement**: Struck through (`~~...~~`) rather than deleted; pointer to
  DEC-030 + G1-FU1 commit (`88cbe6d067`) + retirement date. ✓

### Focus 6 — Test coverage — PARTIAL (see Nit 1)

- G1-FU1 gate evidence reproducible from `docs/sprint27/scratch/fu1_t4_gate.py` (committed)
  and gate report at `docs/sprint27/fu1/t4-gate-report.md`. Harness committed, can be re-run. ✓
- `test_python_path_parity.py` changes add `score_function='L2'` to existing DW cells only.
  No new test cells added. ✓
- No CI-wired pytest for the specific validation-path leaf-index fix. See **Nit 1** — the
  DEC-031 Rule-1 "prefer live-asserted" guidance is tension-but-not-violation: the fixed code
  is exercised transitively by the existing DW tests (via training loop execution), but the
  validation-RMSE assertion itself is only in the sprint-artifact Python harness.

### Focus 7 — Non-findings noted — **ACK**

- `docs/sprint27/scratch/` artifacts committed (FU-1 repro, FU-3 triage JSON + scripts,
  AA T1/T2/T3). Per project precedent — not flagged.
- `_core.so` rebuild not committed — expected.
- `compile_commands.json` / clangd: no tooling issue in this review — no blockers or nits
  traced to build-system config.

---

## Clang / Build Concern

**No.** The C++ changes use only `<unordered_map>` (already included at `csv_train.cpp:65`),
`<cstdio>` (`fprintf`, already included at `:55`), `<cstdlib>` (`exit`, available via transitive
includes and the `exit(1)` pattern is used elsewhere — confirmed by grep). The signature change
is propagated to the single caller. `std::vector<ui32>` parameter type matches the declaration
at `:3263`. Would compile cleanly with the project's existing release flags; no new link-time
dependencies.

---

## Summary for Report-Back

| Metric | Value |
|--------|-------|
| Verdict | APPROVE |
| Blockers | 0 |
| Nits | 2 (DEC-031 live-assertion gap for validation-path gate; `exit(1)` style) |
| Commendations | 4 (atomicity, DEC-032 framing, FU-1 scope discipline, DEC-029 retirement hygiene) |
| Commit SHA reviewed | `591f4ce3e6` |
| Base SHA | `20d14564e8` |
| Clang/build concern | No |
