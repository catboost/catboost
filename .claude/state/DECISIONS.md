# Decision Log

<!-- Agents record design decisions here. Append-only — to reverse a decision, add a new entry. -->
<!-- Format: DEC-NNN sequential IDs. Status: Active / Superseded by DEC-XXX / Deprecated -->

## DEC-001 — Loss parameter syntax: normalize to positional form before binary call
- **Date:** 2026-04-09
- **Status:** Active
- **Decided by:** ml-engineer (Sprint 3 follow-up)
- **Context:** CatBoost's canonical loss string syntax uses named params (`Quantile:alpha=0.7`). The Python layer needs to accept both the named form (user-facing) and the positional form that `csv_train` expects. BUG-002 revealed that passing the named form directly to the binary caused a parse failure.
- **Decision:** Python `_build_train_args` always normalizes to positional form before constructing the CLI invocation. The named form is only used in `_validate_params` for range checking and user error messages. The binary never sees `param=value` keys.
- **Alternatives:** Teach `csv_train` to accept named params. Rejected: that would require C++ changes to the binary's arg parser and create a second divergence surface.
- **Trade-offs:** Slightly more logic in the Python layer; binary interface stays simple and consistent.

---

## DEC-003 — float32 bucket-count limit in GPU partition layout
- **Date:** 2026-04-09
- **Status:** Active
- **Decided by:** ml-engineer (Sprint 4 follow-up, Sprint 5 documented)

`ComputePartitionLayout` uses `mx::scatter_add_axis` with float32 accumulators
to count docs per partition. float32 represents integers exactly only up to
2^24 = 16,777,216. A runtime `CB_ENSURE` in `structure_searcher.cpp` guards
`numDocs` against this limit. Datasets above 16M rows will need int32
accumulation — filed as a future enhancement, not blocking current targets.

**Alternatives:** Use int32 scatter_add throughout. Deferred because MLX's
`scatter_add_axis` on int32 requires casting around the API; float32 is
currently simpler and covers all realistic training sizes on Apple Silicon
(M-series unified memory caps out well below the 16M row limit in practice).

**Commits:** `8717ddd` (guard added), `fff9f02` (Sprint 4 merge)

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
