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

## DEC-002 — Sprint branch policy: dedicated branches per sprint, merge to master after QA/MLOps sign-off
- **Date:** 2026-04-09
- **Status:** Active
- **Decided by:** ml-product-owner (process decision, Sprint 3 retrospective)
- **Context:** Sprints 1–3 committed directly to master, which meant QA-found bugs (BUG-001, BUG-002) had to be fixed with follow-up commits on master rather than before merge. This creates noisy master history and risks shipping untested states.
- **Decision:** Starting Sprint 4, all sprint work lands on a dedicated branch `mlx/sprint-<N>-<short-topic>` on `RR-AMATOK/catboost-mlx`. Merging to master requires QA sign-off (test suite green) and MLOps sign-off (no new unbounded sync regression). Sprints 1–3 are already on master and will not be rewritten.
- **Alternatives:** Keep committing directly to master with required CI passing. Rejected: CI does not currently include the full MLOps sync audit, so bugs like BUG-001/BUG-002 can still reach master before QA review.
- **Trade-offs:** Slightly more branch management overhead; master history is cleaner and every merged state is verified. PRs will also give a natural review checkpoint for the agent team.
