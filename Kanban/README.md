---
title: Metal CatBoost parity
updated: 2026-09-13
tags: [catboost, metal, kanban]
---

# Metal CatBoost parity

Backlog from the 2026-09-13 parity assessment, ordered by recommended priority.
The target is CatBoost's CUDA-supported behavior. Each card records remaining
work; existing implementation progress does not mean the card is complete.

## Backlog

- [ ] [3. Finish remaining training modes](03-training-modes.md)
- [ ] [4. Close API and option gaps](04-api-and-options.md)
- [ ] [5. Establish stronger numerical agreement](05-numerical-agreement.md)
- [ ] [6. Prove scale and performance](06-scale-and-performance.md)

## In progress

No cards currently in progress.

## Done

- [x] [1. Finish greedy PairLogit](01-greedy-pairlogit.md)
- [x] [2. Complete categorical combinations](02-categorical-combinations.md)

Move a card's link between sections and update its `status` as work progresses.
Tick its checklist items only when supported by validation evidence.

## Evidence

- [Implementation status](../catboost/metal/IMPLEMENTATION_STATUS.md)
- [Greedy PairLogit release report](../catboost/metal/GREEDY_PAIRLOGIT_PORT.md)
- [Compound CTR release report](../catboost/metal/COMPOUND_CTR_PORT.md)
- [Metal backend README](../catboost/metal/README.md)

Current installed checkpoint: `20260913T220233Z`. Its full matrix passes
11,323 tests plus 16 subtests. [Compound CTR release report](../catboost/metal/COMPOUND_CTR_PORT.md).
