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

None.

## In progress

None.

## Deferred

- [ ] [5. Establish stronger numerical agreement](05-numerical-agreement.md) — paused at the user's request; unfinished work preserved on a local branch.
- [ ] [6. Prove scale and performance](06-scale-and-performance.md) — deferred given the available M3 Pro's 18 GB unified memory and missing comparison hardware.

## Done

- [x] [1. Finish greedy PairLogit](01-greedy-pairlogit.md)
- [x] [2. Complete categorical combinations](02-categorical-combinations.md)
- [x] [3. Finish remaining training modes](03-training-modes.md)
- [x] [4. Close API and option gaps](04-api-and-options.md)
- [x] [7. Support model-based feature analysis](07-model-based-feature-analysis.md)

Move a card's link between sections and update its `status` as work progresses.
Tick its checklist items only when supported by validation evidence.

## Evidence

- [Implementation status](../catboost/metal/IMPLEMENTATION_STATUS.md)
- [Greedy PairLogit release report](../catboost/metal/GREEDY_PAIRLOGIT_PORT.md)
- [Compound CTR release report](../catboost/metal/COMPOUND_CTR_PORT.md)
- [Training modes release report](../catboost/metal/TRAINING_MODES_PORT.md)
- [API and options release report](../catboost/metal/API_OPTIONS_PORT.md)
- [Model-based feature analysis release report](../catboost/metal/MODEL_BASED_EVAL_PORT.md)
- [Metal backend README](../catboost/metal/README.md)

Current installed checkpoint: `20260914T052913Z`. 14,551 tests plus 16 subtests in the full matrix; 6,187 tests alternate; 2,606 tests installed; 348 CLI; 203 preinstall smoke; 203 installed smoke; 350 exact preceding snapshot recoveries; 143 C++ checks.
Counts overlap. [Model-based feature analysis release report](../catboost/metal/MODEL_BASED_EVAL_PORT.md).
Cards 5 and 6 remain deferred, including shared RNG carryover between feature
experiments, vector prefix rounding differences, and broader hardware evidence.
