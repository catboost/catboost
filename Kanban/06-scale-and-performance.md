---
title: Prove scale and performance
status: backlog
priority: 6
created: 2026-09-13
tags: [catboost, metal, parity]
---

# Prove scale and performance

[Board](README.md)

Only the M3 Pro has been exercised. Existing FlightData full-pool fits and
synthetic scaling runs demonstrate supported execution, but do not establish
NVIDIA performance parity or support across other M-series generations.

- [ ] Account for complete GPU and host memory use, including copies and retained category tables.
- [ ] Improve and validate packed feature layouts and feature/query tiling.
- [ ] Revisit software guards: major working buffers at 1 GiB, outputs at 512 MiB, typical scalar rows at 2^24 and numeric borders at 255.
- [ ] Validate wider/deeper workloads and larger categorical/ranking configurations.
- [ ] Run repeated end-to-end and GPU timing measurements with recorded configurations and build identities.
- [ ] Compare quality, memory and speed on actual NVIDIA hardware with matching workloads.
- [ ] Validate additional M-series generations when hardware is available.
- [ ] Run the complete regression and installation matrix on final matching sources and packages.

Done when supported capacity is backed by complete memory measurements and
repeatable workload/hardware evidence, with performance claims tied to the
actual configurations tested.

[Performance evidence and limits](../catboost/metal/IMPLEMENTATION_STATUS.md)
