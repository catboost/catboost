# CI Histogram Gate Design — Sprint 17

## How the gate triggers

The gate runs on every pull request targeting `master` and on every push to
`mlx/**` branches, matching the existing wall-clock gate trigger. It is a new
step in the same job (`perf-regression-check`) in
`.github/workflows/mlx-perf-regression.yaml`, added after the wall-clock check.
`continue-on-error: false` is set on both the capture step and the check step,
so a histogram regression is a hard merge blocker.

## What it measures

The gate compares `histogram_ms` (mean across all recorded iterations) between
the committed Sprint 16 baseline JSONs in `.cache/profiling/sprint17/` and a
freshly captured stage-profile run on the PR's code. `histogram_ms` is recorded
by `TStageProfiler` (see `catboost/mlx/methods/stage_profiler.h`) as a
synchronous, `mx::eval()`-fenced wall-clock measurement of the Metal histogram
kernel, not end-to-end training time. This makes it substantially less noisy
than the wall-clock gate: run-to-run variance on the M1 CI runner is estimated
at 1–2% for `histogram_ms` vs 2–4% for full wall-clock (Sprint 16 per-iter
spread: range across 50 iterations is ~40 ms on a ~310 ms mean, ~13% peak-to-
peak, but inter-run mean variance is far smaller).

## Threshold rationale: why 5%

5% gives approximately 2.5σ margin above the observed inter-run mean variance
for `histogram_ms` (estimated 1–2% 1σ on the macos-14 runner). A formal
variance study during the S17-05 CI dry-run (S17-G5) should replace this
estimate with measured CI σ and tighten or loosen the threshold accordingly.
The value is a placeholder pending that dry-run; if CI shows 1σ above 2.5%,
raise the limit to `max(2 * measured_sigma, 0.05)`. The threshold matches the
Sprint 17 acceptance criterion (S17-G2: no config regresses >5%) so the CI gate
and the manual acceptance gate are consistent.

## Failure modes

**Noisy CI hardware (R8 in sprint17 plan):** If the macos-14 runner delivers
abnormally high variance on a given run, a legitimate PR may see a spurious
failure. Mitigation: re-trigger the workflow once before investigating. If
flakes recur more than once per sprint, escalate the threshold by 1σ or switch
the capture run to `--iterations 100` to reduce mean variance.

**Intentional regression dry-run (S17-G5):** To verify the gate catches real
regressions, create a branch that reverts `kernel_sources.h:160–181` to the
Sprint 16 serial reduction, push it, and confirm the CI `histogram_gate` step
fails. This dry-run is part of Sprint 17 acceptance and must be completed before
the sprint PR merges.

**Deliberate performance change:** If a future sprint intentionally changes
`histogram_ms` (e.g., a new variant that trades histogram speed for accuracy),
regenerate the baselines:

```bash
# Rebuild with CATBOOST_MLX_STAGE_PROFILE=ON, then:
python benchmarks/bench_mlx_vs_cpu.py \
    --scales 1000,10000,50000 \
    --bins 32,128 \
    --mlx-stage-profile \
    --hardware "$(sysctl -n machdep.cpu.brand_string)" \
    --output .cache/profiling/sprint17/new_baseline.json
# Split per-config and commit to .cache/profiling/sprint17/.
```

## How to run locally

Single config (Sprint 17 acceptance gate S17-G1):
```bash
python benchmarks/check_histogram_gate.py \
    --before .cache/profiling/sprint17/baseline_10000_rmse_d6_128bins.json \
    --after  /tmp/after_10000_rmse_d6_128bins.json \
    --min-reduction 0.30
```

Full 18-config regression check (Sprint 17 S17-G2):
```bash
python benchmarks/check_histogram_gate.py \
    --18config \
    --before-dir .cache/profiling/sprint17 \
    --after-dir  /tmp/hist_gate_after \
    --max-regression 0.05
```
