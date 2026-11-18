# Metal validation allocation optimization

Successful validation checks previously constructed `std::string` from literal error messages. Many messages exceeded the short-string capacity, causing one heap allocation and free per checked value. A 64-output evaluation therefore spent far more wall time validating buffers than executing its Metal kernels.

Seventeen private `Require` helpers across sixteen native files now have a `const char*` overload. Literal messages allocate an exception string only when the check fails. The existing `std::string` overload still handles dynamically constructed messages. All predicates, error text, memory bounds, failure states and GPU algorithms remain unchanged.

The helpers cover scalar/vector/Ordered training, scalar/vector evaluation, sort, projection/CTRs, query/pair sampling and full-matrix target workspaces. This is a host validation optimization; it does not change GPU arithmetic.

## Measured on the M3 Pro

Five alternating warmed measurements compare the preserved preceding Metal library with the new library in one process. Both use 65536 rows, four binned features and sixteen identical trees. All output arrays are bit identical; source scripts, library hashes, binaries and raw samples are archived.

| Outputs | Previous median | New median | Wall-time speedup |
|---|---:|---:|---:|
| 1 | 28.976 ms | 5.269 ms | 5.50x |
| 3 | 80.389 ms | 6.722 ms | 11.96x |
| 7 | 176.012 ms | 10.126 ms | 17.38x |
| 64 | 1547.021 ms | 67.806 ms | 22.82x |

Native fit comparisons run in two persistent processes with the preceding/new standard extensions. Each case uses 16384 rows, eight features, twelve depth-four trees and weighted validation; five warmed runs alternate between the two processes. The hashes of model leaves, leaf weights, tree sizes, GPU predictions and metric histories agree in every run.

| GPU training path | Previous median | New median | Fit speedup |
|---|---:|---:|---:|
| scalar_greedy | 78.436 ms | 80.372 ms | 0.98x |
| vector_greedy | 125.712 ms | 124.013 ms | 1.01x |
| scalar_ordered | 40.375 ms | 41.648 ms | 0.97x |

Native fit medians differ by approximately three percent or less in these workloads; this does not establish a training speedup. The large measured improvement is in evaluation.

These are measurements of these workloads on this M3 Pro, against the preceding Metal implementation. They are not NVIDIA comparisons or evidence of complete CUDA performance parity.

## Validation and recovery

- 8687 combined tests plus 16 subtests; 2629 alternate acceptance cases. Existing malformed-input, overflow, cursor rollback, snapshot and model checks pass.
- 343 installed GPU fit/prediction paths and all 92 CLI variants pass.
- Thirty-six snapshots written with the preceding standalone sources/runtime resume exactly: four scalar/vector losses, three growth policies and numeric/one-hot/CTR data. Original snapshots are preserved separately from their resumed files.
- Installed checkpoint `20260913T132447Z`, wheel SHA256 `eb98a22385a20f275d4e0279dca63ade00f12bcbc0d1d582ec172da7d88bfaaa`. The preceding checkpoint `20260913T130953Z` remains intact.

No CPU CatBoost fitting or NVIDIA/other M-series execution was performed. Dynamic CTRs, categorical/grouped Ordered, full CUDA GPU RNG, estimated features and wider workload validation remain open.
