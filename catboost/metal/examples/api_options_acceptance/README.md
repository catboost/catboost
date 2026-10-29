# Card 4 release acceptance

`smoke.py list` inventories the new CLI and installed API checks without
importing CatBoost or using the GPU. The inventory exposes
`expected_cases_cli`, `expected_cases_smoke`, per-mode family counts, and every
case. CLI runs exclude the explicitly identified Python-only CV and standalone
estimator routes; each mode requires every case in its own inventory to pass.

```sh
python catboost/metal/examples/api_options_acceptance/smoke.py list
python catboost/metal/examples/api_options_acceptance/smoke.py cli \
  --package /path/to/frozen/package-root --cli /path/to/frozen/catboost \
  --output-dir /new/card4-cli
python catboost/metal/examples/api_options_acceptance/smoke.py smoke \
  --package /path/to/installed/site-packages \
  --standalone /path/to/frozen/standalone/python \
  --output-dir /new/card4-api-smoke
```

The CLI grid covers Simple scalar/vector/full-matrix leaves, greedy fixed
prefixes, feature weights, learned simple CTR priors, full-matrix RSM,
normalization, ridge and meta-L2, and text/embedding training. Dedicated CLI
flags carry the new score/leaf options; text processing remains in the JSON
parameters file. The public grid adds standalone FeatureParallel regression,
classification and ranking in Plain/Ordered modes, plus explicit-fold CV with
metrics-only/returned-model equivalence and an independent held-out RMSE check.
The added Full-counter cases keep different learn/evaluation category counts and
independently check both the native/CLI training evaluation cursor and the
learn-only final reader. Langevin covers scalar DP/FP/Ordered and greedy leaves,
including the source Simple no-op and explicit disabled flag with positive
temperature. Native one-hot cases exercise all 256 known categories and unseen
inference. The grid contains 61 CLI cases and 69 API cases.

Each run requires the selected native package, creates a fresh output directory,
and writes `report.json` with exact expected/completed counts, zero skipped
cases, source/extension checksums, timing and failures. CLI evidence includes
the command, input data, column description and fit log for every case. Ordinary
models exercise both CPU and Metal CBM/JSON readers. Text and embedding models
exercise CBM shared-reader round trips and require the explicit unsupported
estimated-feature GPU-prediction error, matching CUDA's reader boundary.
Every fit requests Metal; CPU model fitting is forbidden.

`baselines.py` independently captures/replays 106 card 3 snapshot fixtures. The
overall compatibility runner retains the older snapshot, CLI and smoke checks;
these new cases supplement their evidence. Host-only helper tests live in
`tests/test_api_options_acceptance_runner.py`. Native execution passed against
the frozen build in installed checkpoint `20260914T025201Z`, including all 350
snapshot recoveries, 348 CLI cases and 203 smoke cases before and after install.
