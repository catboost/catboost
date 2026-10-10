# Greedy PairLogit release checks

`run.py` runs bounded acceptance checks against an explicitly selected CatBoost
package. Every native fit requires `task_type='GPU'`; native metadata and GPU
predictions are checked. The CLI and installed-package checks cover Depthwise,
Lossguide and Region with numeric, one-hot and CTR features under Newton and
Gradient leaf estimation (18 cases per command), including CBM/JSON readers.

Run from the repository root, using a Python environment with NumPy and the
CatBoost runtime dependencies. Replace the package, executable and output paths
with the release under test. `--package` names the directory **containing** the
`catboost` package. Omitting it uses the installed package or `PYTHONPATH`.
Existing output directories are rejected. Reports record the package and
extension SHA256, completed cases, status, and any failure traceback. Redirect
stdout/stderr to an adjacent log to retain native diagnostics.

```sh
python catboost/metal/examples/greedy_pairlogit_acceptance/run.py cli \
  --package /path/to/standard \
  --cli /path/to/catboost \
  --output-dir /path/to/new-cli-output

python catboost/metal/examples/greedy_pairlogit_acceptance/run.py smoke \
  --output-dir /path/to/new-installed-smoke-output

python catboost/metal/examples/greedy_pairlogit_acceptance/run.py legacy-replay \
  --package /path/to/standard \
  --standalone catboost/metal/python \
  --output-dir /path/to/new-legacy-output
```

`legacy-replay` reads 194 preserved `original-snapshots` plus `expected.npz`
fixtures from `catboost/metal/.build/releases/20260913T155047Z` by default (override
with `--release`). Those fixtures were generated using checkpoint
`20260913T151643Z` and previously validated by `20260913T155047Z`. There are 96
native/standalone scalar Ordered cases and 98 scalar/vector greedy or symmetric
query cases. The runner copies only required fixture files into new directories;
it never retrains the old package or changes the archive. Every saved array must
match exactly. Standalone resume must report two restored iterations. Native
callbacks must report only newly trained iterations 3, 4 and 5, proving that the
first two trees came from the snapshot. Source-file checksums are recorded and
rechecked after replay.

Capture additional fixtures with the old installed checkpoint **before**
installing its replacement:

```sh
catboost/metal/.venv/bin/python \
  catboost/metal/examples/greedy_pairlogit_acceptance/run.py baseline-create \
  --output-dir /path/to/new-old-baselines

python catboost/metal/examples/greedy_pairlogit_acceptance/run.py baseline-replay \
  --package /path/to/standard \
  --baseline /path/to/new-old-baselines \
  --output-dir /path/to/new-baseline-replay
```

The baseline creator verifies the old extension hash against the selected release
manifest, then saves eight native cases: symmetric PairLogit and greedy RMSE for
all three policies, each with numeric and CTR features. Each case retains inputs,
options, an interrupted two-tree snapshot, full expected arrays and checksums.
Replay requires exact outputs and the native resume callback proof. These checks
supplement the full regression suites; they do not establish CUDA numerical or
performance parity.

`prior_cli.py` replays the preceding 153 CLI variants using 14 archived scripts
from the same release. It remaps their executable, output and helper paths,
copies just two required seed files, preserves their assertions, and additionally
requires GPU CLI fits, finite predictions and Metal model metadata. All model
and data output stays under a new staging directory; original artifacts stay
unchanged.

```sh
python catboost/metal/examples/greedy_pairlogit_acceptance/prior_cli.py \
  --package /path/to/standard \
  --cli /path/to/catboost \
  --output-dir /path/to/new-prior-cli-output
```
