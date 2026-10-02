# Training-mode release checks

`run.py` captures and replays thirty native snapshots from the preceding
installed Metal release. Every fit requires `task_type="GPU"`; capture also
requires the exact extension checksum recorded in the selected release.
Each command rejects an existing output directory.

The baseline grid contains sixteen Plain/Ordered numeric, one-hot, simple CTR
and compound CTR configurations with requested P1/P4; six greedy RMSE
configurations across Depthwise/Lossguide/Region and P1/P4; and eight symmetric
QueryRMSE, QuerySoftMax, YetiRank and YetiRankPairwise configurations with P1/P4.
Plain numeric and one-hot routes retain their existing effective P1 behavior.

Each case preserves the actual inputs, options, two-tree snapshot, partial
model JSON, and complete old-build leaf arrays, predictions and histories.
All four compound partial models must already contain a compound projection.
Replay checks every input checksum, requires callbacks only for newly trained
iterations, and compares final arrays exactly. It cannot pass by silently
retraining from iteration one. Numeric data are restored as the float32 values
used by native Pools, and categorical values retain their original strings.

```sh
python catboost/metal/examples/training_modes_acceptance/run.py baseline-create \
  --release /path/to/releases/20260913T220233Z \
  --output-dir /new/old-baselines
python catboost/metal/examples/training_modes_acceptance/run.py baseline-replay \
  --package /path/to/tested/package-root --baseline /new/old-baselines \
  --output-dir /new/baseline-replay
```

The earlier 214 immutable fixtures remain in the preserved 220233Z release:
twelve under `old-195124Z-baselines/` and 202 under
`preceding-202-fixtures/`. They are reused alongside these thirty cases.
Baseline creation reuses the source-controlled compound acceptance data
generators and records their checksums; replay reads the saved inputs instead.

## CLI and installed smoke

`smoke.py list` prints the exact case inventory without importing CatBoost or
using the GPU. The current grid contains 100 cases: twelve greedy YetiRank,
64 Plain FeatureParallel/Ordered query configurations, and 24 Combination
configurations. Query modes cover QueryRMSE, QuerySoftMax, PairLogit and YetiRank
with numeric, one-hot, simple CTR and compound data and requested P1/P4. The
Combination mixtures are Huber + RMSE and RMSE + QueryRMSE. Every compound
case must select a compound projection, and all cases compare CBM/JSON GPU
and standard reader predictions, including unseen categories.

Custom MSL objectives require a native Python descriptor. Their Python tests
cover that creator API; the CLI cannot supply the descriptor and is excluded
from that specific interface check.

```sh
python catboost/metal/examples/training_modes_acceptance/smoke.py list
python catboost/metal/examples/training_modes_acceptance/smoke.py cli \
  --package /path/to/frozen/package-root --cli /path/to/frozen/catboost \
  --output-dir /new/training-mode-cli
python catboost/metal/examples/training_modes_acceptance/smoke.py smoke \
  --package /path/to/installed/site-packages --output-dir /new/training-mode-smoke
```

`compatibility.py` reuses the source-controlled preceding helpers in fresh
subdirectories. Snapshot mode checks the prior 214 plus the new thirty old
snapshots (244). CLI mode retains all 187 preceding configurations and adds
the current inventoried grid (287 total); installed smoke retains all 34
preceding configurations and adds that grid (134 total). Reports record both
expected and actual counts, per-suite command and helper checksums, child
report checksums and complete case inventories.

```sh
python catboost/metal/examples/training_modes_acceptance/compatibility.py snapshots \
  --package /path/to/frozen/package-root --output-dir /new/snapshots
python catboost/metal/examples/training_modes_acceptance/compatibility.py cli \
  --package /path/to/frozen/package-root --cli /path/to/frozen/catboost \
  --output-dir /new/cli
python catboost/metal/examples/training_modes_acceptance/compatibility.py smoke \
  --package /path/to/installed/site-packages --output-dir /new/installed-smoke
```

These commands require the preserved source-release fixtures described above.
Packaging, installation and the full native/standalone matrices are separate
release gates. The helpers do not install a package or replace prior evidence.
