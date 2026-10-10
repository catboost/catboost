# Compound CTR acceptance helpers

These helpers use native Metal training (`task_type="GPU"`) and standard
CBM/JSON readers. CPU prediction checks interoperability; no CPU fit is used.
Every output path must be new. Commands preserve their selected package,
extension and helper checksums, reports, input files and training logs.

`run.py baseline-create` captures twelve old installed RMSE snapshot cases:
Plain/Ordered × numeric/one-hot/simple CTR × requested P1/P4. Capture is gated
on the selected release's extension checksum. Plain numeric and one-hot data
have one effective permutation in that checkpoint. `baseline-replay` requires
bit-for-bit leaf arrays, held-out predictions and histories, and verifies that
only iterations 3–5 invoke callbacks after restoring two trees.

```sh
python catboost/metal/examples/compound_ctr_acceptance/run.py baseline-create \
  --release /path/to/releases/20260913T195124Z \
  --output-dir /new/old-baselines
python catboost/metal/examples/compound_ctr_acceptance/run.py baseline-replay \
  --package /path/to/tested/package-root \
  --baseline /new/old-baselines --output-dir /new/baseline-replay
```

`smoke.py` covers sixteen configurations: Plain/Ordered × P1/P4 × maximum
complexity 2/3 × RMSE/Logloss. Each trained model must actually contain a
compound CTR. Both CBM and JSON exports must retain predictions for observed,
absent-joint and unseen categories through native GPU and standard readers.
The maximum-complexity option is a bound; a case need not select that many
categorical constituents. The native acceptance tests check those additional
structure, mixed-feature and independent final-table invariants.

```sh
python catboost/metal/examples/compound_ctr_acceptance/smoke.py smoke \
  --package /path/to/installed/site-packages --output-dir /new/installed-smoke
python catboost/metal/examples/compound_ctr_acceptance/smoke.py cli \
  --package /path/to/tested/package-root --cli /path/to/tested/catboost \
  --output-dir /new/cli-smoke
```

`compatibility.py` orchestrates the current and preserved checks in fresh
subdirectories. Its `snapshots` mode replays the twelve new baselines and the
preceding 202 immutable fixtures; `cli` runs the previous 171 and current 16
configurations; `smoke` runs 18 preceding PairLogit and 16 compound cases.
The original release archives must remain available: checkpoint 195124Z
contains the prior 202 fixtures, and 155047Z contains the older CLI scripts.

```sh
python catboost/metal/examples/compound_ctr_acceptance/compatibility.py snapshots \
  --package /path/to/tested/package-root --baseline /new/old-baselines \
  --release /path/to/releases/20260913T195124Z --output-dir /new/snapshots
python catboost/metal/examples/compound_ctr_acceptance/compatibility.py cli \
  --package /path/to/tested/package-root --cli /path/to/tested/catboost \
  --older-release /path/to/releases/20260913T155047Z --output-dir /new/cli
python catboost/metal/examples/compound_ctr_acceptance/compatibility.py smoke \
  --package /path/to/installed/site-packages --output-dir /new/smoke
```

Run the full native/standalone pytest suite and the alternate extension matrix
separately. Release directories retain exact commands, source manifests, XML
reports, wheel and binary hashes, and original snapshot inputs. These scripts
neither install packages nor replace an existing output directory.
