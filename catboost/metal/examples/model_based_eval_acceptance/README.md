# Model-based feature analysis example

**Status: accepted in installed checkpoint `20260914T052913Z`.** Use its archived
Metal CLI and matching installed package. The preceding `20260914T025201Z`
checkpoint remains preserved and does not provide this analysis path.

From the repository root, create a deterministic numeric fixture in a fresh
directory. This step only writes data; it does not fit a CPU or GPU model.

```sh
model_eval_data="$(mktemp -d /tmp/metal-model-eval-data.XXXXXX)"
catboost/metal/.venv/bin/python - "$model_eval_data" <<'PY'
import csv
from pathlib import Path
import random
import sys

directory = Path(sys.argv[1])
generator = random.Random(9127)
for name, count in (("train", 128), ("test", 64)):
    with (directory / (name + ".tsv")).open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        for row in range(count):
            retained = generator.gauss(0, 1)
            tested = float(generator.random() > .5)
            auxiliary = generator.gauss(0, 1)
            target = 1.2 * retained + 1.4 * (2 * tested - 1) + .15 * auxiliary
            writer.writerow((target, retained, tested, auxiliary, 1.0))
(directory / "columns.cd").write_text(
    "0\tTarget\n1\tNum\tretained\n2\tNum\ttested\n"
    "3\tNum\tauxiliary\n4\tNum\tconstant\n"
)
print(directory)
PY
```

Keep `model_eval_data` set to the printed absolute directory, choose a fresh
absolute `model_eval_run` directory, and run the baseline/analysis commands in
[MODEL_BASED_EVAL_PORT.md](../../MODEL_BASED_EVAL_PORT.md#author-a-baseline-and-run-analysis).
Use the existing shell variable instead of replacing it with the placeholder
data path shown there.

For this column description, `'1;2;1-2'` and
`'tested;auxiliary;tested-auxiliary'` specify the same three evaluated sets.
Feature 0 (`retained`) supplies background signal. The default baseline fit
ignores features `'1:2'`; the analysis command receives all original columns.
To exercise the full-baseline policy, author a separate baseline without that
ignore list and add the no-value
`--use-evaluated-features-in-baseline-model` switch to analysis.

With eight baseline trees, offset four, two experiments and experiment size two,
the starts are baseline prefixes four and six. Expect six trial directories,
`feature_set0_fold0` through `feature_set2_fold1`, each containing
`learn_error.tsv` and `test_error.tsv` with local iterations 0 and 1. The baseline
snapshot remains unchanged. The command produces metric histories; it does not
export trial models or aggregate feature-importance rankings.

For a Lossguide variant with `max_leaves=31` and `depth=2`, the optional
per-permutation history stores four padded leaf slots per tree, matching the
runtime's depth-bounded capacity. Ordinary snapshot padding retains its existing
31-slot contract. A tree with fewer leaves still uses the history's four-slot
stride.

The executable acceptance tests can use the archived CLI and matching installed
package:

```sh
CATBOOST_NATIVE_METAL_TESTS=1 \
CATBOOST_METAL_CLI="$PWD/catboost/metal/.build/releases/20260914T052913Z/catboost" \
PYTHONPATH="$PWD/catboost/metal/python" \
catboost/metal/.venv/bin/python -m pytest -q \
  catboost/metal/tests/test_native_model_based_eval.py \
  catboost/metal/tests/test_native_model_based_eval_contracts.py
```

The 69 CLI/contract cases passed within the full, alternate and installed
matrices. The release also passed 350 preceding snapshot recoveries, 348 CLI
checks, and separate 203-case preinstall and installed smoke gates. See the
[release report](../../MODEL_BASED_EVAL_PORT.md#source-mapping-and-acceptance)
for all counts and hashes; these selections overlap.

Run the GPU test command when no other Metal acceptance run is active. Metal
restarts its random helpers for each experiment, unlike CUDA's shared generator;
rounded vector-prefix replay can differ from live fused updates. These limits
remain in deferred card 5. Scale/performance card 6 also remains deferred under
the available hardware. This example establishes no NVIDIA comparison.
