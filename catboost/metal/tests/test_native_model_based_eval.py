"""CLI model-based feature analysis using actual Metal baseline snapshots.

There is no upstream Python training API for this command. These tests author
snapshots with ordinary GPU CLI fits and inspect the analysis output through
Python. Numerical expectations are independent loss equations or baseline
history slices when a full-baseline experiment keeps exactly the same features;
ordinary init_model continuation is not used to stand in for permutation state.
"""

import csv
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import time

import numpy as np
import pytest
from catboost import CatBoost


pytestmark = pytest.mark.skipif(
    os.environ.get("CATBOOST_NATIVE_METAL_TESTS") != "1",
    reason="requires the rebuilt native Metal model-based-eval CLI",
)


@pytest.fixture(scope="module")
def cli():
    path = Path(os.environ.get("CATBOOST_METAL_CLI", "/tmp/catbooster-native-build/catboost/app/catboost"))
    assert path.is_file(), f"Set CATBOOST_METAL_CLI to the rebuilt CLI: {path}"
    return str(path)


@pytest.fixture(autouse=True)
def forbid_cpu_fit(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Model-based-eval acceptance must not fit CPU CatBoost")
    monkeypatch.setattr(CatBoost, "fit", forbidden)


@dataclass
class Data:
    train: Path
    test: Path
    cd: Path
    targets: np.ndarray
    test_targets: np.ndarray
    weights: np.ndarray
    test_weights: np.ndarray
    baseline: np.ndarray
    test_baseline: np.ndarray


def write_data(directory, x, target, test_x, test_target, *, categorical=False, groups=False,
               weights=None, test_weights=None, baseline=None, test_baseline=None):
    directory.mkdir(parents=True, exist_ok=True)
    target, test_target = np.asarray(target, np.float32), np.asarray(test_target, np.float32)
    weights = np.ones(len(target), np.float32) if weights is None else np.asarray(weights, np.float32)
    test_weights = np.ones(len(test_target), np.float32) if test_weights is None else np.asarray(test_weights, np.float32)
    has_baseline = baseline is not None
    baseline = np.zeros(len(target), np.float32) if baseline is None else np.asarray(baseline, np.float32)
    test_baseline = np.zeros(len(test_target), np.float32) if test_baseline is None else np.asarray(test_baseline, np.float32)
    names = ("retained", "tested", "auxiliary", "constant")
    metadata = ["Target", "Weight"] + (["GroupId"] if groups else []) + (["Baseline"] if has_baseline else [])
    cd = directory / "columns.cd"
    lines = [f"{i}\t{kind}" for i, kind in enumerate(metadata)]
    lines.extend(f"{len(metadata) + i}\t{'Categ' if categorical and i == 0 else 'Num'}\t{names[i]}"
                 for i in range(x.shape[1]))
    cd.write_text("\n".join(lines) + "\n")
    paths = (directory / "train.tsv", directory / "test.tsv")
    for path, features, labels, w, base in zip(paths, (x, test_x), (target, test_target),
                                             (weights, test_weights), (baseline, test_baseline)):
        with path.open("w") as stream:
            for row in range(len(labels)):
                values = [labels[row], w[row]] + ([row // 4] if groups else []) + ([base[row]] if has_baseline else [])
                values += list(features[row])
                stream.write("\t".join(str(value) for value in values) + "\n")
    return Data(*paths, cd, target, test_target, weights, test_weights, baseline, test_baseline)


def problem(directory, loss="RMSE", *, categorical=False, inverse=False):
    rng = np.random.default_rng(9127)
    train_x, test_x = (rng.normal(size=(count, 4)).astype(np.float32) for count in (96, 48))
    for x in (train_x, test_x):
        x[:, 1] = (x[:, 1] > 0).astype(np.float32)
        x[:, 3] = 1
    train_y, test_y = (1.2 * x[:, 0] + 1.4 * (2 * x[:, 1] - 1) + .15 * x[:, 2]
                       for x in (train_x, test_x))
    if categorical:
        outputs = []
        for x in (train_x, test_x):
            categories = np.arange(len(x)) % 6
            y = 2.5 * ((categories % 3) - 1) + .25 * (2 * x[:, 1] - 1)
            values = x.astype(object)
            values[:, 0] = [f"category-{category}" for category in categories]
            outputs.append((values, y.astype(np.float32)))
        (train_x, train_y), (test_x, test_y) = outputs
    if loss in ("MultiClass", "MultiClassOneVsAll"):
        train_y, test_y = (np.asarray(np.digitize(y, [-.5, .5]), np.float32) for y in (train_y, test_y))
    elif loss in ("Logloss", "QueryCrossEntropy"):
        train_y, test_y = (np.asarray(y > 0, np.float32) for y in (train_y, test_y))
    elif loss.startswith("YetiRank"):
        train_y, test_y = (np.asarray(.125 + .75 / (1 + np.exp(-y)), np.float32) for y in (train_y, test_y))
    if inverse:
        test_x = train_x.copy()
        test_y = -train_y
    grouped = loss.startswith(("Query", "Pair", "Yeti"))
    w = (.5 + (np.arange(len(train_y)) % 7) / 4).astype(np.float32)
    tw = (.75 + (np.arange(len(test_y)) % 5) / 5).astype(np.float32)
    w[::23] = 0
    return write_data(directory, train_x, train_y, test_x, test_y, categorical=categorical,
                      groups=grouped, weights=w, test_weights=tw)


def settings(**overrides):
    result = {
        "--task-type": "GPU", "--loss-function": "RMSE", "--boosting-type": "Plain",
        "--data-partition": "DocParallel", "--grow-policy": "SymmetricTree",
        "-i": 8, "--depth": 3, "--learning-rate": .2, "--l2-leaf-reg": 2,
        "--leaf-estimation-method": "Newton", "--leaf-estimation-iterations": 1,
        "--leaf-estimation-backtracking": "No", "--score-function": "Cosine",
        "--boost-from-average": "false", "--use-best-model": "false",
        "--bootstrap-type": "No", "--random-strength": 0,
        "--random-seed": 619, "--permutations": 1, "--border-count": 16,
        "--thread-count": 2, "--metric-period": 1, "--logging-level": "Silent",
        "--one-hot-max-size": 2, "--max-ctr-complexity": 1, "--model-size-reg": 0,
        "--learn-err-log": "learn_error.tsv", "--test-err-log": "test_error.tsv",
    }
    result.update(overrides)
    return result


def command(cli, mode, data, directory, config):
    args = [cli, mode, "-f", str(data.train), "-t", str(data.test), "--cd", str(data.cd),
            "--train-dir", str(directory)]
    if int(config.get("--permutations", 1)) == 1:
        args.append("--has-time")
    for key, value in config.items():
        if value is not None:
            args.extend([key, str(value)])
    return args


def execute(args, *, ok=True):
    # Every subprocess capable of training is explicitly sent to Metal.
    assert args[args.index("--task-type") + 1] == "GPU"
    result = subprocess.run(args, text=True, capture_output=True, timeout=180)
    if ok:
        assert result.returncode == 0, result.stdout + result.stderr
    else:
        assert result.returncode != 0, "Unsupported analysis unexpectedly succeeded"
    return result


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit_baseline(cli, data, directory, config, *, ignored=()):
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / "baseline.cbm"
    baseline_config = config | {"--snapshot-file": "baseline.snapshot", "--snapshot-interval": 0,
                                "--model-file": str(model_path)}
    if ignored:
        baseline_config["--ignore-features"] = ":".join(map(str, ignored))
    execute(command(cli, "fit", data, directory, baseline_config))
    snapshot = directory / "baseline.snapshot"
    assert snapshot.is_file() and snapshot.stat().st_size > 0
    model = CatBoost().load_model(model_path)
    assert model.get_metadata()["metal_backend"] == "METAL"
    return snapshot, model


def analysis_command(cli, data, directory, config, snapshot, features="1", *, offset=4, count=2, size=2, full=False):
    analysis = config | {"--baseline-model-snapshot": str(snapshot), "--features-to-evaluate": features,
                         "--offset": offset, "--experiment-count": count, "--experiment-size": size}
    result = command(cli, "model-based-eval", data, directory, analysis)
    if full:
        result.append("--use-evaluated-features-in-baseline-model")
    return result


def history(path):
    with path.open() as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        records = list(reader)
        columns = reader.fieldnames
    assert columns and columns[0] == "iter" and len(columns) > 1
    iterations = np.array([int(row["iter"]) for row in records])
    values = np.array([[float(row[column]) for column in columns[1:]] for row in records])
    assert np.isfinite(values).all()
    return columns[1:], iterations, values


def fold_histories(directory, sets, folds, size):
    result = {}
    for feature_set in sets:
        for fold in range(folds):
            path = directory / f"feature_set{feature_set}_fold{fold}"
            for dataset in ("learn", "test"):
                columns, iterations, values = history(path / f"{dataset}_error.tsv")
                np.testing.assert_array_equal(iterations, np.arange(size))
                assert not list(path.glob("*.snapshot")), "Experiments must not replace baseline snapshot state"
                assert not list(path.glob("*.cbm")), "CUDA analysis exports histories, not trial models"
                result[feature_set, fold, dataset] = columns, values
    return result


@pytest.mark.parametrize("full", (False, True))
@pytest.mark.parametrize("offset,count,size", ((5, 2, 2), (8, 2, 2), (3, 1, 3)))
def test_model_based_eval_depth_zero_prefixes_match_independent_weighted_rmse(cli, tmp_path, full, offset, count, size):
    x = np.column_stack((np.arange(16), np.arange(16) % 3)).astype(np.float32)
    y = np.tile([0., 1., .25, .75], 4).astype(np.float32)
    w = np.tile([1., 2., 3., 4.], 4).astype(np.float32)
    baseline = np.tile([-.25, .5, .25, -.5], 4).astype(np.float32)
    data = write_data(tmp_path / "data", x, y, x, y[::-1], weights=w, test_weights=w[::-1],
                      baseline=baseline, test_baseline=baseline[::-1])
    config = settings(**{"--depth": 0, "--learning-rate": .2})
    directory = tmp_path / "run"
    snapshot, model = fit_baseline(cli, data, directory, config, ignored=() if full else (1,))
    assert model.tree_count_ == 8
    before = digest(snapshot)
    execute(analysis_command(cli, data, directory, config | {"--metric-period": 7}, snapshot,
                             offset=offset, count=count, size=size, full=full))
    logs = fold_histories(directory, [0], count, size)
    # Independent scalar Newton recurrence at float32 cursor boundaries.
    cursor, test_cursor = data.baseline.copy(), data.test_baseline.copy()
    expected_train, expected_test = [], []
    for _ in range(8):
        gradient = np.float32(data.weights * np.float32(data.targets - cursor)).sum(dtype=float)
        leaf = np.float32(.2) * np.float32(gradient / (data.weights.sum(dtype=float) + 2))
        cursor, test_cursor = np.float32(cursor + leaf), np.float32(test_cursor + leaf)
        expected_train.append(np.sqrt(np.average((data.targets.astype(float) - cursor) ** 2, weights=data.weights)))
        expected_test.append(np.sqrt(np.average((data.test_targets.astype(float) - test_cursor) ** 2, weights=data.test_weights)))
    for fold in range(count):
        start = 8 - offset + (offset // count) * fold
        for dataset, expected in (("learn", expected_train), ("test", expected_test)):
            columns, values = logs[0, fold, dataset]
            assert columns == ["RMSE"]
            np.testing.assert_allclose(values[:, 0], expected[start:start + size], rtol=6e-6, atol=3e-7)
    assert digest(snapshot) == before


@pytest.mark.parametrize("permutations", (1, 4))
@pytest.mark.parametrize("categorical", (False, True))
def test_model_based_eval_unchanged_full_feature_set_replays_baseline_history_prefixes(cli, tmp_path, permutations, categorical):
    data = problem(tmp_path / "data", categorical=categorical)
    config = settings(**{"--permutations": permutations, "--custom-metric": "MAE"})
    directory = tmp_path / "run"
    snapshot, model = fit_baseline(cli, data, directory, config)
    if categorical:
        path = tmp_path / "baseline.json"
        model.save_model(path, format="json")
        assert json.loads(path.read_text())["features_info"].get("ctrs"), "Baseline must use ordered categorical history"
    base = {dataset: history(directory / f"{dataset}_error.tsv") for dataset in ("learn", "test")}
    before = digest(snapshot)
    # Both identical feature sets must start from independent baseline cursors.
    execute(analysis_command(cli, data, directory, config, snapshot, "tested;tested", offset=5, count=2, size=2, full=True))
    logs = fold_histories(directory, [0, 1], 2, 2)
    for feature_set in (0, 1):
        for fold, start in enumerate((3, 5)):
            for dataset in ("learn", "test"):
                columns, values = logs[feature_set, fold, dataset]
                assert columns == base[dataset][0]
                np.testing.assert_allclose(values, base[dataset][2][start:start + 2], rtol=6e-6, atol=3e-7)
    assert digest(snapshot) == before


@pytest.mark.parametrize("full", (False, True))
def test_model_based_eval_named_ranges_overlapping_sets_and_order_are_independent(cli, tmp_path, full):
    data = problem(tmp_path / "data")
    config = settings()
    directory = tmp_path / "run"
    snapshot, _ = fit_baseline(cli, data, directory, config, ignored=() if full else (1, 2))
    before = digest(snapshot)
    execute(analysis_command(cli, data, directory, config, snapshot, "tested;auxiliary;tested-auxiliary", full=full))
    first = fold_histories(directory, [0, 1, 2], 2, 2)
    reordered_dir = tmp_path / "reordered"
    execute(analysis_command(cli, data, reordered_dir, config, snapshot, "1-2;2;1", full=full))
    reordered = fold_histories(reordered_dir, [0, 1, 2], 2, 2)
    for feature_set in (0, 1, 2):
        for fold in range(2):
            for dataset in ("learn", "test"):
                columns, values = first[feature_set, fold, dataset]
                expected_columns, expected_values = reordered[2 - feature_set, fold, dataset]
                assert columns == expected_columns
                np.testing.assert_array_equal(values, expected_values)
    # Distinct feature sets must materially affect this deliberately informative dataset.
    assert any(not np.array_equal(first[0, fold, "test"][1], first[1, fold, "test"][1]) for fold in range(2))
    assert digest(snapshot) == before


@pytest.mark.parametrize("loss,policy", (("RMSE", "Depthwise"), ("Logloss", "Region"),
    ("QueryRMSE", "SymmetricTree"), ("PairLogit", "Lossguide"),
    ("PairLogitPairwise", "SymmetricTree"), ("QueryCrossEntropy", "SymmetricTree"),
    ("YetiRankPairwise", "SymmetricTree")))
def test_model_based_eval_registered_scalar_greedy_and_fullmatrix_paths(cli, tmp_path, loss, policy):
    data = problem(tmp_path / "data", loss)
    config = settings(**{"--loss-function": loss, "--grow-policy": policy, "--depth": 2,
                         "--score-function": "NewtonCosine" if loss in ("PairLogitPairwise", "QueryCrossEntropy", "YetiRankPairwise") else "Cosine"})
    if policy == "Lossguide":
        # The native leaf bank is bounded by depth (four), while the ordinary
        # Lossguide snapshot retains the requested larger capacity.
        config["--max-leaves"] = 31
    if loss.startswith("YetiRank"):
        config["--eval-metric"] = "PFound:hints=skip_train~false"
    directory = tmp_path / "run"
    snapshot, _ = fit_baseline(cli, data, directory, config)
    before = digest(snapshot)
    execute(analysis_command(cli, data, directory, config, snapshot, "1;1", full=True))
    logs = fold_histories(directory, [0, 1], 2, 2)
    for fold in range(2):
        for dataset in ("learn", "test"):
            assert logs[0, fold, dataset][0] == logs[1, fold, dataset][0]
            np.testing.assert_array_equal(logs[0, fold, dataset][1], logs[1, fold, dataset][1])
    assert digest(snapshot) == before


@pytest.mark.parametrize("loss", ("MultiClass", "MultiClassOneVsAll", "RMSEWithUncertainty"))
@pytest.mark.parametrize("policy", ("SymmetricTree", "Depthwise", "Lossguide", "Region"))
@pytest.mark.parametrize("permutations", (1, 4))
def test_model_based_eval_vector_prefixes_replay_all_dimensions_and_permutations(cli, tmp_path, monkeypatch, loss, policy, permutations):
    data = problem(tmp_path / "data", loss, categorical=permutations == 4)
    config = settings(**{"--loss-function": loss, "--grow-policy": policy,
                         "--depth": 2, "--permutations": permutations})
    if policy == "Lossguide":
        config["--max-leaves"] = 31  # Exceeds the depth-bounded native history bank.
    directory = tmp_path / "run"
    snapshot, model = fit_baseline(cli, data, directory, config)
    assert model.tree_count_ == 8
    assert model.get_leaf_values().size > model.get_leaf_weights().size
    if permutations == 4:
        path = tmp_path / "vector.json"
        model.save_model(path, format="json")
        assert json.loads(path.read_text())["features_info"].get("ctrs"), "P4 baseline must use categorical history"
    base = {dataset: history(directory / f"{dataset}_error.tsv") for dataset in ("learn", "test")}
    before = digest(snapshot)
    execute(analysis_command(cli, data, directory, config, snapshot, "tested;tested",
                             offset=5, count=2, size=2, full=True))
    logs = fold_histories(directory, [0, 1], 2, 2)
    rounded = None
    if (loss, policy, permutations) == ("RMSEWithUncertainty", "SymmetricTree", 4):
        # CUDA replays scaled float leaves. Accepted Metal vector training can
        # fuse rate*raw+cursor, and an ULP difference can choose another tied
        # CTR. Validate that boundary and use independent rounded prefix state.
        from model_based_eval_oracle import rounded_uncertainty_prefix_history
        monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "python"))
        rounded = rounded_uncertainty_prefix_history(data, json.loads(path.read_text()), snapshot, tmp_path)
    for feature_set in (0, 1):
        for fold, start in enumerate((3, 5)):
            for dataset in ("learn", "test"):
                columns, values = logs[feature_set, fold, dataset]
                assert columns == base[dataset][0]
                expected = rounded[fold, dataset] if rounded is not None else base[dataset][2][start:start + 2]
                np.testing.assert_allclose(values, expected, rtol=6e-6, atol=3e-7)
    assert digest(snapshot) == before


def test_model_based_eval_constant_feature_set_is_skipped_without_renumbering(cli, tmp_path):
    data = problem(tmp_path / "data")
    config = settings(**{"--logging-level": "Verbose"})
    directory = tmp_path / "run"
    snapshot, _ = fit_baseline(cli, data, directory, config, ignored=(1, 3))
    before = digest(snapshot)
    result = execute(analysis_command(cli, data, directory, config, snapshot, "constant;tested"))
    fold_histories(directory, [1], 2, 2)
    assert not (directory / "feature_set0_fold0").exists()
    assert "constant" in (result.stdout + result.stderr).lower() or "ignored" in (result.stdout + result.stderr).lower()
    assert digest(snapshot) == before


def test_model_based_eval_uses_best_baseline_prefix_not_completed_training_length(cli, tmp_path):
    x = np.column_stack((np.linspace(-2, 2, 64), np.arange(64) % 2)).astype(np.float32)
    data = write_data(tmp_path / "data", x, x[:, 0], x, -x[:, 0])
    config = settings(**{"-i": 12, "--learning-rate": .4, "--use-best-model": "true"})
    directory = tmp_path / "run"
    snapshot, model = fit_baseline(cli, data, directory, config)
    base = history(directory / "test_error.tsv")
    best_count = int(np.argmin(base[2][:, 0])) + 1
    assert 1 <= best_count == model.tree_count_ < 12
    before = digest(snapshot)
    execute(analysis_command(cli, data, directory, config, snapshot, offset=1, count=1, size=1, full=True))
    logs = fold_histories(directory, [0], 1, 1)
    np.testing.assert_allclose(logs[0, 0, "test"][1], base[2][best_count - 1:best_count], rtol=6e-6, atol=3e-7)
    result = execute(analysis_command(cli, data, tmp_path / "bad_offset", config, snapshot,
                                     offset=best_count + 1, count=1, size=1, full=True), ok=False)
    assert "offset" in (result.stdout + result.stderr).lower()
    assert digest(snapshot) == before


def strip_analysis_history(snapshot, destination, *, trees, permutations, leaf_capacity, dimension=1):
    """Remove only the validated final optional MMB1 record to recreate v6."""
    payload = snapshot.read_bytes()
    tag = struct.pack("<I", 0x4D4D4231)
    positions = []
    at = payload.find(tag)
    while at >= 0:
        if at + 24 <= len(payload):
            stored_tag, p, capacity, dim, count = struct.unpack_from("<IIIIQ", payload, at)
            if (stored_tag == 0x4D4D4231 and p == permutations and capacity == leaf_capacity
                    and dim == dimension and count == trees * permutations * leaf_capacity * dimension
                    and at + 24 + 4 * count == len(payload)):
                positions.append(at)
        at = payload.find(tag, at + 1)
    assert len(positions) == 1, "Expected exactly one structurally valid final per-permutation model-history record"
    destination.write_bytes(payload[:positions[0]])


@pytest.mark.parametrize("permutations", (1, 4))
def test_model_based_eval_legacy_snapshot_policy_preserves_ordinary_resume(cli, tmp_path, permutations):
    data = problem(tmp_path / "data", categorical=permutations == 4)
    config = settings(**{"--permutations": permutations, "--depth": 2})
    directory = tmp_path / "run"
    snapshot, _ = fit_baseline(cli, data, directory, config)
    legacy = directory / "legacy.snapshot"
    strip_analysis_history(snapshot, legacy, trees=8, permutations=permutations, leaf_capacity=4)
    before = digest(legacy)
    args = analysis_command(cli, data, directory, config, legacy, full=True)
    result = execute(args, ok=permutations == 1)
    if permutations == 1:
        fold_histories(directory, [0], 2, 2)
    else:
        text = (result.stdout + result.stderr).lower()
        assert "snapshot" in text and ("history" in text or "permutation" in text)
    assert digest(legacy) == before
    resumed_path = directory / "resumed.cbm"
    resume = config | {"-i": 10, "--snapshot-file": str(legacy), "--model-file": str(resumed_path)}
    execute(command(cli, "fit", data, directory, resume))
    assert CatBoost().load_model(resumed_path).tree_count_ == 10


@pytest.mark.parametrize("extra,message", (
    ({"--data-partition": "FeatureParallel"}, "(?i)DocParallel|feature.parallel"),
    ({"--boosting-type": "Ordered", "--data-partition": "FeatureParallel"}, "(?i)DocParallel|feature.parallel|Ordered"),
    ({"--features-to-evaluate": ""}, "(?i)features|empty|format|no.*evaluate"),
    ({"--features-to-evaluate": "99"}, "(?i)feature.*(large|range|count|index)|features"),
    ({"--features-to-evaluate": "tested", "--ignore-features": "tested"}, "(?i)ignored"),
    ({"--offset": 9}, "(?i)offset|baseline"),
    ({"--offset": 3, "--experiment-count": 2, "--experiment-size": 2}, "(?i)offset|experiment"),
    ({"--experiment-count": 0}, "(?i)experiment|positive|count"),
    ({"--experiment-size": 0}, "(?i)experiment|positive|size"),
    ({"--offset": -1}, "(?i)offset|positive"),
))
def test_model_based_eval_invalid_options_fail_without_changing_baseline(cli, tmp_path, extra, message):
    data = problem(tmp_path / "data", "Logloss")
    config = settings(**{"--loss-function": "Logloss"})
    directory = tmp_path / "run"
    snapshot, _ = fit_baseline(cli, data, directory, config)
    before = digest(snapshot)
    args = analysis_command(cli, data, directory, config, snapshot, full=True)
    for key, value in extra.items():
        if key in args:
            args[args.index(key) + 1] = str(value)
        else:
            args.extend([key, str(value)])
    result = execute(args, ok=False)
    import re
    assert re.search(message, result.stdout + result.stderr), result.stdout + result.stderr
    assert digest(snapshot) == before
    assert not list(directory.glob("feature_set*_fold*"))


@pytest.mark.parametrize("loss", ("MultiRMSE", "MultiLogloss", "MultiCrossEntropy"))
def test_model_based_eval_rejects_multitarget_objectives(cli, tmp_path, loss):
    data = problem(tmp_path / "baseline_data")
    config = settings()
    directory = tmp_path / "run"
    snapshot, _ = fit_baseline(cli, data, directory, config)
    before = digest(snapshot)
    multi = problem(tmp_path / "multitarget_data")
    for path in (multi.train, multi.test):
        rows = []
        for line in path.read_text().splitlines():
            fields = line.split("\t")
            y = float(fields[0])
            if loss == "MultiLogloss":
                targets = [str(float(y > 0)), str(float(y > .5))]
            elif loss == "MultiCrossEntropy":
                targets = [str(.1 + .8 / (1 + np.exp(-y))), str(.1 + .8 / (1 + np.exp(y)))]
            else:
                targets = [str(y), str(.5 * y + .1)]
            rows.append("\t".join(targets + fields[1:]))
        path.write_text("\n".join(rows) + "\n")
    original = multi.cd.read_text().splitlines()
    columns = ["0\tTarget", "1\tTarget"]
    for line in original[1:]:
        index, description = line.split("\t", 1)
        columns.append(f"{int(index) + 1}\t{description}")
    multi.cd.write_text("\n".join(columns) + "\n")
    result = execute(analysis_command(cli, multi, directory, config | {"--loss-function": loss},
                                     snapshot, full=True), ok=False)
    assert "multitarget" in (result.stdout + result.stderr).lower()
    assert digest(snapshot) == before
    assert not list(directory.glob("feature_set*_fold*"))


@pytest.mark.parametrize("kind", ("Text", "NumVector"))
def test_model_based_eval_rejects_text_and_embedding_estimators(cli, tmp_path, kind):
    data = problem(tmp_path / "baseline_data")
    config = settings()
    directory = tmp_path / "run"
    snapshot, _ = fit_baseline(cli, data, directory, config)
    before = digest(snapshot)
    unsupported = problem(tmp_path / "unsupported_data")
    for path in (unsupported.train, unsupported.test):
        rows = []
        for row, line in enumerate(path.read_text().splitlines()):
            fields = line.split("\t")
            fields[2] = ("metal native feature analysis shared words " + ("regression" if row % 2 else "classification")
                         if kind == "Text" else f"{fields[2]};{row % 3};{row % 5}")
            rows.append("\t".join(fields))
        path.write_text("\n".join(rows) + "\n")
    unsupported.cd.write_text(unsupported.cd.read_text().replace("2\tNum\tretained", f"2\t{kind}\tretained"))
    if kind == "Text":
        config["--feature-calcers"] = "BoW"
    result = execute(analysis_command(cli, unsupported, directory, config, snapshot, full=True), ok=False)
    message = (result.stdout + result.stderr).lower()
    assert "model-based evaluation" in message and ("text" in message or "embedding" in message), message
    assert digest(snapshot) == before
    assert not list(directory.glob("feature_set*_fold*"))


@pytest.mark.parametrize("kind", ("missing", "truncated", "changed_data", "missing_eval", "multiple_eval"))
def test_model_based_eval_requires_a_compatible_existing_snapshot_and_one_eval(cli, tmp_path, kind):
    data = problem(tmp_path / "data")
    config = settings()
    directory = tmp_path / "run"
    snapshot, _ = fit_baseline(cli, data, directory, config)
    before = digest(snapshot)
    chosen = snapshot
    if kind == "missing":
        chosen = directory / "missing.snapshot"
    elif kind == "truncated":
        chosen = directory / "truncated.snapshot"
        chosen.write_bytes(snapshot.read_bytes()[:-7])
    elif kind == "changed_data":
        lines = data.train.read_text().splitlines()
        values = lines[0].split("\t")
        values[0] = str(float(values[0]) + .125)
        lines[0] = "\t".join(values)
        data.train.write_text("\n".join(lines) + "\n")
    args = analysis_command(cli, data, directory, config, chosen, full=True)
    if kind == "missing_eval":
        index = args.index("-t")
        del args[index:index + 2]
    elif kind == "multiple_eval":
        args[args.index("-t") + 1] = f"{data.test},{data.test}"
    result = execute(args, ok=False)
    text = (result.stdout + result.stderr).lower()
    assert any(word in text for word in ("snapshot", "eval", "test", "data", "load", "stream")), text
    assert digest(snapshot) == before
    assert not list(directory.glob("feature_set*_fold*"))


def test_model_based_eval_output_failure_and_interruption_leave_baseline_reusable(cli, tmp_path):
    data = problem(tmp_path / "data")
    config = settings(**{"-i": 32, "--depth": 2})
    directory = tmp_path / "run"
    snapshot, _ = fit_baseline(cli, data, directory, config)
    before = digest(snapshot)
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    (blocked / "feature_set0_fold0").write_text("occupied by a file\n")
    args = analysis_command(cli, data, blocked, config, snapshot, "1;2", offset=24, count=4, size=6, full=True)
    execute(args, ok=False)
    assert digest(snapshot) == before
    interrupted = tmp_path / "interrupted"
    args = analysis_command(cli, data, interrupted, config, snapshot, "1;2", offset=24, count=4, size=6, full=True)
    with (tmp_path / "interruption.stdout").open("w") as stdout, (tmp_path / "interruption.stderr").open("w") as stderr:
        process = subprocess.Popen(args, stdout=stdout, stderr=stderr)
        deadline = time.monotonic() + 90
        observed = interrupted / "feature_set0_fold0" / "test_error.tsv"
        try:
            while process.poll() is None and time.monotonic() < deadline:
                if observed.is_file() and len(observed.read_text().splitlines()) >= 2:
                    process.send_signal(signal.SIGINT)
                    break
                time.sleep(.002)
            else:
                pytest.fail("Analysis completed or failed before a running experiment could be interrupted")
            assert process.wait(timeout=30) != 0
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
    assert digest(snapshot) == before
    execute(args)
    actual = fold_histories(interrupted, [0, 1], 4, 6)
    clean = tmp_path / "clean"
    execute(analysis_command(cli, data, clean, config, snapshot, "1;2", offset=24, count=4, size=6, full=True))
    expected = fold_histories(clean, [0, 1], 4, 6)
    for key, (columns, values) in actual.items():
        assert columns == expected[key][0]
        np.testing.assert_array_equal(values, expected[key][1])
    assert digest(snapshot) == before
