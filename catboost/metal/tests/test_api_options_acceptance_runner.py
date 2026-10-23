"""Host-only release gate inventory and CLI input contract checks."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1] / "examples/api_options_acceptance/smoke.py"
spec = importlib.util.spec_from_file_location("card4_release_smoke", SOURCE)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_inventory_is_explicit_unique_and_has_no_optional_release_cases():
    inventory = runner.inventory()
    cases = inventory["cases"]
    assert inventory == runner.inventory()
    assert inventory["expected_cases_cli"] == 61
    assert inventory["expected_cases_smoke"] == 69
    assert len({case["name"] for case in cases}) == 69
    assert sum(inventory["families_cli"].values()) == 61
    assert sum(inventory["families_smoke"].values()) == 69
    assert {case["family"] for case in cases if case.get("api_only")} == {
        "public-regressor", "public-classifier", "public-ranker", "cv"}
    assert {case["loss"] for case in cases if case["family"] == "simple-vector"} == set(runner.VECTORS)


@pytest.mark.parametrize("case", list(runner.configurations()), ids=lambda case: case["name"])
def test_every_dataset_and_cli_descriptor_roundtrip(case, tmp_path):
    x, y, options = runner.data(case)
    assert x.ndim == 2 and len(x) == len(y) == len(options["weight"])
    assert np.isfinite(y).all() and np.isfinite(options["weight"]).all()
    assert np.all(options["weight"] > 0)
    if "group_id" in options:
        assert len(options["group_id"]) == len(y)
    if "pairs" in options:
        edges = options["pairs"]
        assert edges.shape[1] == 2 and len(edges) == len(options["pairs_weight"])
        np.testing.assert_array_equal(options["group_id"][edges[:, 0]], options["group_id"][edges[:, 1]])
        assert np.all(y[edges[:, 0]] > y[edges[:, 1]])
    runner.write_cli_data(tmp_path, x, y, options)
    columns = [line.split("\t") for line in (tmp_path / "columns.cd").read_text().splitlines()]
    rows = [line.split("\t") for line in (tmp_path / "learn.tsv").read_text().splitlines()]
    assert len(rows) == len(y) and all(len(row) == len(columns) for row in rows)
    assert [int(column[0]) for column in columns] == list(range(len(columns)))
    target_columns = sum(column[1] == "Target" for column in columns)
    targets = np.asarray([[float(value) for value in row[:target_columns]] for row in rows])
    np.testing.assert_array_equal(targets, np.asarray(y).reshape(len(y), -1))
    np.testing.assert_array_equal([float(row[target_columns]) for row in rows], options["weight"])
    first_feature = len(columns) - x.shape[1]
    for i in range(x.shape[1]):
        values = [row[first_feature + i] for row in rows]
        kind = columns[first_feature + i][1]
        if kind == "NumVector":
            np.testing.assert_array_equal([[float(item) for item in value.split(";")] for value in values],
                                          np.stack(x[:, i]))
        elif kind == "Num":
            np.testing.assert_array_equal(np.asarray(values, float), x[:, i].astype(float))
        else:
            np.testing.assert_array_equal(values, x[:, i])
    config = runner.options(case)
    assert config["task_type"] == "GPU"
    assert json.loads(json.dumps(config))["loss_function"] == case["loss"]


def test_named_cli_flags_are_exercised_and_configuration_is_not_mutated():
    required = {"--leaf-estimation-method", "--fixed-binary-splits", "--feature-weights", "--simple-ctr",
                "--rsm", "--fold-size-loss-normalization", "--add-ridge-penalty-for-loss-function",
                "--meta-l2-leaf-exponent", "--meta-l2-leaf-frequency", "--counter-calc-method",
                "--one-hot-max-size", "--langevin", "--diffusion-temperature"}
    actual = set()
    for case in runner.configurations():
        config = runner.options(case)
        before = json.dumps(config, sort_keys=True)
        params, flags = runner.cli_parameters(config)
        assert json.dumps(config, sort_keys=True) == before
        assert params["task_type"] == "GPU"
        actual.update(flag for flag in flags if flag.startswith("--"))
    assert actual == required


def test_learned_prior_fixture_has_identifiable_beta_binomial_counts():
    case = next(case for case in runner.configurations() if case["family"] == "prior")
    x, y, _ = runner.data(case)
    assert [int(y[x[:, 0] == f"cat{i}"].sum()) for i in range(6)] == [3, 8, 13, 21, 29, 35]
    assert [int((x[:, 0] == f"cat{i}").sum()) for i in range(6)] == [40] * 6


@pytest.mark.parametrize("mode", runner.MODES)
def test_full_counter_cli_keeps_distinct_first_eval_frequencies(mode, tmp_path):
    case = next(case for case in runner.configurations() if case["family"] == "full-counter" and case["mode"] == mode)
    learn = runner.data(case)
    evaluation = runner.evaluation_data(case, learn)
    assert learn[0].shape == (120, 1) and evaluation[0].shape == (84, 1)
    combined = np.concatenate((learn[0][:, 0], evaluation[0][:, 0]))
    # Full and final model statistics reverse the two categories around .3.
    assert np.mean(learn[0][:, 0] == "a") < .3 < np.mean(combined == "a")
    assert np.mean(combined == "d") < .3 < np.mean(learn[0][:, 0] == "d")
    assert "eval-only" not in learn[0] and "eval-only" in evaluation[0]
    runner.write_cli_data(tmp_path, *learn)
    runner.write_cli_data(tmp_path, *evaluation, prefix="test")
    assert len((tmp_path / "learn.tsv").read_text().splitlines()) == 120
    assert len((tmp_path / "test.tsv").read_text().splitlines()) == 84
