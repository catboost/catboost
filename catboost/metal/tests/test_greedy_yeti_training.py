"""Greedy classic YetiRank target streams and recoverable training cursors."""

import json

import numpy as np
import pytest

from catboost_metal import _greedy
from catboost_metal._greedy_training import run_training
from catboost_metal._greedy_yeti_rng import GreedyYetiRankRng
from catboost_metal._training import _fingerprint, _json, _shared_metric
from test_ordered_rng import ReferenceMt64
from test_yeti_rank_training import problem as symmetric_problem


POLICIES = ("Depthwise", "Lossguide", "Region")


@pytest.fixture(autouse=True)
def no_cpu_training(monkeypatch):
    from catboost import CatBoost

    def forbidden(*args, **kwargs):
        pytest.fail("Greedy YetiRank acceptance must not fit CPU CatBoost")

    monkeypatch.setattr(CatBoost, "_fit", forbidden)


def problem(policy="Depthwise", kind="No", count=1, **options):
    result = symmetric_problem(objective="YetiRank", grow_policy=policy, iterations=4,
        depth=3, max_leaves=6, bias=0., random_seed=0xABCD01234567,
        random_strength=.3, bootstrap_type=kind,
        subsample=.7 if kind in ("Bernoulli", "Poisson") else 1.)
    if count > 1:
        banks = np.stack([result["bins"].copy() for _ in range(count)])
        for index in range(1, count):
            banks[index, 0] = np.roll(banks[index, 0], index * 3)
        result["permutation_bins"] = banks
    result.update(options)
    return result


def assert_same(actual, expected):
    assert len(actual.trees) == len(expected.trees)
    for first, second in zip(actual.trees, expected.trees):
        for field in ("nodes", "leaf_values", "leaf_weights"):
            np.testing.assert_array_equal(getattr(first, field), getattr(second, field))
    for field in ("predictions", "loss", "eval_predictions"):
        np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))
    assert actual.evals_result == expected.evals_result
    assert actual.stats["yeti_rng"] == expected.stats["yeti_rng"]


@pytest.mark.parametrize("kind", ("No", "Bayesian", "Bernoulli", "Poisson"))
@pytest.mark.parametrize("count", (1, 4))
@pytest.mark.parametrize("leaf_iterations", (1, 3))
def test_rng_uses_actual_greedy_search_count_and_restores_exactly(kind, count, leaf_iterations):
    seed = 0xFEABCDEF01234567
    expected = ReferenceMt64(seed)
    expected.next()  # Constructor BaseIterationSeed.
    rng = GreedyYetiRankRng(seed, kind, leaf_iterations, dataset_permutations=count)
    for iteration, attempts in enumerate((0, 1, 17, 35, 2)):
        assert rng.begin() == expected.next()
        if iteration == 0 and kind != "No":
            expected.advance(65537)
        expected.advance(attempts)
        assert rng.leaves(attempts) == [expected.next() for _ in range(count * (leaf_iterations + (leaf_iterations > 1)))]
        rng.complete()
        state = rng.state()
        assert state["learner"] == "greedy_v1"
        assert state["words"] == expected.words and state["index"] == expected.index
        rng = GreedyYetiRankRng(seed, kind, leaf_iterations, dataset_permutations=count,
            iteration_offset=iteration + 1, initial_state=json.loads(json.dumps(state)))


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("count", (1, 4))
@pytest.mark.parametrize("kind", ("No", "Bayesian", "Bernoulli", "Poisson"))
def test_snapshot_retains_every_history_rng_and_extended_continuation(tmp_path, policy, count, kind):
    args = problem(policy, kind, count)
    args.update(eval_bins=args["bins"], eval_targets=args["targets"],
                eval_weight=args["sample_weight"], eval_group_offsets=args["group_offsets"],
                subgroup_hashes=np.arange(len(args["targets"]), dtype=np.uint32) % 3,
                eval_subgroup_hashes=np.arange(len(args["targets"]), dtype=np.uint32) % 4,
                use_best_model=False)
    save = dict(save_snapshot=True, snapshot_interval=0, snapshot_file=tmp_path / "resume.npz")
    direct_save = {**save, "snapshot_file": tmp_path / "direct.npz"}
    direct = run_training(**args, **direct_save)
    seen = []

    def callback(info):
        seen.append(info.iteration)
        return info.iteration < 2

    partial = run_training(**args, **save, callback=callback)
    assert seen == [1, 2] and partial.trained_iterations == 2
    resumed = run_training(**args, **save)
    assert resumed.resumed_iterations == 2
    assert_same(resumed, direct)
    assert resumed.selected_metric == "PFound" and resumed.stats["metric_maximized"]
    expected_metric = _shared_metric("PFound", resumed.eval_predictions, args["eval_targets"],
        args["eval_weight"], args["eval_group_offsets"], subgroup_hashes=args["eval_subgroup_hashes"])
    assert resumed.evals_result["validation"]["PFound"][-1] == pytest.approx(expected_metric, abs=1e-12)
    with np.load(save["snapshot_file"], allow_pickle=False) as actual, np.load(direct_save["snapshot_file"], allow_pickle=False) as expected:
        assert set(actual.files) == set(expected.files)
        for name in actual.files:
            assert actual[name].dtype.kind != "O"
            if name != "metadata":
                np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)
        if count > 1:
            assert actual["permutation_predictions"].shape == (count, len(args["targets"]))
    extended = run_training(**{**args, "iterations": 6}, **save)
    fresh = run_training(**{**args, "iterations": 6})
    assert extended.resumed_iterations == 4
    assert_same(extended, fresh)


@pytest.mark.parametrize("change", ("missing", "learner", "words", "index", "completed", "bootstrap", "datasets"))
def test_semantically_corrupt_rng_is_rejected_even_with_valid_snapshot_checksum(tmp_path, monkeypatch, change):
    args = problem(iterations=2)
    path = tmp_path / "state.npz"
    run_training(**args, save_snapshot=True, snapshot_file=path)
    with np.load(path, allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in saved.files if name != "metadata"}
        header = json.loads(saved["metadata"].item())
    state = header["stats"]["yeti_rng"]
    if change == "missing":
        del header["stats"]["yeti_rng"]
    elif change == "learner":
        state["learner"] = "symmetric"
    elif change == "words":
        state["words"] = [0] * 312
    elif change == "index":
        state["index"] = 313
    elif change == "completed":
        state["completed_iterations"] += 1
    elif change == "datasets":
        state["dataset_permutations"] = 4
    else:
        state["bootstrap_initialized"] = True
    header.pop("checksum")
    header["checksum"] = _fingerprint(arrays, header)
    np.savez(path, **arrays, metadata=np.asarray(_json(header)))
    monkeypatch.setattr(_greedy, "build_library", lambda: pytest.fail("Malformed snapshot reached GPU loading"))
    with pytest.raises(ValueError, match="YetiRank|dataset_permutations"):
        # Already-completed snapshots must be validated too, without opening
        # a new session as an incidental way to detect bad RNG state.
        run_training(**args, save_snapshot=True, snapshot_file=path)


@pytest.mark.parametrize("field", ("permutations", "decay", "legacy_prefix_centering", "group_offsets", "subgroup_hashes"))
def test_snapshot_identity_includes_target_options_and_grouping(tmp_path, monkeypatch, field):
    args = problem(iterations=2, subgroup_hashes=np.arange(67, dtype=np.uint32) % 3)
    path = tmp_path / "identity.npz"
    run_training(**args, save_snapshot=True, snapshot_file=path)
    if field == "permutations":
        args[field] += 1
    elif field == "decay":
        args[field] = .7
    elif field == "legacy_prefix_centering":
        args[field] = True
    else:
        args[field] = args[field].copy()
        args[field][1] += 1
    monkeypatch.setattr(_greedy, "build_library", lambda: pytest.fail("Changed snapshot identity reached GPU loading"))
    with pytest.raises(ValueError, match="Snapshot does not match"):
        run_training(**args, save_snapshot=True, snapshot_file=path)


@pytest.mark.parametrize("options", [dict(leaf_estimation_method="Gradient"),
    dict(leaf_estimation_backtracking="Armijo"), dict(leaf_estimation_backtracking="AnyImprovement"),
    dict(permutations=True), dict(permutations=0), dict(permutations=10001),
    dict(decay=np.nan), dict(decay=-.1), dict(decay=1.1), dict(legacy_prefix_centering=1),
    dict(group_offsets=np.arange(68, dtype=np.uint32)), dict(targets=np.full(67, 1.1)),
    dict(initial_rng_state={}), dict(iteration_offset=2), dict(bootstrap_type="MVS")])
def test_invalid_yeti_configuration_fails_before_gpu_loading(monkeypatch, options):
    monkeypatch.setattr(_greedy, "build_library", lambda: pytest.fail("Invalid YetiRank options reached GPU loading"))
    with pytest.raises(ValueError):
        _greedy.TrainingSession(**problem(**options))


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("count", (1, 4))
def test_controller_rng_matches_independent_draw_transcript(policy, count):
    args = problem(policy, "Bayesian", count)
    matrices = args.pop("permutation_bins", None)
    expected = ReferenceMt64(args["random_seed"])
    expected.next()
    with _greedy.TrainingSession(**args, dataset_permutations=count) as session:
        if matrices is not None:
            session.configure_permutations(matrices)
        for iteration in range(args["iterations"]):
            expected.next()  # Weak target.
            tree = session.step()
            if iteration == 0:
                expected.advance(65537)
            expected.advance(tree.stats["yeti_search_attempts"])
            expected.advance(count * (args["leaf_estimation_iterations"] + 1))
            state = session.result().stats["yeti_rng"]
            assert state["words"] == expected.words and state["index"] == expected.index
            assert state["completed_iterations"] == iteration + 1
            assert tree.loss == pytest.approx(_shared_metric("PFound", session.predictions(), args["targets"],
                args["sample_weight"], args["group_offsets"]), abs=1e-12)
