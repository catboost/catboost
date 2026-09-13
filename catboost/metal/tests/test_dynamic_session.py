"""Staged Metal tree growth and dynamic candidate publication regressions.

Existing full-step training is the state-machine equivalence check. Dynamic
structure/leaf expectations below use independent CUDA scalar equations; no
CPU CatBoost trainer supplies expected values.
"""

import ctypes as ct
import math
import platform

import numpy as np
import pytest
from catboost import CatBoost, CatBoostClassifier, CatBoostRegressor, CatBoostRanker

from catboost_metal import _native
from cuda_reference import _score_children
from cuda_scalar_reference import auxiliary_score_children, objective_terms, weighted_loss
from cuda_querywise_reference import query_terms, query_loss, leaf_reference as query_leaf_reference
from test_backtracking import cuda_leaf_walker
from test_bootstrap import uniforms
from test_feature_penalties import ctr_penalties, structure_reference as penalty_structure_reference
from test_permutation_session import assert_result_equal, problem as permutation_problem
from test_pairwise_training import problem as pair_problem, training_terms as pair_terms, leaf_reference as pair_leaf_reference
from test_score_noise import _normal


@pytest.fixture(autouse=True)
def no_cpu_training(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Dynamic Metal tests must not fit CPU CatBoost")
    for kind in (CatBoost, CatBoostClassifier, CatBoostRegressor, CatBoostRanker):
        monkeypatch.setattr(kind, "fit", forbidden)


@pytest.fixture(scope="module")
def metal_device():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("Staged training requires an Apple Silicon Metal GPU")
    return _native.device_info()


PROFILES = [
    ("RMSE", None, "Newton", "No"),
    ("CrossEntropy", None, "Newton", "Armijo"),
    ("Huber", 0.8, "Gradient", "AnyImprovement"),
    ("Quantile", 0.35, "Exact", "No"),
    ("QueryRMSE", None, "Gradient", "Armijo"),
    ("QuerySoftMax", None, "Newton", "AnyImprovement"),
    ("PairLogit", None, "Newton", "Armijo"),
]
SAMPLING = [
    dict(bootstrap_type="No", random_strength=0),
    dict(bootstrap_type="Bayesian", bagging_temperature=1.4, random_strength=0.35),
    dict(bootstrap_type="MVS", subsample=0.6, random_strength=0.35),
]


def problem(profile=PROFILES[0], count=1, **changes):
    objective, parameter, method, mode = profile
    if objective == "PairLogit":
        args = pair_problem()
        matrices = [np.roll(args["bins"], p * 29, axis=1).copy() for p in range(count)]
        cursors = np.stack([args["initial_predictions"] + np.float32(p * 0.03) for p in range(count)])
    else:
        base, matrices, targets, weights, features, borders, cursors = permutation_problem(count, objective)
        if objective == "QuerySoftMax":
            targets = np.exp(targets * np.float32(0.5)).astype(np.float32)
        args = dict(bins=base, targets=targets, sample_weight=weights,
                    candidate_features=features, candidate_bins=borders,
                    initial_predictions=cursors[0])
        if objective in ("QueryRMSE", "QuerySoftMax"):
            args["group_offsets"] = np.array([0, 1, 7, 64, 129, targets.size], np.uint32)
    args.update(iterations=3, depth=3, learning_rate=0.2, l2_leaf_reg=2, bias=0,
                score_function="Cosine", objective=objective, objective_param=parameter,
                leaf_estimation_method=method, leaf_estimation_iterations=4,
                leaf_estimation_backtracking=mode, random_seed=94371)
    args.update(changes)
    return args, matrices, cursors


def configure(session, matrices, cursors, state=None, used=None):
    session.configure_permutations(matrices,
        initial_predictions=cursors if state is None else state["predictions"],
        mvs_lambdas=None if state is None else state["mvs_lambdas"],
        mvs_valid=None if state is None else state["mvs_valid"])
    session.configure_feature_penalties(np.array([0, 11, 73], np.uint32),
                                       feature_weights=[1, 0.7, 1.3], used_features=used)


def staged_step(session, maximum_depth):
    completed = session.completed_iterations
    session.begin_tree()
    selected = []
    for _ in range(maximum_depth + 1):
        info = session.grow_tree()
        assert {"depth", "finished", "has_split", "feature", "bin", "type", "score", "gain"} <= info.keys()
        if info["has_split"]:
            assert np.isfinite([info["score"], info["gain"]]).all()
            selected.append((info["feature"], info["bin"], info["type"]))
        assert info["depth"] == len(selected)
        assert session.completed_iterations == completed
        if info["finished"]:
            break
    else:
        pytest.fail("Staged tree did not stop within the configured depth")
    step = session.finish_tree()
    assert session.completed_iterations == completed + 1
    assert step.depth == len(selected)
    assert list(zip(step.split_features, step.split_bins, step.split_types)) == selected
    return step


def assert_step_equal(actual, expected):
    for name in ("completed_iterations", "finished", "depth", "loss"):
        assert getattr(actual, name) == getattr(expected, name), name
    for name in ("split_features", "split_bins", "split_types", "leaf_values", "leaf_weights"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name), err_msg=name)


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("count", [1, 4])
@pytest.mark.parametrize("sampling", SAMPLING)
def test_staged_growth_is_bitexact_to_step_for_weighted_objectives_and_permutations(metal_device, profile, count, sampling):
    args, matrices, cursors = problem(profile, count, **sampling)
    with _native.Session(**args) as full, _native.Session(**args) as staged:
        configure(full, matrices, cursors)
        configure(staged, matrices, cursors)
        for index in range(args["iterations"]):
            chosen = [2, 0, 1][index] % count
            full.select_permutation(chosen)
            staged.select_permutation(chosen)
            expected = full.step()
            actual = staged_step(staged, args["depth"])
            assert_step_equal(actual, expected)
            assert_result_equal(staged.result(), full.result())
            for key in full.permutation_state:
                np.testing.assert_array_equal(staged.permutation_state[key], full.permutation_state[key])
            np.testing.assert_array_equal(staged.feature_penalty_state["used_features"],
                                          full.feature_penalty_state["used_features"])
            assert staged.bootstrap_state == full.bootstrap_state


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("sampling", SAMPLING[1:])
def test_completed_staged_trees_resume_all_cursors_and_sampler_state_exactly(metal_device, tmp_path, profile, sampling):
    args, matrices, cursors = problem(profile, 4, iterations=5, **sampling)
    choices = [2, 0, 3, 1, 2]

    def run(config, selected, state=None, used=None):
        with _native.Session(**config) as session:
            configure(session, matrices, cursors, state, used)
            for chosen in selected:
                session.select_permutation(chosen)
                staged_step(session, args["depth"])
            return session.result(), session.permutation_state, session.feature_penalty_state["used_features"]

    full, full_state, full_used = run(args, choices)
    first, state, used = run(args, choices[:2])
    path = tmp_path / "completed-staged-state.npz"
    np.savez(path, used_features=used, **state)
    with np.load(path, allow_pickle=False) as saved:
        restored = {key: saved[key].copy() for key in state}
        restored_used = saved["used_features"].copy()
    rest, rest_state, rest_used = run(args | {"iterations": 3, "iteration_offset": 2},
                                     choices[2:], restored, restored_used)
    for name in ("depths", "split_features", "split_bins", "split_types", "leaf_values", "leaf_weights"):
        np.testing.assert_array_equal(np.concatenate([getattr(first, name), getattr(rest, name)]), getattr(full, name))
    np.testing.assert_array_equal(np.concatenate([first.loss, rest.loss[1:]]), full.loss)
    np.testing.assert_array_equal(rest.predictions, full.predictions)
    for key in full_state:
        np.testing.assert_array_equal(rest_state[key], full_state[key])
    np.testing.assert_array_equal(rest_used, full_used)


@pytest.mark.parametrize("depth", [0, 1, 4])
def test_staged_empty_candidate_trees_finish_once_without_changing_structure(metal_device, depth):
    args, matrices, cursors = problem(iterations=2, depth=depth)
    args.update(candidate_features=np.array([], np.uint32), candidate_bins=np.array([], np.uint32))
    with _native.Session(**args) as session:
        for _ in range(2):
            step = staged_step(session, depth)
            assert step.depth == 0
        with pytest.raises(RuntimeError):
            session.begin_tree()


def test_staged_duplicate_split_terminates_without_installing_it_twice(metal_device):
    args, _, _ = problem(iterations=1, depth=4)
    args.update(candidate_features=np.array([0], np.uint32), candidate_bins=np.array([2], np.uint32))
    with _native.Session(**args) as session:
        step = staged_step(session, 4)
    assert step.depth == 1


@pytest.mark.parametrize("operation", [
    lambda s: s.step(), lambda s: s.begin_tree(), lambda s: s.result(),
    lambda s: s.predictions(), lambda s: s.permutation_state,
    lambda s: s.bootstrap_state, lambda s: s.feature_penalty_state,
    lambda s: s.select_permutation(0),
    lambda s: s.configure_permutations([np.zeros((3, 259), np.uint8)]),
    lambda s: s.configure_feature_penalties(np.zeros(3, np.uint32)),
])
def test_open_tree_rejects_incompatible_operations_without_poisoning_session(metal_device, operation):
    args, _, _ = problem(iterations=1, depth=2)
    expected = _native.train(**args)
    with _native.Session(**args) as session:
        session.begin_tree()
        with pytest.raises((ValueError, RuntimeError)):
            operation(session)
        for _ in range(3):
            if session.grow_tree()["finished"]:
                break
        session.finish_tree()
        assert_result_equal(session.result(), expected)


@pytest.mark.parametrize("operation", [lambda s: s.grow_tree(), lambda s: s.finish_tree()])
def test_idle_session_rejects_tree_operations_without_poisoning_session(metal_device, operation):
    args, _, _ = problem(iterations=1)
    with _native.Session(**args) as session:
        with pytest.raises(RuntimeError):
            operation(session)
        staged_step(session, args["depth"])


@pytest.mark.parametrize("depth", [0, 1])
def test_finish_tree_can_explicitly_stop_at_a_shallower_depth(metal_device, depth):
    args, matrices, cursors = problem(count=4, iterations=1, depth=3,
                                     bootstrap_type="MVS", subsample=0.6, random_strength=0.5)
    with _native.Session(**args) as staged, _native.Session(**(args | {"depth": depth})) as full:
        configure(staged, matrices, cursors)
        configure(full, matrices, cursors)
        staged.select_permutation(2)
        full.select_permutation(2)
        expected = full.step()
        staged.begin_tree()
        for _ in range(depth):
            assert staged.grow_tree()["has_split"]
        actual = staged.finish_tree()
        assert_step_equal(actual, expected)
        for key in full.permutation_state:
            np.testing.assert_array_equal(staged.permutation_state[key], full.permutation_state[key])


def test_growing_a_finished_tree_is_idempotent_until_finish(metal_device):
    args, _, _ = problem(iterations=1, depth=1)
    with _native.Session(**args) as session:
        session.begin_tree()
        first = session.grow_tree()
        assert first["finished"] and first["has_split"] and first["depth"] == 1
        repeated = session.grow_tree()
        assert repeated["finished"] and not repeated["has_split"] and repeated["depth"] == 1
        assert session.grow_tree() == repeated
        assert session.finish_tree().depth == 1


class DynamicOracle:
    """Full-feature scalar oracle with publication masks at each depth.

    Derivatives, sampling multipliers and noise scale are frozen once per tree.
    Candidate publication changes only routing/scoring inputs for later levels.
    """

    def __init__(self, args, matrices, cursors, chosen, features, borders, types, counts, feature_weights,
                 feature_flags=None, feature_parallel=True):
        self.args, self.matrices, self.cursors = args, np.asarray(matrices), np.asarray(cursors).copy()
        self.chosen = chosen
        self.features, self.borders, self.types = features, borders, types
        self.counts, self.feature_weights = counts, feature_weights
        self.flags = (np.full(len(counts), 2, np.uint8) if feature_flags is None
                      else np.asarray(feature_flags, np.uint8).copy())
        self.used = np.zeros(len(counts), np.uint8)
        self.feature_parallel = feature_parallel
        self.splits, self.before = [], np.float32(0)
        cursor, targets = self.cursors[chosen], args["targets"]
        weights = np.asarray(args.get("sample_weight", np.ones(targets.size)), np.float32)
        if args["objective"] in ("QueryRMSE", "QuerySoftMax"):
            gradient, hessian, _, _ = query_terms(targets, cursor, weights, args["group_offsets"], args["objective"])
        elif args["objective"] == "PairLogit":
            terms = pair_terms(cursor, args["pair_winners"], args["pair_losers"], args["pair_weights"])
            gradient, hessian, weights = terms["gradients"], terms["curvature"], terms["incident_weights"]
        else:
            _, g, h = objective_terms(targets, cursor, args["objective"], args.get("objective_param"))
            gradient, hessian = weights.astype(np.float64) * g, weights.astype(np.float64) * h
        gradient = gradient.astype(np.float32)
        mass = (hessian if args["score_function"].startswith("Newton") else weights).astype(np.float32)
        weak = np.where(np.abs(gradient) < np.float32(1e-15), 0, gradient / (mass + np.float32(1e-15)))
        variance = np.sum(mass.astype(np.float64) * weak.astype(np.float64) ** 2) / targets.size
        absolute = args.get("iteration_offset", 0)
        decay = 1 / (1 + math.exp(absolute * float(np.float32(args["learning_rate"])) - math.log(targets.size)))
        self.scale = np.float32(math.sqrt(variance) * args.get("random_strength", 0) * decay)
        factors = np.ones(targets.size, np.float32)
        bootstrap = args.get("bootstrap_type", "No")
        if bootstrap != "No":
            draws = uniforms(targets.size, seed=args["random_seed"], iteration=absolute, stream=0)
            if bootstrap == "Bernoulli":
                factors = (draws < np.float32(args["subsample"])).astype(np.float32)
            elif bootstrap == "Bayesian":
                factors = np.power(-np.log(draws.astype(np.float64) + 1e-20),
                                   float(np.float32(args["bagging_temperature"]))).astype(np.float32)
            else:
                raise ValueError("Dynamic independent fixtures use No/Bernoulli/Bayesian sampling")
        self.gradient = (gradient * factors).astype(np.float64)
        self.mass = (mass * factors).astype(np.float64)

    def ids(self, permutation):
        ids = np.zeros(self.matrices.shape[2], np.int64)
        for level, (feature, border, kind) in enumerate(self.splits):
            row = self.matrices[permutation, feature]
            right = row == border if kind else row > border
            ids |= right.astype(np.int64) << level
        return ids

    def grow(self, published, active):
        ids = self.ids(self.chosen)
        penalties = dynamic_penalties(self.counts[:published], self.used[:published],
                                      self.flags[:published], active)
        family = self.args["score_function"].removeprefix("Newton")
        noise = np.zeros(published, np.float32)
        if family == "Cosine":
            noise = np.array([np.float32(_normal(feature, seed=self.args["random_seed"],
                iteration=self.args.get("iteration_offset", 0), stream=len(self.splits) + 1)) * self.scale
                for feature in range(published)], np.float32)
        evaluated = []
        for candidate, (feature, border, kind) in enumerate(zip(self.features, self.borders, self.types)):
            if feature >= published or not active[feature]:
                continue
            row = self.matrices[self.chosen, feature]
            right = row == border if kind else row > border
            children = ids | right.astype(np.int64) << len(self.splits)
            sums = np.bincount(children, weights=self.gradient, minlength=2 << len(self.splits))
            masses = np.bincount(children, weights=self.mass, minlength=2 << len(self.splits))
            raw = (auxiliary_score_children(sums, masses, family) if family in ("SolarL2", "LOOL2")
                   else _score_children(sums, masses, self.args["l2_leaf_reg"], family))
            score = np.float32(np.float32(np.float32(raw) + noise[feature]) * penalties[feature])
            gain = np.float32(np.float32(score - self.before) * np.float32(self.feature_weights[feature]))
            evaluated.append((float(gain), candidate, float(score), int(feature), int(border), int(kind)))
        if not evaluated:
            return None
        gain, _, score, feature, border, kind = min(evaluated)
        self.before = np.float32(score)
        tree_ctr = bool(self.counts[feature] and (self.flags[feature] & 1))
        if tree_ctr or (not self.feature_parallel and self.counts[feature]):
            self.flags[feature] |= 2
        split = feature, border, kind
        added = split not in self.splits
        if added:
            self.splits.append(split)
        if ((tree_ctr and added) or not self.feature_parallel) and self.counts[feature]:
            self.used[feature] = 1
        return dict(feature=feature, bin=border, type=kind, score=score, gain=gain,
                    depth=len(self.splits), has_split=added)

    def finish(self):
        args = self.args
        for p in range(len(self.cursors)):
            ids = self.ids(p)
            if args["objective"] == "PairLogit":
                values, masses, _ = pair_leaf_reference(args, self.cursors[p], ids, 1 << len(self.splits))
            else:
                common = dict(objective=args["objective"], l2_leaf_reg=args["l2_leaf_reg"],
                              leaf_estimation_method=args["leaf_estimation_method"],
                              leaf_estimation_iterations=args["leaf_estimation_iterations"],
                              leaf_estimation_backtracking=args["leaf_estimation_backtracking"])
                if args["objective"] in ("QueryRMSE", "QuerySoftMax"):
                    point, masses, _ = query_leaf_reference(args["targets"], self.cursors[p], args["sample_weight"],
                        args["group_offsets"], ids, 1 << len(self.splits), **common)
                else:
                    point, masses, _ = cuda_leaf_walker(args["targets"], self.cursors[p], args["sample_weight"],
                        ids, 1 << len(self.splits), objective_param=args.get("objective_param"), **common)
                values = point * np.float32(args["learning_rate"])
            self.cursors[p] = (self.cursors[p] + values[ids]).astype(np.float32)
        return self.cursors.copy(), values, masses


def dynamic_penalties(counts, used, flags, active):
    counts, used, flags, active = (np.asarray(v) for v in (counts, used, flags, active))
    dynamic = (flags & 1) != 0
    registered = (flags & 2) != 0
    dynamic_maximum = max(1, int(counts[dynamic & active.astype(bool) & ~used.astype(bool)].max(initial=0)))
    static_maximum = max(dynamic_maximum, int(counts[registered & ~used.astype(bool)].max(initial=0)))
    maximum = np.where(dynamic, dynamic_maximum, static_maximum).astype(np.float32)
    penalized = (counts > 0) & (dynamic | ~used.astype(bool))
    result = np.ones(counts.shape, np.float32)
    base = np.float32(1) + counts[penalized].astype(np.float32) / maximum[penalized]
    result[penalized] = np.power(base.astype(np.float64), -0.5).astype(np.float32)
    return result


def test_dynamic_oracle_with_all_features_matches_existing_independent_penalty_oracle():
    args, matrices, cursors = problem(count=1)
    counts = np.array([0, 11, 73], np.uint32)
    feature_weights = np.array([1, 0.7, 1.3], np.float32)
    types = np.zeros(args["candidate_features"].size, np.uint8)
    expected, _, used, trace = penalty_structure_reference(args["bins"], args["targets"], cursors[0],
        args["sample_weight"], args["candidate_features"], args["candidate_bins"], counts,
        np.zeros(3, np.uint8), 0.5, feature_weights, args)
    oracle = DynamicOracle(args, matrices, cursors, 0, args["candidate_features"], args["candidate_bins"],
                            types, counts, feature_weights, feature_parallel=False)
    for level in range(args["depth"]):
        info = oracle.grow(3, np.ones(3, np.uint8))
        assert info["score"] == pytest.approx(trace[level]["weighted_scores"][trace[level]["winner"]], rel=2e-6)
        if not info["has_split"]:
            break
    assert [(feature, border) for feature, border, _ in oracle.splits] == expected
    np.testing.assert_array_equal(oracle.used, used)


def dynamic_problem(objective, count, bootstrap):
    rng = np.random.default_rng(1931)
    rows = 257
    matrix = np.stack([rng.integers(0, bins, rows, dtype=np.uint8) for bins in (4, 8, 5, 32)])
    matrices = []
    for p in range(count):
        current = matrix.copy()
        current[:, p::5] = np.roll(matrix, p * 23, axis=1)[:, p::5]
        matrices.append(current)
    signal = (1.5 * (matrix[0] > 1) + 2 * (matrix[1] > 3) + 4 * (matrix[2] == 2)
              + 5 * (matrix[3] > 15) + rng.normal(0, 0.15, rows))
    signal -= signal.mean()
    weights = rng.uniform(0.3, 2.5, rows).astype(np.float32)
    weights[::23] = 0
    cursors = rng.normal(0, 0.2, (count, rows)).astype(np.float32)
    features = np.concatenate([np.full(size, feature, np.uint32) for feature, size in enumerate((3, 7, 5, 31))])
    borders = np.concatenate([np.arange(size, dtype=np.uint32) for size in (3, 7, 5, 31)])
    types = (features == 2).astype(np.uint8)
    groups = np.array([0, 3, 19, 129, rows], np.uint32)
    args = dict(bins=matrices[0][:1].copy(), targets=signal.astype(np.float32), sample_weight=weights,
                candidate_features=features[:3], candidate_bins=borders[:3], candidate_types=types[:3],
                initial_predictions=cursors[0], objective=objective, objective_param=None,
                iterations=1, depth=3, learning_rate=0.2, l2_leaf_reg=2, bias=0,
                score_function="Cosine" if bootstrap == "Bayesian" else "L2",
                leaf_estimation_method="Newton", leaf_estimation_iterations=3, leaf_estimation_backtracking="No",
                bootstrap_type=bootstrap, bagging_temperature=1.4, subsample=0.6,
                random_strength=0.6, random_seed=61843, iteration_offset=7)
    if objective in ("QueryRMSE", "PairLogit"):
        args["group_offsets"] = groups
    if objective == "PairLogit":
        winners, losers = [], []
        for start, end in zip(groups[:-1], groups[1:]):
            a = rng.integers(start, end, 193)
            b = start + (a - start + rng.integers(1, end - start, 193)) % (end - start)
            prefer = signal[a] > signal[b]
            winners.extend(np.where(prefer, a, b))
            losers.extend(np.where(prefer, b, a))
        args.pop("sample_weight")
        args.update(pair_winners=np.asarray(winners, np.uint32), pair_losers=np.asarray(losers, np.uint32),
                    pair_weights=rng.lognormal(0, 0.3, len(winners)).astype(np.float32))
    return (args, np.asarray(matrices), cursors, features, borders, types,
            np.array([0, 11, 0, 73], np.uint32), np.array([0.8, 1.2, 0.7, 1.5], np.float32))


def append_block(session, matrices, features, borders, types, counts, weights, first, last):
    candidates = (features >= first) & (features < last)
    return session.append_features(np.ascontiguousarray(matrices[:, first:last]),
        (features[candidates] - np.uint32(first)).astype(np.uint32), borders[candidates],
        candidate_types=types[candidates], ctr_unique_values=counts[first:last],
        feature_weights=weights[first:last])


def assert_growth_matches_oracle(actual, expected):
    assert expected is not None
    for name in ("depth", "has_split", "feature", "bin", "type"):
        assert actual[name] == expected[name], name
    for name in ("score", "gain"):
        assert actual[name] == pytest.approx(expected[name], rel=5e-4, abs=1e-4), name


@pytest.mark.parametrize("objective", ["RMSE", "QueryRMSE", "PairLogit"])
@pytest.mark.parametrize("count", [1, 4])
@pytest.mark.parametrize("bootstrap", ["Bernoulli", "Bayesian"])
def test_midtree_appends_preserve_frozen_sampling_and_each_permutation_leaf_update(metal_device, objective, count, bootstrap):
    args, matrices, cursors, features, borders, types, counts, feature_weights = dynamic_problem(objective, count, bootstrap)
    chosen = 2 % count
    oracle = DynamicOracle(args, matrices, cursors, chosen, features, borders, types, counts, feature_weights,
                            feature_flags=np.array([2, 3, 3, 3], np.uint8))
    with _native.Session(**args) as session:
        session.configure_permutations(matrices[:, :1], initial_predictions=cursors)
        session.configure_feature_penalties(counts[:1], feature_weights=feature_weights[:1])
        session.select_permutation(chosen)
        session.begin_tree()
        assert_growth_matches_oracle(session.grow_tree(), oracle.grow(1, [1]))
        # The first publication adds a numeric feature and a one-hot feature;
        # both have a larger bin range than the original session allocation.
        append_block(session, matrices, features, borders, types, counts, feature_weights, 1, 3)
        assert_growth_matches_oracle(session.grow_tree(), oracle.grow(3, [1, 1, 1]))
        append_block(session, matrices, features, borders, types, counts, feature_weights, 3, 4)
        # Previously chosen masked features must still route the fixed tree.
        session.set_feature_activity(np.array([0, 1, 0, 1], np.uint8))
        assert_growth_matches_oracle(session.grow_tree(), oracle.grow(4, [0, 1, 0, 1]))
        step = session.finish_tree()
        actual_state = session.permutation_state
        assert 0 < session.workspace["estimated_peak_gpu_bytes"] <= 1024 ** 3
        np.testing.assert_array_equal(session.feature_penalty_state["used_features"], oracle.used)
    expected_cursors, expected_values, expected_masses = oracle.finish()
    assert list(zip(step.split_features, step.split_bins, step.split_types)) == oracle.splits
    np.testing.assert_allclose(step.leaf_values, expected_values, rtol=5e-4, atol=7e-5)
    np.testing.assert_allclose(step.leaf_weights, expected_masses, rtol=3e-6, atol=3e-4)
    np.testing.assert_allclose(actual_state["predictions"], expected_cursors, rtol=5e-4, atol=7e-5)
    if objective == "PairLogit":
        terms = pair_terms(expected_cursors[-1], args["pair_winners"], args["pair_losers"], args["pair_weights"])
        expected_loss = terms["objective"][0] / terms["objective"][1]
    elif objective == "QueryRMSE":
        expected_loss = query_loss(args["targets"], expected_cursors[-1], args["sample_weight"], args["group_offsets"], objective)
    else:
        expected_loss = weighted_loss(args["targets"], expected_cursors[-1], args["sample_weight"], objective)
    assert step.loss == pytest.approx(expected_loss, rel=3e-5, abs=3e-5)


def test_dynamic_penalty_maxima_distinguish_active_dynamic_and_registered_static_pools():
    counts = np.array([10, 100, 7, 200], np.uint32)
    flags, active = np.array([2, 3, 1, 3], np.uint8), np.array([1, 1, 1, 0], np.uint8)
    np.testing.assert_allclose(dynamic_penalties(counts, [0, 0, 0, 0], flags, active),
        1 / np.sqrt([1.05, 2, 1.07, 3]), rtol=1e-7)
    # Used dynamic CTRs remain penalized, while leaving the active unused maximum.
    np.testing.assert_allclose(dynamic_penalties(counts, [1, 1, 0, 1], flags, active),
        [1, 1 / np.sqrt(1 + 100 / 7), 1 / np.sqrt(2), 1 / np.sqrt(1 + 200 / 7)], rtol=1e-7)


@pytest.mark.parametrize("score", ["L2", "Cosine"])
def test_inactive_registered_dynamic_ctr_remains_in_static_penalty_maximum(metal_device, score):
    bins = np.array([[0, 0, 1, 1]], np.uint8)
    candidates = np.array([0], np.uint32)
    args = dict(iterations=1, depth=1, learning_rate=1, l2_leaf_reg=1, bias=0,
                score_function=score, objective="RMSE")
    with _native.Session(bins, np.array([-2, -2, 2, 2], np.float32), candidates, candidates, **args) as session:
        session.configure_feature_penalties(np.array([10], np.uint32))
        first = session.append_features(np.stack([np.repeat(bins, 2, axis=0)]),
            np.array([0, 1], np.uint32), np.array([0, 0], np.uint32),
            ctr_unique_values=np.array([100, 7], np.uint32))
        assert first == 1
        session.set_feature_activity(np.array([1, 0, 1], np.uint8))
        step = staged_step(session, 1)
    # Identical raw scores: static c10 uses max100 (including inactive registered
    # dynamic c100), while active dynamic c7 uses max7. Static feature0 wins.
    np.testing.assert_array_equal(step.split_features, [0])


@pytest.mark.parametrize("score", ["L2", "Cosine"])
def test_restored_used_dynamic_ctr_is_still_penalized(metal_device, score):
    bins, candidates = np.array([[0, 0, 1, 1]], np.uint8), np.array([0], np.uint32)
    with _native.Session(bins, np.array([-2, -2, 2, 2], np.float32), candidates, candidates,
            iterations=1, depth=1, learning_rate=1, l2_leaf_reg=1, bias=0, score_function=score) as session:
        session.configure_feature_penalties(np.array([10], np.uint32))
        session.append_features(np.stack([bins]), candidates, candidates,
            ctr_unique_values=np.array([7], np.uint32), used_features=np.array([1], np.uint8))
        step = staged_step(session, 1)
        np.testing.assert_array_equal(session.feature_penalty_state["used_features"], [0, 1])
    # Dynamic c7 was used, so it leaves the unused maximum (now1) but keeps its
    # own penalty. Static c10 uses max10 and wins the identical raw-score tie.
    np.testing.assert_array_equal(step.split_features, [0])


@pytest.mark.parametrize("reason", ["maximum_depth", "duplicate"])
def test_appending_features_does_not_reopen_terminal_tree_stops(metal_device, reason):
    args, matrices, cursors = problem(iterations=1, depth=1 if reason == "maximum_depth" else 3)
    args.update(candidate_features=np.array([0], np.uint32), candidate_bins=np.array([2], np.uint32))
    with _native.Session(**args) as session:
        session.begin_tree()
        assert session.grow_tree()["has_split"]
        if reason == "duplicate":
            stop = session.grow_tree()
            assert stop["finished"] and not stop["has_split"]
        session.append_features(np.asarray(matrices)[:, 1:2], np.array([0], np.uint32), np.array([2], np.uint32))
        terminal = session.grow_tree()
        assert terminal["finished"] and not terminal["has_split"] and terminal["depth"] == 1
        assert session.finish_tree().depth == 1


@pytest.mark.parametrize("restore", ["append", "activity"])
def test_no_active_candidate_stop_can_resume_after_candidates_become_available(metal_device, restore):
    args, matrices, _ = problem(iterations=1, depth=2)
    with _native.Session(**args) as session:
        session.set_feature_activity(np.zeros(3, np.uint8))
        session.begin_tree()
        stop = session.grow_tree()
        assert stop["finished"] and not stop["has_split"] and stop["depth"] == 0
        if restore == "append":
            assert session.append_features(np.asarray(matrices)[:, :1], np.array([0], np.uint32),
                                            np.array([2], np.uint32)) == 3
        else:
            session.set_feature_activity(np.array([1, 0, 0], np.uint8))
        assert session.grow_tree()["has_split"]
        assert session.finish_tree().depth == 1


@pytest.mark.parametrize("change", [
    {"permutation_bins": np.zeros((1, 1, 2), np.uint8)},
    {"permutation_bins": np.zeros((2, 1, 259), np.uint8)},
    {"permutation_bins": np.zeros((1, 1, 259), np.float32)},
    {"permutation_bins": np.full((1, 1, 259), -1, np.int16)},
    {"permutation_bins": np.full((1, 1, 259), 256, np.uint16)},
    {"candidate_features": np.array([1], np.uint32)},
    {"candidate_bins": np.array([255], np.uint32)},
    {"candidate_types": np.array([2], np.uint8)},
    {"ctr_unique_values": np.array([-1], np.int32)},
    {"feature_weights": np.array([-1], np.float32)},
    {"feature_flags": np.array([4], np.uint8)},
    {"used_features": np.array([2], np.uint8)},
    {"bins_per_feature": 0}, {"bins_per_feature": 257}, {"bins_per_feature": 2},
])
def test_invalid_append_is_rejected_in_python_and_leaves_open_tree_usable(metal_device, monkeypatch, change):
    args, matrices, _ = problem(iterations=1, depth=1)
    append = dict(permutation_bins=np.asarray(matrices)[:, :1],
                  candidate_features=np.array([0], np.uint32), candidate_bins=np.array([2], np.uint32)) | change
    with _native.Session(**args) as session:
        session.begin_tree()
        with monkeypatch.context() as patch:
            patch.setattr(session._lib, "cbm_session_append_features",
                          lambda *args: pytest.fail("Invalid append reached native mutation"))
            with pytest.raises((ValueError, TypeError)):
                session.append_features(append.pop("permutation_bins"), append.pop("candidate_features"),
                                        append.pop("candidate_bins"), **append)
        assert session.grow_tree()["has_split"]
        assert session.finish_tree().depth == 1


@pytest.mark.parametrize("active", [[1, 0], [1, 1, 1, 1], [1.0, 1.0, 1.0], [1, 2, 1], [1, -1, 1]])
def test_invalid_activity_is_rejected_in_python_and_leaves_open_tree_usable(metal_device, monkeypatch, active):
    args, _, _ = problem(iterations=1, depth=1)
    with _native.Session(**args) as session:
        session.begin_tree()
        with monkeypatch.context() as patch:
            patch.setattr(session._lib, "cbm_session_set_feature_activity",
                          lambda *args: pytest.fail("Invalid activity reached native mutation"))
            with pytest.raises((ValueError, TypeError)):
                session.set_feature_activity(active)
        assert session.grow_tree()["has_split"]
        assert session.finish_tree().depth == 1


@pytest.mark.parametrize("objective", ["RMSE", "QueryRMSE", "PairLogit"])
@pytest.mark.parametrize("bootstrap", ["Bayesian", "MVS"])
def test_completed_dynamic_tree_restores_feature_metadata_and_all_permutation_state(metal_device, tmp_path, objective, bootstrap):
    args, matrices, cursors, features, borders, types, counts, feature_weights = dynamic_problem(objective, 4, bootstrap)
    args.update(iterations=3, score_function="Cosine", random_strength=0.6)
    choices = [2, 0, 3]

    def run(config, selected, state=None, metadata=None):
        with _native.Session(**config) as session:
            session.configure_permutations(matrices[:, :1],
                initial_predictions=cursors if state is None else state["predictions"],
                mvs_lambdas=None if state is None else state["mvs_lambdas"],
                mvs_valid=None if state is None else state["mvs_valid"])
            session.configure_feature_penalties(counts[:1], feature_weights=feature_weights[:1],
                used_features=None if metadata is None else metadata["used_features"][:1])
            if metadata is not None:
                appended = features > 0
                session.append_features(matrices[:, 1:], features[appended] - np.uint32(1), borders[appended],
                    candidate_types=types[appended], ctr_unique_values=metadata["ctr_unique_values"][1:],
                    feature_weights=metadata["feature_weights"][1:], feature_flags=metadata["feature_flags"][1:],
                    used_features=metadata["used_features"][1:])
                session.set_feature_activity(metadata["active_features"])
            for tree, chosen in enumerate(selected):
                session.select_permutation(chosen)
                if metadata is None and tree == 0:
                    session.begin_tree()
                    assert session.grow_tree()["has_split"]
                    append_block(session, matrices, features, borders, types, counts, feature_weights, 1, 3)
                    assert session.grow_tree()["has_split"]
                    append_block(session, matrices, features, borders, types, counts, feature_weights, 3, 4)
                    session.set_feature_activity(np.array([0, 1, 0, 1], np.uint8))
                    session.grow_tree()
                    with pytest.raises(RuntimeError):
                        _ = session.feature_metadata
                    session.finish_tree()
                else:
                    staged_step(session, args["depth"])
            return session.result(), session.permutation_state, session.feature_metadata

    full, full_state, full_metadata = run(args, choices)
    first, state, metadata = run(args, choices[:1])
    assert set(metadata) == {"ctr_unique_values", "feature_weights", "feature_flags", "used_features", "active_features"}
    np.testing.assert_array_equal(metadata["ctr_unique_values"], counts)
    np.testing.assert_array_equal(metadata["feature_weights"], feature_weights)
    np.testing.assert_array_equal(metadata["feature_flags"], [2, 3, 3, 3])
    np.testing.assert_array_equal(metadata["active_features"], [0, 1, 0, 1])
    path = tmp_path / "dynamic-feature-snapshot.npz"
    np.savez(path, **state, **{f"feature_{key}": value for key, value in metadata.items()})
    with np.load(path, allow_pickle=False) as saved:
        restored_state = {key: saved[key].copy() for key in state}
        restored_metadata = {key: saved[f"feature_{key}"].copy() for key in metadata}
    rest, rest_state, rest_metadata = run(args | {"iterations": 2, "iteration_offset": args["iteration_offset"] + 1},
                                         choices[1:], restored_state, restored_metadata)
    for name in ("depths", "split_features", "split_bins", "split_types", "leaf_values", "leaf_weights"):
        np.testing.assert_array_equal(np.concatenate([getattr(first, name), getattr(rest, name)]), getattr(full, name))
    np.testing.assert_array_equal(np.concatenate([first.loss, rest.loss[1:]]), full.loss)
    np.testing.assert_array_equal(rest.predictions, full.predictions)
    for key in full_state:
        np.testing.assert_array_equal(rest_state[key], full_state[key])
    for key in full_metadata:
        np.testing.assert_array_equal(rest_metadata[key], full_metadata[key])


@pytest.mark.parametrize("depth,count", [(10, 4), (16, 1)])
def test_deep_append_and_activity_reach_requested_depth_with_correct_leaf_statistics(metal_device, depth, count):
    rows = np.arange(1 << 16, dtype=np.uint32)
    bins = ((rows[None, :] >> np.arange(15, -1, -1, dtype=np.int32)[:, None]) & 1).astype(np.uint8)
    targets = np.sum((np.float32(2) * bins - np.float32(1)) * np.arange(16, 0, -1, dtype=np.float32)[:, None], axis=0)
    matrices = np.stack([np.roll(bins, p * 37, axis=1) for p in range(count)])
    zero = np.array([0], np.uint32)
    with _native.Session(bins[:1], targets, zero, zero, iterations=1, depth=depth,
            learning_rate=0.2, l2_leaf_reg=0, bias=0, score_function="L2") as session:
        session.configure_permutations(matrices[:, :1])
        session.select_permutation(0)
        session.begin_tree()
        first = session.grow_tree()
        assert first["has_split"] and first["feature"] == 0
        assert session.append_features(matrices[:, 1:], np.arange(15, dtype=np.uint32),
                                        np.zeros(15, np.uint32)) == 1
        active = np.ones(16, np.uint8)
        active[0] = 0
        session.set_feature_activity(active)
        for level in range(1, depth):
            info = session.grow_tree()
            assert info["has_split"] and info["depth"] == level + 1 and info["feature"] == level
            active[level] = 0
            session.set_feature_activity(active)
        step = session.finish_tree()
        state = session.permutation_state
        assert 0 < session.workspace["estimated_peak_gpu_bytes"] <= 1024 ** 3
    assert step.depth == depth
    np.testing.assert_array_equal(step.split_features, np.arange(depth))
    for p in range(count):
        ids = np.zeros(rows.size, np.int64)
        for level in range(depth):
            ids |= matrices[p, level].astype(np.int64) << level
        masses = np.bincount(ids, minlength=1 << depth)
        gradient = np.bincount(ids, weights=targets.astype(np.float64), minlength=1 << depth)
        values = (gradient / masses).astype(np.float32) * np.float32(0.2)
        np.testing.assert_allclose(state["predictions"][p], values[ids], rtol=3e-6, atol=3e-6)
    np.testing.assert_array_equal(step.leaf_weights, masses)
    np.testing.assert_allclose(step.leaf_values, values, rtol=3e-6, atol=3e-6)


def test_inconsistent_used_unregistered_ctr_restore_rejects_without_poisoning_open_tree(metal_device):
    args, matrices, _ = problem(iterations=1, depth=1)
    with _native.Session(**args) as session:
        session.begin_tree()
        with pytest.raises((ValueError, RuntimeError)):
            session.append_features(np.asarray(matrices)[:, :1], np.array([0], np.uint32), np.array([2], np.uint32),
                ctr_unique_values=np.array([7], np.uint32), feature_flags=np.array([1], np.uint8),
                used_features=np.array([1], np.uint8))
        assert session.grow_tree()["has_split"]
        assert session.finish_tree().depth == 1


def restore_feature_metadata(session, flags, used, active, count=None):
    """Exercise the additive C ABI used by native FeatureParallel snapshots."""
    values = [None if value is None else np.ascontiguousarray(value, dtype=np.uint8)
              for value in (flags, used, active)]
    function = session._lib.cbm_session_restore_feature_metadata
    byte_ptr = ct.POINTER(ct.c_uint8)
    function.argtypes = [ct.c_void_p, ct.c_uint32, byte_ptr, byte_ptr, byte_ptr,
                         ct.c_char_p, ct.c_size_t]
    function.restype = ct.c_int
    error = ct.create_string_buffer(2048)
    pointers = [None if value is None else value.ctypes.data_as(byte_ptr) for value in values]
    if count is None:
        count = next(value.size for value in values if value is not None)
    session._check(function(session._handle, count, *pointers, error, len(error)), error)


@pytest.mark.parametrize("feature_parallel", [False, True])
def test_only_feature_parallel_tree_ctr_winners_enter_used_set(metal_device, feature_parallel):
    bins = np.array([[0, 0, 1, 1]], np.uint8)
    zero = np.array([0], np.uint32)
    with _native.Session(bins, np.array([-2, -2, 2, 2], np.float32), zero, zero,
            iterations=2, depth=3, learning_rate=0.2, l2_leaf_reg=1, bias=0,
            score_function="L2") as session:
        session.configure_feature_penalties(np.array([10], np.uint32))
        if feature_parallel:
            session.set_feature_activity(np.ones(1, np.uint8))
        for _ in range(2):
            # The sole static CTR wins, then terminates through its duplicate.
            assert staged_step(session, 3).depth == 1
            np.testing.assert_array_equal(session.feature_metadata["used_features"],
                                          [0 if feature_parallel else 1])
            np.testing.assert_array_equal(session.feature_metadata["feature_flags"], [2])


def test_restore_metadata_on_original_columns_enables_dynamic_penalty_semantics(metal_device):
    bins = np.array([[0, 0, 1, 1]] * 3, np.uint8)
    with _native.Session(bins, np.array([-2, -2, 2, 2], np.float32),
            np.arange(3, dtype=np.uint32), np.zeros(3, np.uint32),
            iterations=1, depth=1, learning_rate=0.2, l2_leaf_reg=1, bias=0,
            score_function="L2") as session:
        session.configure_feature_penalties(np.array([10, 200, 7], np.uint32))
        restore_feature_metadata(session, [2, 1, 3], [0, 0, 1], [1, 0, 1])
        metadata = session.feature_metadata
        np.testing.assert_array_equal(metadata["feature_flags"], [2, 1, 3])
        np.testing.assert_array_equal(metadata["used_features"], [0, 0, 1])
        np.testing.assert_array_equal(metadata["active_features"], [1, 0, 1])
        # Inactive transient c200 is excluded; used dynamic c7 keeps its size
        # penalty with denominator1. Static c10 uses denominator10 and wins.
        np.testing.assert_array_equal(staged_step(session, 1).split_features, [0])
        np.testing.assert_array_equal(session.feature_metadata["used_features"], [0, 0, 1])


@pytest.mark.parametrize("bootstrap", ["Bayesian", "MVS"])
def test_arbitrary_restored_original_feature_metadata_resumes_all_banks_exactly(metal_device, bootstrap):
    args, matrices, cursors, features, borders, types, counts, weights = dynamic_problem("RMSE", 4, bootstrap)
    args.update(bins=matrices[0], candidate_features=features, candidate_bins=borders,
                candidate_types=types, iterations=3, score_function="Cosine")
    initial_metadata = dict(feature_flags=np.array([2, 1, 2, 3], np.uint8),
                            used_features=np.array([0, 0, 0, 1], np.uint8),
                            active_features=np.array([0, 1, 1, 1], np.uint8))

    def run(config, choices, state=None, metadata=initial_metadata):
        with _native.Session(**config) as session:
            session.configure_permutations(matrices,
                initial_predictions=cursors if state is None else state["predictions"],
                mvs_lambdas=None if state is None else state["mvs_lambdas"],
                mvs_valid=None if state is None else state["mvs_valid"])
            session.configure_feature_penalties(counts, feature_weights=weights)
            restore_feature_metadata(session, metadata["feature_flags"],
                                     metadata["used_features"], metadata["active_features"])
            for chosen in choices:
                session.select_permutation(chosen)
                staged_step(session, config["depth"])
            return session.result(), session.permutation_state, session.feature_metadata

    full, full_state, full_metadata = run(args, [2, 0, 3])
    first, state, metadata = run(args, [2])
    rest, rest_state, rest_metadata = run(args | {"iterations": 2,
        "iteration_offset": args["iteration_offset"] + 1}, [0, 3], state, metadata)
    for name in ("depths", "split_features", "split_bins", "split_types", "leaf_values", "leaf_weights"):
        np.testing.assert_array_equal(np.concatenate([getattr(first, name), getattr(rest, name)]), getattr(full, name))
    np.testing.assert_array_equal(np.concatenate([first.loss, rest.loss[1:]]), full.loss)
    np.testing.assert_array_equal(rest.predictions, full.predictions)
    for key in full_state:
        np.testing.assert_array_equal(rest_state[key], full_state[key])
    for key in full_metadata:
        np.testing.assert_array_equal(rest_metadata[key], full_metadata[key])


@pytest.mark.parametrize("change", [
    {"count": 2}, {"flags": [2, 4, 3]}, {"flags": None},
    {"used": [0, 2, 0]}, {"used": None},
    {"active": [1, 2, 1]}, {"active": None},
    {"flags": [2, 1, 3], "used": [0, 1, 0]},
    {"used": [1, 0, 0]},
])
def test_invalid_metadata_restore_is_transactional_and_preserves_session(metal_device, change):
    args, _, _ = problem(iterations=1, depth=1)
    with _native.Session(**args) as session:
        session.configure_feature_penalties(np.array([0, 11, 73], np.uint32))
        original = session.feature_metadata
        values = dict(flags=[2, 3, 3], used=[0, 0, 0], active=[1, 1, 1]) | change
        with pytest.raises(RuntimeError):
            restore_feature_metadata(session, **values)
        for key in original:
            np.testing.assert_array_equal(session.feature_metadata[key], original[key])
        assert staged_step(session, 1).depth == 1


@pytest.mark.parametrize("when", ["open_tree", "completed_tree"])
def test_metadata_restore_rejects_training_in_progress_without_poisoning_session(metal_device, when):
    args, _, _ = problem(iterations=2, depth=1)
    with _native.Session(**args) as session:
        if when == "open_tree":
            session.begin_tree()
        else:
            staged_step(session, 1)
        with pytest.raises(RuntimeError):
            restore_feature_metadata(session, [2, 2, 2], [0, 0, 0], [1, 1, 1])
        if when == "open_tree":
            assert session.grow_tree()["has_split"]
            assert session.finish_tree().depth == 1
        assert staged_step(session, 1).depth == 1
