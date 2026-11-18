"""Read-only last-tree export replays every resident DocParallel cursor exactly."""
import ctypes as ct
import platform

import numpy as np
import pytest

from catboost_metal import _native, _yeti
from cuda_scalar_reference import exact_leaf_value
from test_permutation_session import problem, options, fit_fixed_structure
from test_regularization import configure
from test_scalar_langevin import Noise, open_session


pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64", reason="Apple GPU required")


@pytest.fixture(autouse=True)
def no_cpu_fit(monkeypatch):
    from catboost import CatBoost
    def forbidden(*args, **kwargs):
        raise AssertionError("Permutation leaf export checks must not fit CPU CatBoost")
    monkeypatch.setattr(CatBoost, "_fit", forbidden)


def copy_raw(session, output, count=None, max_leaves=None, handle=None):
    function = session._lib.cbm_session_copy_last_permutation_leaves
    function.argtypes = [ct.c_void_p, ct.c_uint32, ct.c_uint32,
                         ct.POINTER(ct.c_float), ct.c_char_p, ct.c_size_t]
    function.restype = ct.c_int
    error = ct.create_string_buffer(4096)
    code = function(session._handle if handle is None else handle,
                    session._permutation_count if count is None else count,
                    (1 << session._params.train.depth) if max_leaves is None else max_leaves,
                    None if output is None else _native._f32(output), error, len(error))
    return code, error.value.decode()


def copy_leaves(session):
    output = np.full((session._permutation_count, 1 << session._params.train.depth), np.nan, np.float32)
    code, error = copy_raw(session, output)
    assert code == 0, error
    return output


def leaf_ids(tree, matrix):
    result = np.zeros(matrix.shape[1], np.int64)
    for level, (feature, border, kind) in enumerate(zip(tree.split_features, tree.split_bins, tree.split_types)):
        right = matrix[feature] == border if kind else matrix[feature] > border
        result |= right.astype(np.int64) << level
    return result


def check_replay(session, matrices, before, tree):
    after = session.permutation_state["predictions"].copy()
    stats = session.result().stats.copy()
    actual = copy_leaves(session)
    for permutation, matrix in enumerate(matrices):
        expected = np.float32(before[permutation] + actual[permutation, leaf_ids(tree, matrix)])
        np.testing.assert_array_equal(after[permutation], expected)
    np.testing.assert_array_equal(actual[-1, :1 << tree.depth], tree.leaf_values)
    np.testing.assert_array_equal(actual[:, 1 << tree.depth:], 0)
    # Repeated reads neither mutate the retained tree nor issue GPU work.
    np.testing.assert_array_equal(copy_leaves(session), actual)
    np.testing.assert_array_equal(session.permutation_state["predictions"], after)
    for key in ("kernel_dispatches", "gpu_seconds"):
        assert session.result().stats[key] == stats[key]
    return actual


@pytest.mark.parametrize("route", ["Newton", "Gradient", "Exact", "Simple", "Regularized", "Langevin"])
@pytest.mark.parametrize("configured_count", [None, 4])
def test_finalized_values_replay_every_history_and_export_bank(route, configured_count):
    count = configured_count or 1
    original, matrices, targets, weights, features, borders, initial = problem(count, rows=71)
    objective = "Quantile" if route == "Exact" else "RMSE"
    method = route if route in ("Newton", "Gradient", "Exact", "Simple") else "Newton"
    params = options(iterations=3, depth=2, learning_rate=.137, objective=objective,
                     objective_param=.7 if route == "Exact" else None,
                     leaf_estimation_method=method,
                     leaf_estimation_iterations=1 if route in ("Simple", "Exact") else 3,
                     leaf_estimation_backtracking="Armijo" if route == "Regularized" else "No")
    with open_session(dict(bins=original, targets=targets, candidate_features=features,
                           candidate_bins=borders, **params, sample_weight=weights,
                           initial_predictions=initial[0])) as session:
        if configured_count:
            session.configure_permutations(matrices, initial_predictions=initial)
        if route == "Regularized":
            configure(session, ridge=True)
        if route == "Langevin":
            noise = Noise()
            noise.install(session)
        previous_export = None
        for selected in [count - 1, 0, min(1, count - 1)]:
            before = session.permutation_state["predictions"].copy()
            session.select_permutation(selected)
            tree = session.step()
            actual = check_replay(session, matrices, before, tree)
            if count > 1 and route == "Simple":
                np.testing.assert_array_equal(actual, np.repeat(actual[:1], count, axis=0))
            elif count > 1:
                assert any(not np.array_equal(actual[0], values) for values in actual[1:])
            if previous_export is not None:
                assert not np.array_equal(actual, previous_export)
            previous_export = actual.copy()
            if route in ("Newton", "Gradient"):
                # Independent per-bank objective equations also verify that
                # centering/scaling is taken from finalized values, not raw leaves.
                splits = list(zip(tree.split_features.tolist(), tree.split_bins.tolist()))
                for permutation, matrix in enumerate(matrices):
                    _, expected, _ = fit_fixed_structure(matrix, targets, before[permutation], weights, splits, params)
                    np.testing.assert_allclose(actual[permutation, :1 << tree.depth], expected,
                                               rtol=2e-4, atol=5e-5)
            elif route == "Exact":
                for permutation, matrix in enumerate(matrices):
                    ids = leaf_ids(tree, matrix)
                    residual = np.float32(targets - before[permutation])
                    expected = np.array([exact_leaf_value(residual[ids == leaf], weights[ids == leaf],
                        "Quantile", .7) for leaf in range(1 << tree.depth)], np.float32)
                    expected *= np.float32(params["learning_rate"])
                    np.testing.assert_array_equal(actual[permutation, :1 << tree.depth], expected)


@pytest.mark.parametrize("objective", ["QueryRMSE", "QuerySoftMax", "PairLogit"])
def test_query_values_include_each_historys_centering_and_learning_rate(objective):
    original, matrices, targets, weights, features, borders, initial = problem(4, rows=72)
    offsets = np.arange(0, 73, 6, dtype=np.uint32)
    extra = {"group_offsets": offsets}
    if objective == "QuerySoftMax":
        targets = np.abs(targets) + np.float32(.1)
    if objective == "PairLogit":
        winners = np.arange(0, 72, 6, dtype=np.uint32)
        extra = dict(pair_winners=winners, pair_losers=winners + 1,
                     pair_weights=np.linspace(.5, 2, len(winners), dtype=np.float32))
    with _native.Session(original, targets, features, borders,
                         **options(iterations=2, objective=objective, learning_rate=.137),
                         sample_weight=None if objective == "PairLogit" else weights, **extra) as session:
        session.configure_permutations(matrices, initial_predictions=initial)
        for selected in (3, 1):
            before = session.permutation_state["predictions"].copy()
            session.select_permutation(selected)
            check_replay(session, matrices, before, session.step())


def test_yeti_oracles_are_captured_without_consuming_extra_seeds():
    from test_yeti_permutations import inputs
    args, matrices = inputs(leaf_iterations=3)
    args["iterations"] = 2
    with _yeti.Session(**args) as session:
        session.configure_permutations(matrices)
        for iteration, selected in enumerate((3, 0)):
            before = session.permutation_state["predictions"].copy()
            session.select_permutation(selected)
            seeds = [0xdef0123400000011 + iteration * 97 + index for index in range(17)]
            check_replay(session, matrices, before, session.step(seeds))


@pytest.mark.parametrize("method", ["Simple", "Newton"])
def test_coupled_values_replay_each_history(method):
    from test_matrix_permutations import inputs
    args, cls, matrices = inputs("PairLogitPairwise", method, iterations=2)
    with cls(**args) as session:
        session.configure_permutations(matrices)
        for selected in (3, 0):
            before = session.permutation_state["predictions"].copy()
            session.select_permutation(selected)
            check_replay(session, matrices, before, session.step())


@pytest.mark.parametrize("depth,count", [(0, 1), (3, 4), (16, 64)])
def test_stumps_are_zero_padded_through_the_maximum_supported_output(depth, count):
    original = np.zeros((1, 3), np.uint8)
    targets = np.array([1, 3, 5], np.float32)
    weights = np.array([1, 2, 1], np.float32)
    initial = np.repeat(np.arange(count, dtype=np.float32)[:, None] * .125, 3, axis=1)
    with _native.Session(original, targets, np.empty(0, np.uint32), np.empty(0, np.uint32),
                         **options(iterations=1, depth=depth, learning_rate=.125,
                                   l2_leaf_reg=4, leaf_estimation_iterations=1), sample_weight=weights) as session:
        session.configure_permutations([original] * count, initial_predictions=initial)
        tree = session.step()
        assert tree.depth == 0
        actual = check_replay(session, [original] * count, initial, tree)
        # All numbers are binary fractions: this direct RMSE oracle is exact.
        expected = np.float32(np.float32((targets[None] - initial) * weights).sum(axis=1) / np.float32(8)) * np.float32(.125)
        np.testing.assert_array_equal(actual[:, 0], expected)


def test_invalid_dimensions_handles_and_open_tree_leave_output_untouched():
    original, matrices, targets, weights, features, borders, initial = problem(4, rows=31)
    with _native.Session(original, targets, features, borders,
                         **options(iterations=2), sample_weight=weights) as session:
        session.configure_permutations(matrices, initial_predictions=initial)
        output = np.full((4, 4), -912.5, np.float32)
        code, error = copy_raw(session, output)
        assert code != 0 and "finished tree" in error
        np.testing.assert_array_equal(output, -912.5)
        session.step()
        for count, max_leaves in [(0, 4), (3, 4), (5, 4), (4, 0), (4, 2), (4, 8), (2**32 - 1, 2**32 - 1)]:
            code, error = copy_raw(session, output, count, max_leaves)
            assert code != 0 and "dimensions" in error
            np.testing.assert_array_equal(output, -912.5)
        assert copy_raw(session, None)[0] != 0
        assert copy_raw(session, output, handle=ct.c_void_p())[0] != 0
        np.testing.assert_array_equal(output, -912.5)
        session.begin_tree()
        code, error = copy_raw(session, output)
        assert code != 0 and "tree is open" in error
        np.testing.assert_array_equal(output, -912.5)
        while not session.grow_tree()["finished"]:
            pass
        session.finish_tree()
        assert np.isfinite(copy_leaves(session)).all()


def test_feature_parallel_export_is_rejected_after_a_completed_tree():
    original, _, targets, weights, features, borders, _ = problem(1, rows=31)
    with _native.Session(original, targets, features, borders,
                         **options(iterations=1), sample_weight=weights) as session:
        session.set_feature_activity(np.ones(original.shape[0], np.uint8))
        session.step()
        output = np.full((1, 4), -912.5, np.float32)
        code, error = copy_raw(session, output)
        assert code != 0 and "DocParallel" in error
        np.testing.assert_array_equal(output, -912.5)


def test_failed_finish_cannot_export_a_previous_or_partially_estimated_tree():
    from test_yeti_permutations import inputs
    args, matrices = inputs(leaf_iterations=3)
    args["iterations"] = 2
    with _yeti.Session(**args) as session:
        session.configure_permutations(matrices)
        session.step(list(range(17)))
        assert np.isfinite(copy_leaves(session)).all()
        session.begin_tree([719])
        while not session.grow_tree()["finished"]:
            pass
        with pytest.raises(RuntimeError, match="leaf seed schedule"):
            session.finish_tree()  # Missing leaf draws fail after the tree opens.
        output = np.full((4, 1 << args["depth"]), -912.5, np.float32)
        assert copy_raw(session, output)[0] != 0
        np.testing.assert_array_equal(output, -912.5)
