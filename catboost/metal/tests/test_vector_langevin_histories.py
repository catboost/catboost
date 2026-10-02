"""CUDA vector leaf-noise ordering across histories, including greedy models."""

import ctypes as ct

import numpy as np
import pytest

from catboost_metal import _multiclass
from test_greedy_vector_training import route as greedy_route
from test_vector_langevin import (
    ACCEPTED_GRADIENT, CACHE, INITIAL_GRADIENT, INITIAL_HESSIAN, OBJECTIVES,
    SEARCH, TRIAL_GRADIENT, Noise, arguments, no_cpu_fit, pytestmark, reference,
    route as symmetric_route,
)


GREEDY_OBJECTIVES = ("MultiClass", "MultiClassOneVsAll", "RMSEWithUncertainty")
POLICIES = ("Depthwise", "Lossguide", "Region")
HISTORY_CASES = [(loss, "SymmetricTree", "Newton") for loss in OBJECTIVES]
HISTORY_CASES += [(loss, policy, "Newton") for loss in GREEDY_OBJECTIVES for policy in POLICIES]
HISTORY_CASES += [("MultiClass", policy, "Gradient") for policy in ("SymmetricTree", "Lossguide")]


def history_problem(objective="MultiClass", policy="SymmetricTree", method="Newton", *, depth=0):
    args = arguments(objective, method, leaf_iterations=2, depth=depth,
        grow_policy=policy, max_leaves=2, bootstrap_type="Bernoulli", subsample=.8)
    rows, dimensions = len(args["targets"]), args["classes"]
    # Distinct bin membership matters in the fixed-prefix tests. Root-only
    # cases retain the same source grid while isolating the leaf oracle.
    banks = np.stack([np.roll(args["bins"], 3 * history, axis=1) for history in range(4)])
    feature_shift = np.linspace(-.18, .24, dimensions, dtype=np.float32)
    row_shift = np.linspace(-.06, .08, rows, dtype=np.float32)[:, None]
    initial = np.stack([
        np.float32(args["initial_predictions"] + history * feature_shift + history * row_shift)
        for history in range(4)
    ])
    if objective == "MultiClass":
        initial[:, :, -1] = 0
    args["initial_predictions"] = initial[0]
    return args, banks, initial


def route(tree, bins):
    return greedy_route(tree, bins) if hasattr(tree, "nodes") else symmetric_route(tree, bins)


def expected_histories(args, tree, banks, initial, *, order=(0, 1, 2, 3)):
    """Independent single-history oracles sharing the source callback stream."""
    noise = Noise()
    count = len(tree.leaf_weights)
    values, weights, predictions = [None] * 4, [None] * 4, [None] * 4
    for history in order:
        ids = route(tree, banks[history])
        local = args | dict(bins=banks[history], initial_predictions=initial[history])
        values[history], weights[history] = reference(local, noise, ids, count)
        predictions[history] = np.float32(initial[history] + values[history][ids])
    return np.stack(values), np.stack(weights), np.stack(predictions), noise


def check_histories(args, tree, banks, initial, state, noise):
    values, weights, predictions, expected_noise = expected_histories(args, tree, banks, initial)
    np.testing.assert_allclose(state["predictions"], predictions, rtol=1.5e-4, atol=1e-5)
    dimensions = args["classes"] - int(args["objective"] == "MultiClass")
    np.testing.assert_allclose(state["optimization_predictions"], predictions[:, :, :dimensions].transpose(0, 2, 1),
                               rtol=1.5e-4, atol=1e-5)
    np.testing.assert_allclose(tree.leaf_values, values[-1], rtol=1.5e-4, atol=1e-5)
    np.testing.assert_allclose(tree.leaf_weights, weights[-1], rtol=5e-6, atol=5e-6)
    assert noise.events == expected_noise.events
    # Searching history 2 must not consume its leaf noise ahead of history 0.
    _, _, wrong_order, _ = expected_histories(args, tree, banks, initial, order=(2, 0, 1, 3))
    assert np.max(np.abs(predictions - wrong_order)) > 1e-4
    assert np.max(np.abs(values[0] - values[2])) > 1e-4


@pytest.mark.parametrize("objective,policy,method", HISTORY_CASES)
def test_selected_history_search_precedes_leaf_oracles_in_history_order(objective, policy, method):
    args, banks, initial = history_problem(objective, policy, method)
    noise = Noise()
    with _multiclass.Session(**args) as session:
        session.configure_permutations(banks, initial)
        session.select_permutation(2)
        noise.install(session)
        tree = session.step()
        state = session.permutation_state
        assert session.completed_iterations == 1
        assert session.bootstrap_state["iteration_offset"] == 1
    assert len(tree.leaf_weights) == 1
    check_histories(args, tree, banks, initial, state, noise)
    assert noise.seeds == [CACHE, SEARCH]
    assert noise.timeline[:2] == [("seed", CACHE), ("seed", SEARCH)]
    assert [event for event, _ in noise.events] == [
        INITIAL_GRADIENT, INITIAL_HESSIAN, TRIAL_GRADIENT,
        ACCEPTED_GRADIENT, TRIAL_GRADIENT, ACCEPTED_GRADIENT,
    ] * 4


def fixed_prefix(session, features):
    values = np.asarray(features, dtype=np.uint32)
    function = session._lib.cbm_multiclass_session_set_fixed_splits
    function.argtypes = [ct.c_void_p, ct.c_uint32, ct.POINTER(ct.c_uint32), ct.c_char_p, ct.c_size_t]
    function.restype = ct.c_int
    error = ct.create_string_buffer(2048)
    code = function(session._handle, len(values), values.ctypes.data_as(ct.POINTER(ct.c_uint32)),
                    error, len(error))
    assert code == 0, error.value.decode()


@pytest.mark.parametrize("policy", POLICIES)
def test_greedy_fixed_prefix_skips_search_seed_but_preserves_history_leaf_order(policy):
    args, banks, initial = history_problem("MultiClass", policy, depth=1)
    noise = Noise()
    with _multiclass.Session(**args) as session:
        session.configure_permutations(banks, initial)
        session.select_permutation(2)
        fixed_prefix(session, [0])
        noise.install(session)
        tree = session.step()
        state = session.permutation_state
    assert len(tree.leaf_weights) == 2
    assert tree.nodes[0, :3].tolist() == [0, 0, 0]
    check_histories(args, tree, banks, initial, state, noise)
    assert noise.seeds == [CACHE]
    assert noise.timeline[0] == ("seed", CACHE)
    assert all(item[0] == "noise" for item in noise.timeline[1:])


@pytest.mark.parametrize("policy", ("SymmetricTree", "Lossguide"))
def test_failure_in_third_history_restores_every_cursor_and_retry_order(policy):
    args, banks, initial = history_problem("MultiClass", policy)
    noise = Noise()
    current_history, fail_once = [-1], [True]

    def callback(event, count):
        if event == INITIAL_GRADIENT:
            current_history[0] += 1
        if event == TRIAL_GRADIENT and current_history[0] == 2 and fail_once[0]:
            fail_once[0] = False
            raise ValueError("injected failure after two histories published")
        return noise.draw(event, count)

    with _multiclass.Session(**args) as session:
        session.configure_permutations(banks, initial)
        session.select_permutation(2)
        session.configure_langevin(2, callback, noise.seed)
        before = session.permutation_state
        bootstrap = session.bootstrap_state
        with pytest.raises(RuntimeError, match="after two histories published"):
            session.step()
        assert current_history[0] == 2 and not fail_once[0]
        assert session.completed_iterations == 0
        assert int(session._info().completed_iterations) == 0
        after = session.permutation_state
        for field in before:
            np.testing.assert_array_equal(after[field], before[field])
        assert session.bootstrap_state == bootstrap
        np.testing.assert_array_equal(session.predictions(), before["predictions"][-1])
        current_history[0] = -1
        noise.reset()
        tree = session.step()
        state = session.permutation_state
        assert session.completed_iterations == 1
        assert int(session._info().completed_iterations) == 1
    check_histories(args, tree, banks, initial, state, noise)
    assert noise.seeds == [CACHE, SEARCH]
