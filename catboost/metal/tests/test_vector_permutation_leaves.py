"""Last completed vector leaf banks versus independently routed cursor deltas.

The additive getter is bound through a separate CDLL object, preserving the
private Python wrapper's ctypes signatures. Reads must not dispatch GPU work.
"""
import ctypes as ct
import platform

import numpy as np
import pytest

from catboost_metal import _multiclass
from test_greedy_vector_training import no_cpu_fit, problem, route as greedy_route
from test_multiclass_permutations import _leaf_ids as symmetric_route
from test_vector_langevin import INITIAL_GRADIENT, TRIAL_GRADIENT
from test_vector_langevin_histories import history_problem


pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64",
    reason="Actual Apple Silicon Metal required",
)

OBJECTIVES = ("MultiClass", "MultiClassOneVsAll", "RMSEWithUncertainty")
POLICIES = ("SymmetricTree", "Depthwise", "Lossguide", "Region")


def leaf_capacity(args):
    policy, depth = args["grow_policy"], args["depth"]
    if policy in ("SymmetricTree", "Depthwise"):
        return 1 << depth
    if policy == "Region":
        return depth + 1
    return min(args["max_leaves"], 1 << min(depth, 16))


class LeafBanks:
    def __init__(self, session, count, capacity):
        self.session, self.count, self.capacity = session, count, capacity
        self.classes = session._params.classes
        # Function pointer argtypes belong to this CDLL object, never _load's
        # shared cached instance used by other multiclass/permutation tests.
        self.library = ct.CDLL(str(session._lib._name))
        self.copy = self.library.cbm_multiclass_session_copy_last_permutation_leaves
        self.copy.argtypes = [ct.c_void_p, ct.c_uint32, ct.c_uint32,
                              ct.POINTER(ct.c_float), ct.c_char_p, ct.c_size_t]
        self.copy.restype = ct.c_int

    def raw(self, count, capacity, output):
        error = ct.create_string_buffer(2048)
        pointer = None if output is None else output.ctypes.data_as(ct.POINTER(ct.c_float))
        code = self.copy(self.session._handle, count, capacity, pointer, error, len(error))
        if code:
            raise RuntimeError(error.value.decode())

    def read(self):
        result = np.full((self.count, self.capacity, self.classes), np.nan, np.float32)
        before = self.session._info()
        self.raw(self.count, self.capacity, result)
        after = self.session._info()
        assert after.completed_iterations == before.completed_iterations
        assert after.stats.kernel_dispatches == before.stats.kernel_dispatches
        assert after.stats.gpu_seconds == before.stats.gpu_seconds
        assert np.isfinite(result).all()
        return result


def route(tree, bins):
    return greedy_route(tree, bins) if hasattr(tree, "nodes") else symmetric_route(tree, bins)


def check_updates(args, tree, banks, before, after, exported):
    count, capacity, classes = exported.shape
    active = len(tree.leaf_weights)
    assert capacity == leaf_capacity(args) and classes == args["classes"]
    assert count == len(banks) and 1 <= active <= capacity
    np.testing.assert_array_equal(exported[-1, :active], tree.leaf_values)
    np.testing.assert_array_equal(exported[:, active:], 0.)
    dimensions = classes - int(args["objective"] == "MultiClass")
    if dimensions < classes:
        np.testing.assert_array_equal(exported[:, :, -1], 0.)
        np.testing.assert_array_equal(after["predictions"][:, :, -1], before["predictions"][:, :, -1])
    for history, bins in enumerate(banks):
        ids = route(tree, bins)
        assert ids.max() < active
        values = exported[history, ids]
        # Compare both independently stored cursors. In particular, subtracting
        # published class gauges cannot stand in for the optimizer's own cursor.
        expected = np.float32(before["predictions"][history] + values)
        np.testing.assert_allclose(after["predictions"][history], expected, rtol=3e-6, atol=5e-7)
        optimizer = np.float32(before["optimization_predictions"][history] + values[:, :dimensions].T)
        np.testing.assert_allclose(after["optimization_predictions"][history], optimizer, rtol=3e-6, atol=5e-7)
        # Float64 subtraction isolates the observed public update, allowing only
        # the rounding incurred when a float32 cursor receives the tree value.
        delta = after["predictions"][history].astype(float) - before["predictions"][history].astype(float)
        scale = max(1., float(np.max(np.abs(before["predictions"][history]))), float(np.max(np.abs(values))))
        np.testing.assert_allclose(delta, values, rtol=3e-6, atol=4 * np.finfo(np.float32).eps * scale)
    assert np.max(np.abs(exported[:, :active])) > 1e-5, "fixture needs a nonzero finalized tree"
    if count > 1:
        if args["leaf_estimation_method"] == "Simple":
            # Simple copies the searched weak model into every history.
            np.testing.assert_array_equal(exported, np.repeat(exported[:1], count, axis=0))
        else:
            assert np.max(np.abs(exported[0] - exported[-1])) > 1e-5


def exercise(objective, policy, count, method, *, configure=True):
    args, banks, initial = problem(objective, policy, count=count, depth=2, max_leaves=3,
                                  iterations=2, leaf_estimation_method=method,
                                  leaf_estimation_iterations=1 if method == "Simple" else 3,
                                  bootstrap_type="Bernoulli", subsample=.73)
    with _multiclass.Session(**args) as session:
        if configure:
            session.configure_permutations(banks, initial_predictions=initial)
        else:
            assert count == 1
        exported = LeafBanks(session, count, leaf_capacity(args))
        for selected in ((2, 0) if count > 1 else (0, 0)):
            if configure:
                session.select_permutation(selected)
            before = session.permutation_state
            tree = session.step()
            actual = exported.read()
            check_updates(args, tree, banks, before, session.permutation_state, actual)
            # Callers receive a copy; modifying it cannot mutate retained state.
            expected = actual.copy(); actual[:] = 1234.
            np.testing.assert_array_equal(exported.read(), expected)


@pytest.mark.parametrize("objective", OBJECTIVES)
@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("count", [1, 4])
def test_finalized_newton_bank_matches_every_history_cursor(objective, policy, count):
    exercise(objective, policy, count, "Newton")


REPRESENTATIVE_METHODS = [(OBJECTIVES[(index + method_index) % len(OBJECTIVES)], policy, method)
                          for index, policy in enumerate(POLICIES)
                          for method_index, method in enumerate(("Gradient", "Simple"))]


@pytest.mark.parametrize("objective,policy,method", REPRESENTATIVE_METHODS)
def test_gradient_and_simple_banks_preserve_each_policy_contract(objective, policy, method):
    exercise(objective, policy, 4, method)


@pytest.mark.parametrize("policy", ["SymmetricTree", "Lossguide"])
def test_unconfigured_single_history_uses_the_same_public_layout(policy):
    exercise("MultiClass", policy, 1, "Newton", configure=False)


@pytest.mark.parametrize("policy", POLICIES)
def test_unused_leaf_capacity_is_zero_padded_for_every_history(policy):
    args, banks, initial = problem("MultiClassOneVsAll", policy, count=4, depth=3,
                                  max_leaves=5, iterations=1, leaf_estimation_iterations=1)
    args.update(candidate_features=np.empty(0, np.uint32), candidate_bins=np.empty(0, np.uint32),
                candidate_types=np.empty(0, np.uint8))
    with _multiclass.Session(**args) as session:
        session.configure_permutations(banks, initial_predictions=initial)
        session.select_permutation(2)
        before = session.permutation_state
        tree = session.step()
        assert len(tree.leaf_weights) == 1
        actual = LeafBanks(session, 4, leaf_capacity(args)).read()
        assert actual.shape[1] > 1
        check_updates(args, tree, banks, before, session.permutation_state, actual)
        np.testing.assert_array_equal(actual[:, 1:], 0.)


@pytest.mark.parametrize("policy", POLICIES)
def test_leaf_getter_rejects_unavailable_or_inexact_geometry_without_writes(policy):
    args, banks, initial = problem("MultiClass", policy, count=4, depth=2, max_leaves=3,
                                  iterations=1, leaf_estimation_iterations=1)
    capacity = leaf_capacity(args)
    with _multiclass.Session(**args) as session:
        session.configure_permutations(banks, initial_predictions=initial)
        reader = LeafBanks(session, 4, capacity)
        # Oversize backing storage makes a mistaken acceptance safe to observe;
        # the ABI must still reject all nonexact shape arguments before writing.
        output = np.full((5, capacity + 1, args["classes"]), 119., np.float32)
        with pytest.raises(RuntimeError):
            reader.raw(4, capacity, output)
        np.testing.assert_array_equal(output, 119.)
        session.select_permutation(1); session.step()
        expected = reader.read()
        for count, leaves in ((0, capacity), (3, capacity), (5, capacity),
                              (4, capacity - 1), (4, capacity + 1)):
            with pytest.raises(RuntimeError):
                reader.raw(count, leaves, output)
            np.testing.assert_array_equal(output, 119.)
        with pytest.raises(RuntimeError):
            reader.raw(4, capacity, None)
        np.testing.assert_array_equal(reader.read(), expected)


@pytest.mark.parametrize("policy", ["SymmetricTree", "Lossguide"])
def test_failed_later_history_keeps_last_successful_bank_and_retry_replaces_it(policy):
    args, banks, initial = history_problem("MultiClass", policy, "Newton", depth=0)
    args["iterations"] = 3
    current_history, fail_once = [-1], [False]

    def noise(event, count):
        if event == INITIAL_GRADIENT:
            current_history[0] += 1
        if event == TRIAL_GRADIENT and current_history[0] == 2 and fail_once[0]:
            fail_once[0] = False
            raise ValueError("injected third-history leaf export failure")
        return np.zeros(count, np.float64)

    with _multiclass.Session(**args) as session:
        session.configure_permutations(banks, initial_predictions=initial)
        session.select_permutation(2)
        session.configure_langevin(2., noise, lambda event: 81733 + event)
        reader = LeafBanks(session, 4, leaf_capacity(args))
        first_before = session.permutation_state
        first = session.step()
        successful = reader.read()
        check_updates(args, first, banks, first_before, session.permutation_state, successful)
        before, bootstrap = session.permutation_state, session.bootstrap_state
        current_history[0], fail_once[0] = -1, True
        with pytest.raises(RuntimeError, match="third-history leaf export failure"):
            session.step()
        assert current_history[0] == 2 and not fail_once[0]
        assert session.completed_iterations == int(session._info().completed_iterations) == 1
        np.testing.assert_array_equal(reader.read(), successful)
        for name, values in before.items():
            np.testing.assert_array_equal(session.permutation_state[name], values)
        assert session.bootstrap_state == bootstrap
        # Retrying is an actual second completed tree, not a stale export of
        # the earlier tree or a partially staged bank from the failed attempt.
        current_history[0] = -1
        second = session.step()
        recovered = reader.read()
        assert session.completed_iterations == 2
        check_updates(args, second, banks, before, session.permutation_state, recovered)
        assert not np.array_equal(recovered, successful)
