"""Actual Metal multiclass permutation cursors; NumPy objective algebra only."""
import platform

import numpy as np
import pytest

from catboost_metal import _multiclass
from test_multiclass import no_cpu_training, reference


pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64",
    reason="Actual Apple Silicon Metal GPU required",
)


def _data(count, objective="MultiClass", **overrides):
    rng = np.random.default_rng(806)
    rows, classes = 192, 3
    labels = np.tile(np.arange(classes, dtype=np.uint32), rows // classes)
    signal = (labels == 0).astype(np.uint8)
    noise = rng.integers(0, 2, rows, dtype=np.uint8)
    bins = np.stack([
        np.stack([signal, noise]) if index == 0 else
        np.stack([noise, signal]) if index == count - 1 else
        np.stack([np.roll(signal, index), np.roll(noise, index * 7)])
        for index in range(count)
    ])
    weights = rng.uniform(.2, 2, rows).astype(np.float32)
    weights[::19] = 0
    initial = rng.normal(0, .12, (count, rows, classes)).astype(np.float32)
    # Nonzero, row-specific anchors must survive coupled-class gauge conversion.
    initial += rng.uniform(-2, 2, (count, rows, 1)).astype(np.float32)
    arguments = dict(
        bins=bins[-1], targets=labels, candidate_features=np.array([0, 1]),
        candidate_bins=np.array([0, 0]), classes=classes, objective=objective,
        sample_weight=weights, iterations=1, depth=1, learning_rate=.2,
        l2_leaf_reg=3., leaf_estimation_method="Newton", score_function="L2",
    )
    arguments.update(overrides)
    return arguments, bins, initial


def _leaf_ids(step, bins):
    ids = np.zeros(bins.shape[1], np.uint32)
    for level in range(step.depth):
        feature, border = step.split_features[level], step.split_bins[level]
        right = bins[feature] == border if step.split_types[level] else bins[feature] > border
        ids |= right.astype(np.uint32) << level
    return ids


def _expected_update(arguments, step, bins, initial):
    objective, classes = arguments["objective"], arguments["classes"]
    dimensions = classes - int(objective == "MultiClass")
    logits = initial[:, :dimensions].astype(np.float64)
    if objective == "MultiClass":
        logits -= initial[:, -1:]
    ids = _leaf_ids(step, bins)
    directions = reference(
        logits.T, arguments["targets"], arguments["sample_weight"], classes,
        objective, ids, 1 << step.depth, arguments["l2_leaf_reg"], "Newton",
    )[-1]
    leaves = np.zeros((1 << step.depth, classes))
    leaves[:, :dimensions] = directions * arguments["learning_rate"]
    return initial + leaves[ids], leaves, np.bincount(
        ids, weights=arguments["sample_weight"], minlength=1 << step.depth,
    )


@pytest.mark.parametrize("count", [2, 4])
@pytest.mark.parametrize("objective", ["MultiClass", "MultiClassOneVsAll"])
@pytest.mark.parametrize("selected_end", ["first", "last"])
def test_shared_structure_has_independent_newton_cursors(count, objective, selected_end):
    arguments, bins, initial = _data(count, objective)
    selected = 0 if selected_end == "first" else count - 1
    with _multiclass.Session(**arguments) as session:
        session.configure_permutations(bins, initial_predictions=initial)
        np.testing.assert_array_equal(session.permutation_state["predictions"], initial)
        session.select_permutation(selected)
        step = session.step()
        state = session.permutation_state
        assert step.depth == 1
        # The informative feature is intentionally different at opposite ends.
        np.testing.assert_array_equal(step.split_features, [int(selected != 0)])
        expected_leaves = []
        for index in range(count):
            prediction, leaves, weights = _expected_update(arguments, step, bins[index], initial[index])
            np.testing.assert_allclose(state["predictions"][index], prediction, rtol=2e-5, atol=3e-6)
            expected_leaves.append(leaves)
        np.testing.assert_allclose(step.leaf_values, expected_leaves[-1], rtol=2e-5, atol=3e-6)
        np.testing.assert_allclose(step.leaf_weights, weights, rtol=2e-6, atol=2e-5)
        assert not np.allclose(expected_leaves[0], expected_leaves[-1])
        np.testing.assert_array_equal(session.predictions(), state["predictions"][-1])
        np.testing.assert_array_equal(session.result().predictions, state["predictions"][-1])
        np.testing.assert_array_equal(state["mvs_lambdas"], np.zeros(count, np.float32))
        np.testing.assert_array_equal(state["mvs_valid"], np.zeros(count, np.uint8))
        if objective == "MultiClass":
            np.testing.assert_array_equal(state["predictions"][:, :, -1], initial[:, :, -1])
            assert not step.leaf_values[:, -1].any()


@pytest.mark.parametrize("objective", ["MultiClass", "MultiClassOneVsAll"])
@pytest.mark.parametrize("bootstrap_type", ["No", "Bayesian"])
def test_single_permutation_matches_ordinary_gpu_session(objective, bootstrap_type):
    arguments, bins, initial = _data(
        1, objective, iterations=3, bootstrap_type=bootstrap_type,
        random_strength=.25, random_seed=702,
    )
    arguments["initial_predictions"] = initial[0]
    ordinary = _multiclass.train(**arguments)
    with _multiclass.Session(**arguments) as session:
        # Omitted initial state inherits the ordinary session's full-class cursor.
        session.configure_permutations(bins)
        session.select_permutation(0)
        for _ in range(arguments["iterations"]):
            session.step()
        result = session.result()
        for key in ("depths", "split_features", "split_bins", "split_types",
                    "leaf_values", "leaf_weights", "predictions", "loss"):
            np.testing.assert_array_equal(getattr(result, key), getattr(ordinary, key))
        np.testing.assert_array_equal(session.permutation_state["predictions"][0], result.predictions)


@pytest.mark.parametrize("count", [2, 4])
@pytest.mark.parametrize("objective", ["MultiClass", "MultiClassOneVsAll"])
def test_permutation_snapshot_resumes_all_cursors_and_bootstrap(count, objective):
    arguments, bins, initial = _data(
        count, objective, iterations=3, bootstrap_type="Bayesian",
        random_strength=.3, random_seed=0x123456789,
    )
    choices = [0, count - 1, 0]
    with _multiclass.Session(**arguments) as session:
        session.configure_permutations(bins, initial_predictions=initial)
        for selected in choices:
            session.select_permutation(selected)
            session.step()
        complete, complete_state = session.result(), session.permutation_state
    with _multiclass.Session(**dict(arguments, iterations=1)) as session:
        session.configure_permutations(bins, initial_predictions=initial)
        session.select_permutation(choices[0])
        session.step()
        state, bootstrap = session.permutation_state, session.bootstrap_state
        assert bootstrap == {"iteration_offset": 1, "mvs_lambda": None}
        saved = state["predictions"].copy()
        state["predictions"][:] = 123
        np.testing.assert_array_equal(session.permutation_state["predictions"], saved)
        state["predictions"] = saved
    with _multiclass.Session(**dict(arguments, iterations=2, iteration_offset=1)) as session:
        session.configure_permutations(
            bins, initial_predictions=state["predictions"],
            mvs_lambdas=state["mvs_lambdas"], mvs_valid=state["mvs_valid"],
        )
        for selected in choices[1:]:
            session.select_permutation(selected)
            session.step()
        continued, continued_state = session.result(), session.permutation_state
        for key in ("depths", "split_features", "split_bins", "split_types"):
            np.testing.assert_array_equal(getattr(continued, key), getattr(complete, key)[1:])
        np.testing.assert_allclose(continued.leaf_values, complete.leaf_values[1:], rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(continued.loss, complete.loss[1:], rtol=2e-6, atol=2e-7)
        np.testing.assert_allclose(continued_state["predictions"], complete_state["predictions"], rtol=2e-5, atol=2e-6)
        assert session.bootstrap_state == {"iteration_offset": 3, "mvs_lambda": None}


def test_bad_permutation_shapes_are_rejected_before_pointer_conversion(monkeypatch):
    arguments, bins, initial = _data(2)
    zeros = np.zeros(2, np.float32)
    bad_initial = initial.copy()
    bad_initial[0, 0, 0] = np.inf
    overflow_gauge = initial.copy()
    overflow_gauge[0, 0] = [np.finfo(np.float32).max, 0, -np.finfo(np.float32).max]
    cases = [
        dict(bins_list=bins[0]), dict(bins_list=bins[:0]),
        dict(bins_list=np.repeat(bins[:1], 65, axis=0)),
        dict(bins_list=bins[:, :1]), dict(bins_list=bins[:, :, :-1]),
        dict(bins_list=bins.astype(np.float32)), dict(bins_list=bins.astype(np.int32) - 1),
        dict(bins_list=bins + 2),
        dict(initial_predictions=initial[:1]), dict(initial_predictions=initial[:, :, :2]),
        dict(initial_predictions=bad_initial), dict(initial_predictions=overflow_gauge),
        dict(mvs_lambdas=zeros), dict(mvs_valid=zeros.astype(np.uint8)),
        dict(mvs_lambdas=np.ones(2), mvs_valid=np.zeros(2, np.uint8)),
        dict(mvs_lambdas=zeros, mvs_valid=np.ones(2, np.uint8)),
        dict(mvs_lambdas=np.zeros(1), mvs_valid=np.zeros(2, np.uint8)),
        dict(mvs_lambdas=zeros, mvs_valid=np.zeros(1, np.uint8)),
        dict(mvs_lambdas=zeros, mvs_valid=np.zeros(2, np.float32)),
    ]

    def forbidden(*args, **kwargs):
        raise AssertionError("Malformed permutation input reached pointer conversion")

    with _multiclass.Session(**arguments) as session:
        with monkeypatch.context() as guard:
            guard.setattr(_multiclass, "_u8", forbidden)
            guard.setattr(_multiclass, "_f32", forbidden)
            for invalid in cases:
                with pytest.raises(ValueError):
                    session.configure_permutations(**dict({"bins_list": bins}, **invalid))
        assert not session.closed
        assert session.completed_iterations == 0
        session.configure_permutations(bins, initial_predictions=initial)
        np.testing.assert_array_equal(session.permutation_state["predictions"], initial)


def test_permutation_configuration_and_selection_lifecycle():
    arguments, bins, initial = _data(2)
    with _multiclass.Session(**arguments) as session:
        with pytest.raises(ValueError, match="Configure"):
            session.select_permutation(0)
        session.configure_permutations(bins, initial_predictions=initial)
        for invalid in (-1, 2, 1.5, True):
            with pytest.raises(ValueError):
                session.select_permutation(invalid)
        with pytest.raises(ValueError, match="once"):
            session.configure_permutations(bins)
        session.select_permutation(0)
        session.step()
        with pytest.raises(ValueError, match="once"):
            session.configure_permutations(bins)
    with pytest.raises(RuntimeError, match="closed"):
        session.select_permutation(0)
    with pytest.raises(RuntimeError, match="closed"):
        _ = session.permutation_state


def test_second_permutation_newton_failure_rolls_back_every_cursor():
    bins = np.array([[0, 1, 2]], np.uint8)
    initial = np.stack([
        np.tile(np.array([.3, -.2, 0], np.float32), (3, 1)),
        np.tile(np.array([-20., 0, 0], np.float32), (3, 1)),
    ])
    arguments = dict(
        bins=bins, targets=[0, 1, 2], candidate_features=[0, 0], candidate_bins=[0, 1],
        classes=3, iterations=2, depth=0, l2_leaf_reg=0, leaf_estimation_iterations=2,
    )
    healthy = _multiclass.train(**dict(arguments, iterations=1, initial_predictions=initial[0]))
    assert not np.allclose(healthy.predictions, initial[0])
    with _multiclass.Session(**arguments) as session:
        session.configure_permutations(np.stack([bins, bins]), initial_predictions=initial)
        session.select_permutation(0)
        before, loss, bootstrap = session.permutation_state, session.result().loss, session.bootstrap_state
        for _ in range(2):
            with pytest.raises(RuntimeError, match="leaf solve failed"):
                session.step()
            for key in before:
                np.testing.assert_array_equal(session.permutation_state[key], before[key])
            np.testing.assert_array_equal(session.predictions(), before["predictions"][-1])
            np.testing.assert_array_equal(session.result().loss, loss)
            assert session.bootstrap_state == bootstrap
            assert session.completed_iterations == 0
            assert not session.closed
