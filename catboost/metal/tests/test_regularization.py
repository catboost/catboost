"""CUDA option equations checked against additive native Metal consumers.

No CPU trainer is used. Score references widen uploaded float statistics before
per-leaf float accumulation; leaf references implement the CUDA global walker.
"""
import ctypes as ct
import hashlib
from pathlib import Path
import platform
import subprocess

import numpy as np
import pytest

from catboost_metal import _native, _ordered
from cuda_scalar_reference import objective_terms


@pytest.fixture(scope="module", autouse=True)
def apple_silicon():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("Metal runtime requires Apple Silicon")


class Regularization(ct.Structure):
    _fields_ = [(key, ct.c_uint32) for key in ("normalize_score", "normalize_leaf", "add_ridge", "reserved")] + [
        ("meta_l2_exponent", ct.c_float), ("meta_l2_frequency", ct.c_double)]


def configure(session, normalize=False, ridge=False, exponent=1, frequency=0):
    options = Regularization(normalize, normalize, ridge, 0, exponent, frequency)
    function = session._lib.cbm_session_set_regularization
    function.argtypes = [ct.c_void_p, ct.POINTER(Regularization), ct.c_char_p, ct.c_size_t]
    error = ct.create_string_buffer(4096)
    assert function(session._handle, ct.byref(options), error, len(error)) == 0, error.value.decode()


@pytest.fixture(scope="module")
def score_probe():
    root = Path(__file__).resolve().parents[1]
    source = Path(__file__).with_name("regularization_probe.mm")
    headers = [root / "native" / name for name in (
        "metal_kernel_abi.h", "metal_kernels.h", "metal_additional_objective_kernels.h",
        "metal_objective_kernels.h", "metal_backtracking_kernels.h", "metal_streaming_score_kernels.h",
        "metal_dynamic_score_kernels.h", "metal_regularization_kernels.h")]
    digest = hashlib.sha256(source.read_bytes() + b"".join(p.read_bytes() for p in headers)).hexdigest()[:20]
    output = root / ".build" / f"regularization_probe_{digest}.dylib"
    output.parent.mkdir(exist_ok=True)
    if not output.exists():
        subprocess.run(["xcrun", "clang++", "-std=c++17", "-O2", "-fobjc-arc", "-dynamiclib",
            "-Wno-deprecated-declarations", "-framework", "Foundation", "-framework", "Metal",
            str(source), "-o", str(output)], check=True, capture_output=True, text=True)
    library = ct.CDLL(str(output))
    function = library.cbm_regularization_score_probe
    function.argtypes = [ct.c_uint32] * 5 + [ct.c_float] * 2 + [ct.c_void_p] * 6 + [ct.c_char_p, ct.c_uint32]
    function.restype = ct.c_int

    def run(sums, weights, leaf_sums, leaf_weights, score, variant, normalize, exponent, l2):
        arrays = [np.ascontiguousarray(a, np.float32) for a in (sums, weights, leaf_sums, leaf_weights)]
        selected, value, error = ct.c_uint32(), ct.c_float(), ct.create_string_buffer(4096)
        code = function(len(leaf_sums), sums.shape[1], score, variant, normalize, exponent, l2,
            *(a.ctypes.data for a in arrays), ct.byref(selected), ct.byref(value), error, len(error))
        assert code == 0, error.value.decode()
        return selected.value, value.value
    return run


def score_reference(sums, weights, leaf_sums, leaf_weights, score, normalize, exponent, l2):
    result = []
    for feature in range(sums.shape[1]):
        total, numerator, denominator = np.float32(0), 0., 1e-10
        for leaf in range(len(leaf_sums)):
            for value, weight in ((sums[leaf, feature], weights[leaf, feature]),
                    (np.float32(leaf_sums[leaf] - sums[leaf, feature]),
                     max(np.float32(leaf_weights[leaf] - weights[leaf, feature]), 0))):
                value, weight = float(value), float(weight)
                if score in (0, 2):
                    term = -value * value / (weight + l2) if weight > np.float32(1e-20) else 0
                    if exponent != 1 and term:
                        term = -pow(abs(term) / weight, exponent) * weight
                    total = np.float32(float(total) + term)
                elif weight > 0:
                    mean = value / (weight + l2 * (weight if normalize else 1))
                    numerator += value * mean
                    denominator += weight * mean * mean
        result.append(float(total) if score in (0, 2) else -numerator / np.sqrt(denominator))
    return np.asarray(result, np.float32)


@pytest.mark.parametrize("score,normalize,exponent", [(0, False, .5), (2, False, 1.7),
    (1, True, 1), (3, True, 1), (0, True, 1), (1, False, 1),
    (0, False, 0), (2, False, -.5)])
def test_all_scalar_score_paths_follow_cuda_equations(score_probe, score, normalize, exponent):
    rng = np.random.default_rng(2478)
    leaf_weights = rng.integers(8, 48, 7).astype(np.float32)
    weights = (leaf_weights[:, None] * rng.uniform(.05, .95, (7, 19))).astype(np.float32)
    sums = rng.normal(0, 9, weights.shape).astype(np.float32)
    leaf_sums = rng.normal(0, 7, 7).astype(np.float32)
    leaf_weights[-1] = 0; weights[-1] = 0; sums[-1] = 0; leaf_sums[-1] = 0
    expected = score_reference(sums, weights, leaf_sums, leaf_weights, score, normalize, exponent, 2.5)
    for variant in (1, 2, 3):
        selected, value = score_probe(sums, weights, leaf_sums, leaf_weights, score, variant, normalize, exponent, 2.5)
        assert selected == np.argmin(expected)
        np.testing.assert_allclose(value, expected[selected], rtol=4e-6, atol=4e-6)
    if not normalize and exponent == 1:
        assert score_probe(sums, weights, leaf_sums, leaf_weights, score, 0, False, 1, 2.5) == (selected, value)


def leaf_walk(tasks, objective, normalize, ridge, l2, iterations, mode):
    point = np.zeros(len(tasks), np.float32)
    masses = np.array([task[1].sum(dtype=np.float64) for task in tasks])
    def evaluate(candidate):
        values, gradients, diagonals = [], [], []
        for index, task in enumerate(tasks):
            targets, weights = task[:2]
            baseline = task[2] if len(task) > 2 else 0
            raw = np.float32(baseline + np.full(len(targets), candidate[index], np.float32))
            losses, g, h = objective_terms(targets, raw, objective, None)
            values.append(-(weights * losses.astype(np.float32)).sum(dtype=np.float64))
            gradients.append((weights.astype(np.float64) * g).astype(np.float32).sum(dtype=np.float64))
            diagonals.append((weights.astype(np.float64) * h).astype(np.float32).sum(dtype=np.float64))
        values, gradients, diagonals = map(np.asarray, (values, gradients, diagonals))
        if normalize:
            values /= masses; gradients /= masses; diagonals /= masses
        if ridge:
            values -= .5 * l2 * candidate.astype(float) ** 2
            gradients -= l2 * candidate.astype(float)
        return values.sum(), gradients, diagonals + l2
    value, gradient, diagonal = evaluate(point)
    updated, fresh, step = False, True, 1.
    for attempt in range(max(iterations, 100)):
        if attempt >= iterations and (updated or mode == "No"):
            break
        if fresh:
            direction = (gradient / (diagonal + 1e-20)).astype(np.float32)
            dot = gradient @ direction.astype(float)
        trial = (point.astype(float) + step * direction.astype(float)).astype(np.float32)
        trial_value, trial_gradient, trial_diagonal = evaluate(trial)
        threshold = value + (1e-5 * step * dot if mode == "Armijo" else 0)
        if mode == "No" or (np.isfinite(trial_value) and trial_value >= threshold):
            point, value, gradient, diagonal = trial, trial_value, trial_gradient, trial_diagonal
            updated, fresh, step = True, True, 1.
        else:
            step *= .5; fresh = False
    return point


@pytest.mark.parametrize("mode", ["No", "AnyImprovement", "Armijo"])
@pytest.mark.parametrize("normalize,ridge", [(False, True), (True, False), (True, True)])
def test_scalar_leaf_normalization_and_ridge(mode, normalize, ridge):
    targets = np.array([0, 2, 1, 8, 3, 1, 2], np.float32)
    weights = np.array([.5, 1, 2, 3, 1, .25, 4], np.float32)
    empty = np.array([], np.uint32)
    expected = leaf_walk([(targets, weights)], "Poisson", normalize, ridge, .4, 4, mode)
    with _native.Session(np.zeros((1, len(targets)), np.uint8), targets, empty, empty,
        iterations=1, depth=0, learning_rate=1, l2_leaf_reg=.4, bias=0, score_function="Cosine",
        objective="Poisson", sample_weight=weights, leaf_estimation_iterations=4,
        leaf_estimation_backtracking=mode) as session:
        configure(session, normalize, ridge)
        session.step()
        actual = session.result()
    np.testing.assert_allclose(actual.leaf_values[0, 0], expected[0], rtol=8e-6, atol=4e-6)
    np.testing.assert_allclose(actual.leaf_weights[0, 0], weights.sum(dtype=np.float64))


@pytest.mark.parametrize("mode", ["No", "AnyImprovement", "Armijo"])
@pytest.mark.parametrize("normalize", [False, True])
def test_ordered_ridge_uses_each_original_task_mass(mode, normalize):
    targets = np.array([0, 2, 1, 8, 3, 1, 2, 7, 1, 0, 3, 4], np.float32)
    weights = np.array([.5, 1, 2, 3, 1, .25, 4, 2, .5, 1, 3, 2], np.float32)
    empty = np.array([], np.uint32)
    permutations = np.stack((np.arange(len(targets)), np.arange(len(targets))[::-1])).astype(np.uint32)
    with _ordered.Session(np.zeros((1, len(targets)), np.uint8), targets, empty, empty,
        iterations=1, depth=0, learning_rate=1, l2_leaf_reg=.4, bias=0, objective="Poisson",
        sample_weight=weights, leaf_estimation_iterations=5, leaf_estimation_backtracking=mode,
        permutation_count=2, permutations=permutations, min_fold_size=2, fold_size_loss_normalization=normalize) as session:
        function = session._lib.cbm_ordered_set_add_ridge_to_target_function
        function.argtypes = [ct.c_void_p, ct.c_uint32, ct.c_char_p, ct.c_size_t]
        error = ct.create_string_buffer(4096)
        assert function(session._handle, 1, error, len(error)) == 0, error.value.decode()
        descriptors = session.state()["descriptors"]
        tasks = [(targets[permutations[p, :end]], weights[permutations[p, :end]]) for end, _, _, p in descriptors]
        expected = leaf_walk(tasks, "Poisson", normalize, True, .4, 5, mode)
        session.step()
        actual = session.result()
        state = session.state()
    np.testing.assert_allclose(actual.leaf_values[0, 0], expected[-1], rtol=8e-6, atol=4e-6)
    for index, (_, size, offset, _) in enumerate(descriptors):
        np.testing.assert_allclose(state["cursors"][offset:offset + size], expected[index], rtol=8e-6, atol=4e-6)


@pytest.mark.parametrize("mode", ["No", "AnyImprovement", "Armijo"])
def test_full_matrix_ridge_changes_rhs_and_backtracking_value(mode):
    from catboost_metal import _pair_matrix
    from test_pairwise_matrix_training import problem
    from test_pairwise_matrix_kernels import reference
    from test_leaf_matrix_kernels import regularize
    from test_pairwise_training import leaf_ids
    args = problem(iterations=1, depth=2, leaf_estimation_iterations=5,
        leaf_estimation_backtracking=mode, l2_leaf_reg=.8, non_diagonal_regularization=.1)
    args["initial_predictions"] *= 8
    with _pair_matrix.Session(**args) as session:
        configure(session, ridge=True)
        tree = session.step()
    ids = leaf_ids(tree, args["bins"])
    count = 1 << tree.depth
    point = np.zeros(count, np.float32)
    masses = np.bincount(ids, weights=args["sample_weight"].astype(float), minlength=count)
    win, lose = args["pair_winners"], args["pair_losers"]
    distinct = ids[win] != ids[lose]
    l2 = float(np.float32(args["l2_leaf_reg"]))
    def value(candidate):
        raw = np.float32(args["initial_predictions"] + candidate[ids])
        difference = np.float32(raw[win[distinct]] - raw[lose[distinct]])
        return (-np.dot(args["pair_weights"][distinct].astype(float), np.logaddexp(0., -difference.astype(float)))
                - .5 * l2 * np.dot(candidate.astype(float), candidate.astype(float)))
    current, step, fresh, updated = value(point), 1., True, False
    for attempt in range(100):
        if attempt >= args["leaf_estimation_iterations"] and (updated or mode == "No"):
            break
        if fresh:
            result = reference(np.float32(args["initial_predictions"] + point[ids]), win, lose,
                args["pair_weights"], ids, count, "Newton", l2, args["non_diagonal_regularization"])
            matrix = regularize(result["hessian"], False, l2, args["non_diagonal_regularization"])
            gradient = np.float32(result["gradient"].astype(np.float32).astype(float) - l2 * point.astype(float))
            direction = np.zeros(count, np.float32)
            direction[:-1] = np.linalg.solve(matrix[:-1, :-1], gradient[:-1])
            dot = gradient.astype(float) @ direction.astype(float)
        trial = np.float32(point.astype(float) + step * direction.astype(float))
        trial[masses < 1e-20] = 0; trial[-1] = 0
        candidate = value(trial)
        threshold = current + (1e-5 * step * dot if mode == "Armijo" else 0)
        if mode == "No" or (np.isfinite(candidate) and candidate >= threshold):
            point, current, fresh, updated, step = trial, candidate, True, True, 1.
        else:
            step *= .5; fresh = False
    expected = np.float32(point - point.mean(dtype=float)) * np.float32(args["learning_rate"])
    np.testing.assert_allclose(tree.leaf_values, expected, rtol=2e-4, atol=5e-6)


def test_fractional_meta_uses_callback_feature_exponents():
    rng = np.random.default_rng(426)
    rows, features = 67, 9
    bins = rng.integers(0, 2, (features, rows), dtype=np.uint8)
    targets = rng.normal(size=rows).astype(np.float32)
    weights = rng.integers(1, 5, rows).astype(np.float32)
    gradients = np.float32(targets * weights)
    sums = np.array([[gradients[bins[f] == 0].sum(dtype=np.float64) for f in range(features)]], np.float32)
    weight_sums = np.array([[weights[bins[f] == 0].sum(dtype=np.float64) for f in range(features)]], np.float32)
    leaf_sums = np.array([gradients.sum(dtype=np.float64)], np.float32)
    leaf_weights = np.array([weights.sum(dtype=np.float64)], np.float32)
    choices = (np.arange(features) % 3 + 1).astype(np.uint8)
    expected = np.array([min(score_reference(sums[:, f:f + 1], weight_sums[:, f:f + 1],
        leaf_sums, leaf_weights, 0, False, exponent, 2)[0]
        for exponent, flag in ((1, 1), (0, 2)) if choices[f] & flag) for f in range(features)])
    calls = []
    callback_type = ct.CFUNCTYPE(ct.c_int, ct.c_void_p, ct.c_uint32, ct.POINTER(ct.c_uint8))
    @callback_type
    def callback(context, count, output):
        calls.append(count)
        np.ctypeslib.as_array(output, shape=(count,))[:] = choices
        return 0
    with _native.Session(bins, targets, np.arange(features, dtype=np.uint32), np.zeros(features, np.uint32),
        iterations=1, depth=1, learning_rate=1, l2_leaf_reg=2, bias=0, score_function="L2",
        sample_weight=weights) as session:
        configure(session, exponent=0, frequency=.5)
        function = session._lib.cbm_session_set_meta_l2_exponent_callback
        function.argtypes = [ct.c_void_p, callback_type, ct.c_void_p, ct.c_char_p, ct.c_size_t]
        error = ct.create_string_buffer(4096)
        assert function(session._handle, callback, None, error, len(error)) == 0, error.value.decode()
        session.begin_tree()
        split = session.grow_tree()
        session.finish_tree()
    assert calls == [features]
    assert split["feature"] == np.argmin(expected)
    np.testing.assert_allclose(split["score"], expected.min(), rtol=4e-6)


@pytest.mark.parametrize("mode", ["AnyImprovement", "Armijo"])
def test_normalized_scalar_histories_share_one_regularized_step(mode):
    targets = np.array([0, 2, 1, 8, 3, 1, 2], np.float32)
    weights = np.array([.5, 1, 2, 3, 1, .25, 4], np.float32)
    empty = np.array([], np.uint32)
    baselines = np.repeat(np.array([[-3], [0], [3]], np.float32), len(targets), axis=1)
    expected = leaf_walk([(targets, weights, baseline) for baseline in baselines],
        "Poisson", True, True, .4, 4, mode)
    with _native.Session(np.zeros((1, len(targets)), np.uint8), targets, empty, empty,
        iterations=1, depth=0, learning_rate=1, l2_leaf_reg=.4, bias=0, score_function="Cosine",
        objective="Poisson", sample_weight=weights, leaf_estimation_iterations=4,
        leaf_estimation_backtracking=mode) as session:
        session.configure_permutations(np.zeros((3, 1, len(targets)), np.uint8), baselines)
        configure(session, normalize=True, ridge=True)
        session.step()
        state = session.permutation_state
    np.testing.assert_allclose(state["predictions"], baselines + expected[:, None], rtol=8e-6, atol=5e-6)
