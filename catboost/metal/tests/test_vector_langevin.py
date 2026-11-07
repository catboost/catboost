"""Independent CUDA vector Langevin walker equations and callback boundaries.

No CPU model is fitted. Deterministic additive callbacks make the source's
full Hessian layout, LAPACK triangle, and repeated gradient noise observable.
"""

import platform

import numpy as np
import pytest

from catboost_metal import _multiclass


pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64",
    reason="Actual Apple Silicon Metal required",
)

OBJECTIVES = ("MultiClass", "MultiClassOneVsAll", "MultiRMSE",
              "RMSEWithUncertainty", "MultiLogloss", "MultiCrossEntropy")
METHODS = ("Gradient", "Newton")
INITIAL_GRADIENT, INITIAL_HESSIAN, TRIAL_GRADIENT, ACCEPTED_GRADIENT = 1, 2, 3, 4
CACHE, SEARCH = 0, 7


@pytest.fixture(autouse=True)
def no_cpu_fit(monkeypatch):
    from catboost import CatBoost

    def forbidden(*args, **kwargs):
        raise AssertionError("Vector Langevin acceptance must not fit CPU models")

    monkeypatch.setattr(CatBoost, "_fit", forbidden)


class Noise:
    def __init__(self, *, zero=False, nonpositive_hessian=False):
        self.zero = zero
        self.nonpositive_hessian = nonpositive_hessian
        self.reset()

    def reset(self):
        self.events, self.seeds, self.timeline = [], [], []

    def draw(self, event, count):
        self.events.append((event, count))
        self.timeline.append(("noise", event, count))
        assert event in (1, 2, 3, 4), "Vector source does not inject weak gradient noise"
        if self.zero:
            return np.zeros(count, np.float64)
        if event == INITIAL_HESSIAN and self.nonpositive_hessian:
            return np.full(count, -1e6, np.float64)
        scale = {1: .071, 2: .013, 3: -.029, 4: .043}[event]
        return scale * (np.arange(count, dtype=np.float64) + 1) * (1 + .17 * (len(self.events) - 1))

    def seed(self, event):
        self.seeds.append(event)
        self.timeline.append(("seed", event))
        assert event in (CACHE, SEARCH), "Vector weak target has no Yeti or scalar Langevin noise"
        return 81733 + len(self.seeds)

    def install(self, session, temperature=2.):
        session.configure_langevin(temperature, self.draw, self.seed)


def arguments(objective="MultiClass", method="Newton", leaf_iterations=1,
              depth=1, candidates=True, **extra):
    rows = 23
    rng = np.random.default_rng(86143)
    dimensions = 2 if objective == "RMSEWithUncertainty" else 4
    bins = np.array([np.arange(rows) >= rows // 2], np.uint8)
    signal = rng.normal(size=(rows, dimensions)).astype(np.float32) * .45
    if objective in ("MultiClass", "MultiClassOneVsAll"):
        targets = np.arange(rows, dtype=np.uint32) % dimensions
    elif objective == "MultiRMSE":
        targets = signal + np.linspace(-.4, .8, dimensions, dtype=np.float32)
    elif objective == "RMSEWithUncertainty":
        targets = signal[:, 0] + .4
    elif objective == "MultiLogloss":
        targets = (signal > .1).astype(np.float32)
    else:
        targets = (.1 + .8 / (1 + np.exp(-signal))).astype(np.float32)
    initial = rng.uniform(-.2, .3, size=(rows, dimensions)).astype(np.float32)
    if objective == "MultiClass":
        initial[:, -1] = 0
    weights = (.4 + np.arange(rows) % 7 / 5).astype(np.float32)
    weights[::11] = 0
    return dict(bins=bins, targets=targets, sample_weight=weights,
        initial_predictions=initial, classes=dimensions, objective=objective,
        candidate_features=np.array([0] if candidates else [], np.uint32),
        candidate_bins=np.array([0] if candidates else [], np.uint32),
        iterations=1, depth=depth, learning_rate=.19, l2_leaf_reg=4.1,
        score_function="Cosine", random_strength=0, random_seed=761,
        bootstrap_type="No", leaf_estimation_method=method,
        leaf_estimation_iterations=leaf_iterations, leaf_estimation_backtracking="No") | extra


def source_statistics(args, point, ids, leaves):
    """Float row derivatives, double reductions, and source full-vector layout."""
    objective, dimensions = args["objective"], args["classes"]
    initial, targets, weights = args["initial_predictions"], args["targets"], args["sample_weight"]
    model_point = point.copy()
    if objective == "MultiClass":
        model_point = np.float32(point - point[:, -1:])
    raw = np.float32(initial + model_point[ids])
    row_hessians = np.zeros((len(ids), dimensions, dimensions), np.float32)
    if objective == "MultiClass":
        exponential = np.exp(raw - raw.max(axis=1, keepdims=True))
        probability = np.float32(exponential / exponential.sum(axis=1, keepdims=True))
        residual = np.eye(dimensions, dtype=np.float32)[targets] - probability
        for row in range(dimensions):
            for column in range(row + 1):
                value = np.float32(weights * probability[:, row] *
                                   (np.float32(row == column) - probability[:, column]))
                row_hessians[:, row, column] = value
                row_hessians[:, column, row] = value
    elif objective == "MultiClassOneVsAll" or objective.startswith(("MultiLogloss", "MultiCrossEntropy")):
        exponential = np.exp(-np.abs(raw))
        probability = np.float32(np.where(raw >= 0, 1 / (1 + exponential), exponential / (1 + exponential)))
        if objective == "MultiClassOneVsAll":
            probability = np.clip(probability, np.float32(1e-7), np.float32(1) - np.float32(1e-7))
            residual = np.eye(dimensions, dtype=np.float32)[targets] - probability
        else:
            residual = targets - probability
        diagonal = np.float32(weights[:, None] * probability * (1 - probability))
        row_hessians[:, np.arange(dimensions), np.arange(dimensions)] = diagonal
    elif objective == "MultiRMSE":
        residual = np.float32(targets - raw)
        row_hessians[:, np.arange(dimensions), np.arange(dimensions)] = weights[:, None]
    else:
        error = np.float32(targets - raw[:, 0])
        inverse_variance = np.exp(np.minimum(np.float32(-2) * raw[:, 1], np.float32(70)))
        normalized = np.float32(error * error * inverse_variance)
        residual = np.column_stack((error, np.float32(normalized - 1)))
        row_hessians[:, 0, 0] = weights
        row_hessians[:, 1, 1] = np.float32(2 * weights * normalized)
    row_gradient = np.float32(residual * weights[:, None])
    gradients = np.zeros((leaves, dimensions), np.float64)
    matrices = np.zeros((leaves, dimensions, dimensions), np.float64)
    masses = np.bincount(ids, weights=weights, minlength=leaves)
    for leaf in range(leaves):
        selected = ids == leaf
        gradients[leaf] = row_gradient[selected].sum(axis=0, dtype=np.float64)
        matrices[leaf] = row_hessians[selected].sum(axis=0, dtype=np.float64)
        if objective == "MultiClass":
            # CUDA reconstructs the last derivative from already reduced C-1
            # statistics before adding independent noise to all C coordinates.
            gradients[leaf, -1] = -gradients[leaf, :-1].sum(dtype=np.float64)
    l2 = float(np.float32(args["l2_leaf_reg"]))
    if args["leaf_estimation_method"] == "Gradient":
        hessian = np.broadcast_to(masses[:, None] + l2, gradients.shape).copy()
    elif objective == "MultiClassOneVsAll":
        hessian = np.diagonal(matrices, axis1=1, axis2=2).copy() + l2
    else:
        hessian = matrices + l2 * np.eye(dimensions)[None]
    return gradients, hessian, masses


def solve_source(gradient, hessian, *, upper=False):
    if hessian.ndim == 2:
        direction = np.zeros_like(gradient, dtype=np.float32)
        positive = hessian > 0
        direction[positive] = np.float32(gradient[positive] / (hessian[positive] + np.float32(1e-20)))
        return direction
    direction = np.empty_like(gradient, dtype=np.float32)
    for leaf in range(len(gradient)):
        # C++ constructs row-major matrices but dposv('U') treats memory as
        # column-major. Only the row-major LOWER noisy triangle is consumed.
        half = np.triu(hessian[leaf]) if upper else np.tril(hessian[leaf])
        symmetric = half + half.T - np.diag(np.diag(half))
        try:
            np.linalg.cholesky(symmetric)
        except np.linalg.LinAlgError:
            # linear_system.cpp checks info>=0, not info==0. dposv leaves
            # the right-hand side unchanged when factorization fails.
            direction[leaf] = np.float32(gradient[leaf])
        else:
            direction[leaf] = np.float32(np.linalg.solve(symmetric, gradient[leaf]))
    return direction


def reference(args, noise, ids, leaves, *, upper=False):
    point = np.zeros((leaves, args["classes"]), np.float32)
    gradient, hessian, masses = source_statistics(args, point, ids, leaves)
    gradient += noise.draw(INITIAL_GRADIENT, gradient.size).reshape(gradient.shape)
    hessian += noise.draw(INITIAL_HESSIAN, hessian.size).reshape(hessian.shape)
    iterations = args["leaf_estimation_iterations"]
    for _ in range(iterations):
        direction = solve_source(gradient, hessian, upper=upper)
        point = np.float32(point.astype(np.float64) + direction.astype(np.float64))
        point[masses < 1e-20] = 0
        if iterations == 1:
            break
        gradient, hessian, _ = source_statistics(args, point, ids, leaves)
        gradient += noise.draw(TRIAL_GRADIENT, gradient.size).reshape(gradient.shape)
        # Backtracking No accepts every finite trial. Its second callback
        # modifies the already-noisy gradient, never the refreshed Hessian.
        gradient += noise.draw(ACCEPTED_GRADIENT, gradient.size).reshape(gradient.shape)
    if args["objective"] == "MultiClass":
        point = np.float32(point - point[:, -1:])
    return np.float32(point * np.float32(args["learning_rate"])), masses


def route(tree, bins):
    ids = np.zeros(bins.shape[1], np.uint32)
    for bit, (feature, border, flag) in enumerate(zip(tree.split_features, tree.split_bins, tree.split_types)):
        right = bins[feature] == border if flag else bins[feature] > border
        ids |= right.astype(np.uint32) << bit
    return ids


@pytest.mark.parametrize("objective", OBJECTIVES)
@pytest.mark.parametrize("method", METHODS)
def test_initial_vector_noise_uses_full_source_hessian_and_class_gauge(objective, method):
    args = arguments(objective, method)
    noise = Noise()
    with _multiclass.Session(**args) as session:
        noise.install(session)
        tree = session.step()
        prediction = session.predictions()
    ids = route(tree, args["bins"])
    assert tree.depth == 1 and np.unique(ids).size == 2
    oracle_noise = Noise()
    expected, weights = reference(args, oracle_noise, ids, 2)
    np.testing.assert_allclose(tree.leaf_values, expected, rtol=8e-5, atol=5e-6)
    np.testing.assert_allclose(tree.leaf_weights, weights, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(prediction, args["initial_predictions"] + expected[ids], rtol=8e-5, atol=6e-6)
    dimensions = args["classes"]
    hessian_size = 2 * dimensions * (dimensions if method == "Newton" and objective != "MultiClassOneVsAll" else 1)
    assert noise.events == oracle_noise.events == [(1, 2 * dimensions), (2, hessian_size)]
    assert noise.seeds == [SEARCH]
    if objective == "MultiClass":
        np.testing.assert_array_equal(tree.leaf_values[:, -1], 0)
    if method == "Newton" and objective != "MultiClassOneVsAll":
        wrong_triangle, _ = reference(args, Noise(), ids, 2, upper=True)
        assert np.max(np.abs(expected - wrong_triangle)) > 1e-5


@pytest.mark.parametrize("objective", OBJECTIVES)
@pytest.mark.parametrize("method", METHODS)
def test_two_iteration_no_backtracking_noises_each_accepted_gradient_twice(objective, method):
    args = arguments(objective, method, leaf_iterations=2, depth=0, candidates=False)
    noise = Noise()
    with _multiclass.Session(**args) as session:
        noise.install(session)
        tree = session.step()
    oracle_noise = Noise()
    expected, _ = reference(args, oracle_noise, np.zeros(len(args["targets"]), np.uint32), 1)
    np.testing.assert_allclose(tree.leaf_values, expected, rtol=1e-4, atol=7e-6)
    assert noise.events == oracle_noise.events
    assert [event for event, _ in noise.events] == [1, 2, 3, 4, 3, 4]
    assert noise.seeds == []


@pytest.mark.parametrize("objective", tuple(loss for loss in OBJECTIVES if loss != "MultiClassOneVsAll"))
def test_nonpositive_definite_full_newton_matrix_preserves_noisy_rhs(objective):
    args = arguments(objective, depth=0, candidates=False)
    noise = Noise(nonpositive_hessian=True)
    with _multiclass.Session(**args) as session:
        noise.install(session)
        tree = session.step()
    expected, _ = reference(args, Noise(nonpositive_hessian=True), np.zeros(len(args["targets"]), np.uint32), 1)
    np.testing.assert_allclose(tree.leaf_values, expected, rtol=6e-5, atol=6e-6)
    assert np.max(np.abs(expected)) > .01


@pytest.mark.parametrize("objective,method", [("MultiClass", "Gradient"), ("MultiClassOneVsAll", "Newton")])
def test_nonpositive_noisy_diagonal_produces_zero_direction(objective, method):
    args = arguments(objective, method, depth=0, candidates=False)
    noise = Noise(nonpositive_hessian=True)
    with _multiclass.Session(**args) as session:
        noise.install(session)
        tree = session.step()
    np.testing.assert_array_equal(tree.leaf_values, 0)
    assert noise.events == [(1, args["classes"]), (2, args["classes"])]


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_zero_temperature_keeps_all_source_leaf_callbacks(objective):
    args = arguments(objective, leaf_iterations=2, depth=0, candidates=False)
    noise = Noise(zero=True)
    with _multiclass.Session(**args) as session:
        noise.install(session, temperature=0)
        actual = session.step()
    with _multiclass.Session(**args) as session:
        expected = session.step()
    np.testing.assert_allclose(actual.leaf_values, expected.leaf_values, rtol=6e-5, atol=6e-6)
    assert [event for event, _ in noise.events] == [1, 2, 3, 4, 3, 4]


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_simple_has_source_search_events_and_no_leaf_noise(objective):
    args = arguments(objective, "Simple", bootstrap_type="Bernoulli", subsample=.8)
    noise = Noise()
    with _multiclass.Session(**args) as session:
        noise.install(session)
        actual = session.step()
    with _multiclass.Session(**args) as session:
        expected = session.step()
    np.testing.assert_array_equal(actual.leaf_values, expected.leaf_values)
    np.testing.assert_array_equal(actual.leaf_weights, expected.leaf_weights)
    assert noise.events == []
    assert noise.seeds == [CACHE, SEARCH]


@pytest.mark.parametrize("depth", (0, 1))
@pytest.mark.parametrize("sampling", ("No", "Bernoulli"))
def test_vector_cache_and_search_events_are_separate_from_leaf_noise(depth, sampling):
    args = arguments("MultiRMSE", depth=depth, iterations=2,
                     bootstrap_type=sampling, subsample=.8)
    noise = Noise(zero=True)
    with _multiclass.Session(**args) as session:
        noise.install(session)
        for _ in range(args["iterations"]):
            session.step()
    per_tree = ([CACHE] if sampling != "No" else []) + [SEARCH]
    assert noise.seeds == per_tree * args["iterations"]
    dimensions, leaves = args["classes"], 1 << depth
    expected = [("seed", event) for event in per_tree]
    expected += [("noise", 1, leaves * dimensions), ("noise", 2, leaves * dimensions * dimensions)]
    assert noise.timeline == expected * args["iterations"]


@pytest.mark.parametrize("failure", ("exception", "shape", "nonfinite"))
@pytest.mark.parametrize("failed_event", (INITIAL_HESSIAN, TRIAL_GRADIENT))
def test_callback_failure_rolls_back_cursors_and_allows_retry(failure, failed_event):
    args = arguments("MultiClass", leaf_iterations=2, depth=0, candidates=False)
    noise = Noise()
    fail_once = [True]

    def callback(event, count):
        if event == failed_event and fail_once[0]:
            fail_once[0] = False
            if failure == "exception":
                raise ValueError("injected vector noise failure")
            return np.zeros(count + 1) if failure == "shape" else np.full(count, np.nan)
        return noise.draw(event, count)

    with _multiclass.Session(**args) as session:
        session.configure_langevin(2, callback, noise.seed)
        prediction = session.predictions()
        optimization = session.optimization_predictions()
        bootstrap = session.bootstrap_state
        with pytest.raises(RuntimeError, match="(?i)(Langevin|callback)"):
            session.step()
        assert session.completed_iterations == 0
        np.testing.assert_array_equal(session.predictions(), prediction)
        np.testing.assert_array_equal(session.optimization_predictions(), optimization)
        assert session.bootstrap_state == bootstrap
        noise.reset()
        actual = session.step()
    expected, _ = reference(args, Noise(), np.zeros(len(args["targets"]), np.uint32), 1)
    np.testing.assert_allclose(actual.leaf_values, expected, rtol=1e-4, atol=7e-6)


def test_default_seed_callback_and_strong_noise_callback_lifetime():
    args = arguments("MultiRMSE", depth=0)
    events = []

    class Callback:
        def __call__(self, event, count):
            events.append((event, count))
            return np.zeros(count, np.float64)

    with _multiclass.Session(**args) as session:
        session.configure_langevin(0, Callback())
        import gc
        gc.collect()
        tree = session.step()
    assert np.isfinite(tree.leaf_values).all()
    assert events == [(1, args["classes"]), (2, args["classes"] ** 2)]
