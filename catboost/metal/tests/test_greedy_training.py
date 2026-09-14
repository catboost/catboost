"""Greedy scalar training on Metal, checked with algebra and model readers.

CatBoost is used only to import, serialize, and evaluate generated trees.
"""
import ctypes as ct
import json
import platform
from types import SimpleNamespace

import numpy as np
import pytest
from catboost import CatBoost, CatBoostClassifier, CatBoostRegressor

from catboost_metal import _greedy
from catboost_metal._greedy_model import model_json


MISSING = np.iinfo(np.uint32).max
POLICIES = ("Depthwise", "Lossguide", "Region")
SCORES = ("L2", "Cosine", "NewtonL2", "NewtonCosine")


@pytest.fixture(autouse=True)
def forbid_cpu_training(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Greedy Metal validation must not train CPU CatBoost models.")
    for cls in (CatBoost, CatBoostRegressor, CatBoostClassifier):
        monkeypatch.setattr(cls, "fit", forbidden)


@pytest.fixture
def metal():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("Actual Apple Silicon Metal GPU required")


def data(objective="RMSE", rows=259):
    rng = np.random.default_rng(7161)
    bins = rng.integers(0, 8, size=(4, rows), dtype=np.uint8)
    signal = 2.3 * (bins[0] > 2) - 1.7 * (bins[1] > 4) + .8 * (bins[2] > 3) * (bins[0] > 4)
    targets = signal + rng.normal(0, .03, rows)
    if objective == "Logloss":
        targets = rng.random(rows) < 1 / (1 + np.exp(-signal + .5))
    weights = rng.uniform(.2, 2.5, rows).astype(np.float32)
    weights[::13] = 0
    cf = np.repeat(np.arange(4, dtype=np.uint32), 7)
    cb = np.tile(np.arange(7, dtype=np.uint32), 4)
    return bins, targets.astype(np.float32), weights, cf, cb


def route(tree, bins):
    result = np.empty(bins.shape[1], np.int64)
    for row in range(bins.shape[1]):
        index = 0
        for _ in range(18):
            feature, border, kind, left, right, leaf = map(int, tree.nodes[index])
            if leaf != MISSING:
                result[row] = leaf
                break
            predicate = bins[feature, row] == border if kind else bins[feature, row] > border
            index = right if predicate else left
        else:
            raise AssertionError("Tree routing did not terminate within depth 16.")
    return result


def tree_depths(tree):
    depths, stack = {}, [(0, 0)]
    while stack:
        index, depth = stack.pop()
        leaf = int(tree.nodes[index, 5])
        if leaf != MISSING:
            depths[leaf] = depth
        else:
            stack.extend((int(child), depth + 1) for child in tree.nodes[index, 3:5])
    return depths


def derivatives(predictions, targets, weights, objective):
    if objective == "RMSE":
        return weights * (targets - predictions), weights
    probability = np.clip(1 / (1 + np.exp(-predictions)), 1e-7, 1 - 1e-7)
    return weights * (targets - probability), weights * probability * (1 - probability)


def loss(predictions, targets, weights, objective):
    if objective == "RMSE":
        return np.sqrt(np.sum(weights * (targets - predictions)**2) / weights.sum())
    return np.sum(weights * (np.logaddexp(0, predictions) - targets * predictions)) / weights.sum()


def expected_values(tree, bins, targets, weights, predictions, *, objective, method, leaf_iterations, l2, rate):
    ids = route(tree, bins)
    values = np.zeros(len(tree.leaf_values), np.float64)
    for _ in range(leaf_iterations):
        gradient, hessian = derivatives(predictions + values[ids], targets, weights, objective)
        for leaf in range(len(values)):
            rows = ids == leaf
            if weights[rows].sum() < 1e-20:
                continue
            denominator = (hessian if method == "Newton" else weights)[rows].sum()
            values[leaf] += gradient[rows].sum() / (denominator + l2 + 1e-20)
    expected_weights = np.bincount(ids, weights=weights, minlength=len(values))
    return values * rate, expected_weights, ids


def root_winner(bins, targets, weights, cf, cb, *, score, objective, l2, bias):
    gradient, hessian = derivatives(np.full(len(targets), bias), targets, weights, objective)
    denominator = hessian if score.startswith("Newton") else weights

    def score_parts(sums, sums_weights):
        if score.endswith("L2"):
            return -sum(g * g / (w + l2) for g, w in zip(sums, sums_weights) if w > 1e-20)
        directions = [g / (w + l2) if w > 0 else 0 for g, w in zip(sums, sums_weights)]
        return -sum(g * d for g, d in zip(sums, directions)) / np.sqrt(
            1e-10 + sum(w * d * d for w, d in zip(sums_weights, directions)))

    before = score_parts([gradient.sum()], [denominator.sum()])
    scores = []
    for feature, border in zip(cf, cb):
        left = bins[feature] <= border
        sums = [gradient[left].sum(), gradient[~left].sum()]
        ws = [denominator[left].sum(), denominator[~left].sum()]
        scores.append(0 if min(ws) < 1e-20 else score_parts(sums, ws) - before)
    return int(np.argmin(scores))


def test_abi_layout():
    assert ct.sizeof(_greedy.Params) == 80
    assert ct.sizeof(_greedy.Node) == 24
    assert ct.sizeof(_greedy.StepInfo) == 296
    assert _greedy.StepInfo.stats.offset == 24


@pytest.mark.parametrize("objective", ("RMSE", "Logloss"))
@pytest.mark.parametrize("score", SCORES)
def test_root_score_matches_independent_weighted_math(metal, objective, score):
    bins, y, w, cf, cb = data(objective)
    winner = root_winner(bins, y.astype(float), w.astype(float), cf, cb,
                         score=score, objective=objective, l2=2.3, bias=.15)
    result = _greedy.train(bins, y, cf, cb, sample_weight=w, objective=objective,
        score_function=score, grow_policy="Lossguide", iterations=1, depth=1, max_leaves=2,
        l2_leaf_reg=2.3, bias=.15)
    assert result.trees[0].nodes[0, :3].tolist() == [int(cf[winner]), int(cb[winner]), 0]


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("objective", ("RMSE", "Logloss"))
@pytest.mark.parametrize("method", ("Newton", "Gradient"))
@pytest.mark.parametrize("score", SCORES)
def test_weighted_greedy_training_math_and_policy(metal, policy, objective, method, score):
    bins, y, w, cf, cb = data(objective)
    options = dict(grow_policy=policy, objective=objective, score_function=score,
        sample_weight=w, iterations=3, depth=3, max_leaves=7, learning_rate=.35,
        l2_leaf_reg=2.3, bias=.15, leaf_estimation_method=method, leaf_estimation_iterations=3)
    result = _greedy.train(bins, y, cf, cb, **options)
    predictions = np.full(len(y), .15, np.float64)
    np.testing.assert_allclose(result.loss[0], loss(predictions, y, w, objective), rtol=2e-6)
    for index, tree in enumerate(result.trees):
        values, weights, ids = expected_values(tree, bins, y.astype(float), w.astype(float), predictions,
            objective=objective, method=method, leaf_iterations=3, l2=2.3, rate=.35)
        np.testing.assert_allclose(tree.leaf_values, values, rtol=4e-5, atol=1e-5)
        np.testing.assert_allclose(tree.leaf_weights, weights, rtol=3e-6, atol=1e-5)
        predictions += values[ids]
        np.testing.assert_allclose(tree.loss, loss(predictions, y, w, objective), rtol=4e-5, atol=1e-6)
        assert tree.completed_iterations == index + 1
        assert tree.finished == (index == 2)
        depths = tree_depths(tree)
        assert max(depths.values()) <= 3 and len(depths) <= 7
        assert len(tree.nodes) == 2 * len(depths) - 1
        if policy == "Region":
            assert len(depths) <= 4
            # A Region tree grows one chain: at most one internal child per node.
            for node in tree.nodes:
                if node[5] == MISSING:
                    assert sum(tree.nodes[int(child), 5] == MISSING for child in node[3:5]) <= 1
    np.testing.assert_allclose(result.predictions, predictions, rtol=5e-5, atol=1e-5)
    assert result.loss[-1] < result.loss[0]
    assert result.completed_iterations == 3
    assert result.stats["device"] and result.stats["kernel_dispatches"] > 0
    assert np.isfinite(result.stats["gpu_seconds"]) and result.stats["gpu_seconds"] >= 0


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("empty", (True, False))
def test_root_only_tree_and_initial_predictions(metal, policy, empty):
    bins = np.zeros((1, 5), np.uint8)
    y = np.array([1., -1., 3., 4., 2.], np.float32)
    w = np.array([.5, 0, 2, 1, 3], np.float32)
    initial = np.linspace(-.2, .4, 5, dtype=np.float32)
    result = _greedy.train(bins, y, [] if empty else [0], [] if empty else [0],
        sample_weight=w, initial_predictions=initial, grow_policy=policy,
        iterations=2, depth=4 if empty else 0, max_leaves=1, learning_rate=.5, l2_leaf_reg=1)
    predictions = initial.astype(float)
    for tree in result.trees:
        expected = np.sum(w * (y - predictions)) / (w.sum() + 1) * .5
        assert tree.nodes.shape == (1, 6)
        assert tree.nodes[0, 5] == 0
        np.testing.assert_allclose(tree.leaf_values, [expected], rtol=2e-6, atol=1e-6)
        np.testing.assert_allclose(tree.leaf_weights, [w.sum()])
        predictions += expected
    np.testing.assert_allclose(result.predictions, predictions, rtol=2e-6, atol=1e-6)


def test_candidate_tie_equality_and_repeatability(metal):
    bins = np.array([[0, 0, 1, 1], [0, 0, 1, 1]], np.uint8)
    y = np.array([-2, -2, 2, 2], np.float32)
    options = dict(grow_policy="Depthwise", depth=1, iterations=2, l2_leaf_reg=0, learning_rate=.5)
    first = _greedy.train(bins, y, [1, 0], [0, 0], **options)
    again = _greedy.train(bins, y, [1, 0], [0, 0], **options)
    assert first.trees[0].nodes[0, :3].tolist() == [1, 0, 0]
    np.testing.assert_array_equal(first.predictions, [-1.5, -1.5, 1.5, 1.5])
    np.testing.assert_array_equal(first.predictions, again.predictions)
    for left, right in zip(first.trees, again.trees):
        np.testing.assert_array_equal(left.nodes, right.nodes)
        np.testing.assert_array_equal(left.leaf_values, right.leaf_values)


@pytest.mark.parametrize("policy", POLICIES)
def test_min_data_in_leaf_checks_parents_after_initial_root(metal, policy):
    # CUDA permits splitting the initial root even when it is smaller than
    # min_data_in_leaf, then marks the two children terminal by row count.
    bins = np.array([[0, 0, 1, 1]], np.uint8)
    result = _greedy.train(bins, [-2, -2, 2, 2], [0], [0], grow_policy=policy,
        iterations=1, depth=3, min_data_in_leaf=100, l2_leaf_reg=0, learning_rate=1.)
    assert len(result.trees[0].leaf_values) == 2
    np.testing.assert_array_equal(result.predictions, [-2, -2, 2, 2])


def test_one_hot_equality_routing_and_export_rejection(metal):
    bins = np.array([[0, 1, 2, 1, 0, 2]], np.uint8)
    y = np.array([-1, 2, -1, 2, -1, -1], np.float32)
    result = _greedy.train(bins, y, [0], [1], candidate_types=[1],
        iterations=1, depth=1, learning_rate=1., l2_leaf_reg=0.)
    assert result.trees[0].nodes[0, :3].tolist() == [0, 1, 1]
    np.testing.assert_array_equal(result.predictions, y)
    with pytest.raises(ValueError, match="numeric"):
        model_json(result, [[.5, 1.5]])


def test_session_lifecycle_input_copy_and_continuation(metal):
    bins, y, w, cf, cb = data()
    options = dict(depth=3, max_leaves=5, learning_rate=.2, sample_weight=w)
    expected = _greedy.train(bins, y, cf, cb, iterations=5, **options)
    input_bins, input_y = bins.copy(), y.copy()
    with _greedy.TrainingSession(input_bins, input_y, cf, cb, iterations=2, **options) as session:
        initial = session.result()
        assert initial.trees == () and initial.completed_iterations == 0 and len(initial.loss) == 1
        input_bins[:] = 0; input_y[:] = 0
        session.step()
        assert session.completed_iterations == 1
        result = session.step()
        assert result.finished
        with pytest.raises(RuntimeError, match="no remaining"):
            session.step()
        predictions = session.predictions()
        predictions[:] = 99
        assert not np.all(session.predictions() == 99)
        partial = session.result()
    assert session.closed
    session.close()
    for operation in (session.step, session.predictions, session.result, session.__enter__):
        with pytest.raises(RuntimeError, match="closed"):
            operation()
    resumed = _greedy.train(bins, y, cf, cb, iterations=3, initial_predictions=partial.predictions, **options)
    np.testing.assert_allclose(resumed.predictions, expected.predictions, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("objective", ("RMSE", "Logloss"))
def test_standard_json_and_cbm_roundtrip_without_cpu_fit(metal, tmp_path, policy, objective):
    bins, y, w, cf, cb = data(objective)
    result = _greedy.train(bins, y, cf, cb, grow_policy=policy, objective=objective,
        sample_weight=w, iterations=4, depth=4, max_leaves=9, learning_rate=.3, bias=.2)
    model = model_json(result, [np.arange(7, dtype=np.float32) + .5 for _ in range(4)],
                       bias=.2, objective=objective, grow_policy=policy,
                       feature_names=["a", "b", "c", "d"])
    path = tmp_path / "greedy.json"
    path.write_text(json.dumps(model, allow_nan=False))
    reader = CatBoost().load_model(path, format="json")
    assert reader.feature_names_ == ["a", "b", "c", "d"]
    np.testing.assert_allclose(reader.predict(bins.T, prediction_type="RawFormulaVal"),
                               result.predictions, rtol=2e-5, atol=2e-6)
    heldout = np.random.default_rng(17).integers(0, 8, size=(4, 67), dtype=np.uint8)
    expected = .2 + sum(tree.leaf_values[route(tree, heldout)].astype(float) for tree in result.trees)
    for format_ in ("cbm", "json"):
        exported = tmp_path / ("roundtrip." + format_)
        reader.save_model(exported, format=format_)
        restored = CatBoost().load_model(exported, format=format_)
        np.testing.assert_allclose(restored.predict(heldout.T, prediction_type="RawFormulaVal"),
                                   expected, rtol=2e-6, atol=2e-6)
        np.testing.assert_array_equal(restored.get_tree_leaf_counts(),
                                      [len(tree.leaf_values) for tree in result.trees])


def synthetic_result(root_only=False):
    nodes = np.array([[0, 0, 0, 1, 2, MISSING], [0, 0, 0, 0, 0, 2],
                      [1, 0, 0, 3, 4, MISSING], [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1]], np.uint32)
    values, weights = np.array([2., 3., -1.]), np.array([3., 4., 2.])
    if root_only:
        nodes = np.array([[0, 0, 0, 0, 0, 0]], np.uint32)
        values, weights = np.array([.75]), np.array([9.])
    tree = SimpleNamespace(nodes=nodes, leaf_values=values, leaf_weights=weights)
    return SimpleNamespace(trees=(tree,), stats={})


@pytest.mark.parametrize("root_only", (False, True))
def test_model_reader_respects_topology_and_leaf_ids(tmp_path, root_only):
    result = synthetic_result(root_only)
    bins = np.array([[0, 1, 0, 1], [0, 0, 1, 1]], np.uint8)
    path = tmp_path / "tree.json"
    path.write_text(json.dumps(model_json(result, [[.5], [.5]], bias=.25)))
    reader = CatBoost().load_model(path, format="json")
    expected = [.75 + .25] * 4 if root_only else [-.75, 2.25, -.75, 3.25]
    np.testing.assert_array_equal(reader.predict(bins.T), expected)
    binary = tmp_path / "tree.cbm"
    reader.save_model(binary)
    np.testing.assert_array_equal(CatBoost().load_model(binary).predict(bins.T), expected)


@pytest.mark.parametrize("change, message", [
    (lambda tree: tree.nodes.__setitem__((0, 3), 0), "cycle"),
    (lambda tree: tree.nodes.__setitem__((0, 4), 1), "shared node"),
    (lambda tree: tree.nodes.__setitem__((0, 3), 20), "invalid child"),
    (lambda tree: tree.nodes.__setitem__((0, 2), 1), "numeric"),
    (lambda tree: tree.nodes.__setitem__((0, 0), 2), "missing numeric"),
    (lambda tree: tree.nodes.__setitem__((0, 1), 1), "missing numeric"),
    (lambda tree: tree.nodes.__setitem__((4, 5), 0), "exactly once"),
    (lambda tree: tree.leaf_values.__setitem__(0, np.nan), "finite"),
    (lambda tree: tree.leaf_weights.__setitem__(0, -1), "signed weights require Simple"),
])
def test_model_rejects_invalid_graph_and_layout(change, message):
    result = synthetic_result()
    change(result.trees[0])
    with pytest.raises(ValueError, match=message):
        model_json(result, [[.5], [.5]])


@pytest.mark.parametrize("options, message", [
    ({"grow_policy": "SymmetricTree"}, "grow_policy"),
    ({"objective": "QueryRMSE"}, "group_offsets"),
    ({"score_function": "unknown-score"}, "score_function"),
    ({"boosting_type": "Ordered"}, "boosting_type=Plain"),
    ({"bootstrap_type": "MVS"}, "bootstrap_type|MVS"),
    ({"leaf_estimation_backtracking": "invalid-backtracking"}, "leaf_estimation_backtracking"),
    ({"leaf_estimation_method": "Exact"}, "Exact supports"),
    ({"random_strength": -.1}, "random_strength"),
    ({"grow_policy": "Depthwise", "depth": 17}, "depth"),
    ({"max_leaves": 0}, "max_leaves"),
    ({"max_leaves": 65537}, "max_leaves"),
    ({"iterations": True}, "iterations"),
    ({"min_data_in_leaf": 0}, "min_data_in_leaf"),
    ({"sample_weight": [0, 0]}, "positive total"),
    ({"sample_weight": [-1, 2]}, "nonnegative"),
    ({"initial_predictions": [np.inf, 0]}, "finite"),
    ({"learning_rate": 0}, "learning_rate"),
    ({"l2_leaf_reg": -1}, "l2_leaf_reg"),
])
def test_rejects_unsupported_options_before_build(options, message, monkeypatch):
    def forbidden():
        raise AssertionError("Invalid inputs must fail before building Metal.")
    monkeypatch.setattr(_greedy, "build_library", forbidden)
    with pytest.raises(ValueError, match=message):
        _greedy.TrainingSession(np.array([[0, 1]], np.uint8), [0, 1], [0], [0], **options)


def test_rejects_truncating_bins_labels_and_candidate_types(monkeypatch):
    def forbidden():
        raise AssertionError("Invalid inputs must fail before building Metal.")
    monkeypatch.setattr(_greedy, "build_library", forbidden)
    for bins in (np.array([[0, 256]]), np.array([[-1, 1]]), np.array([[0., 1.]])):
        with pytest.raises(ValueError, match="bins"):
            _greedy.train(bins, [0, 1], [0], [0])
    bins = np.array([[0, 1]], np.uint8)
    with pytest.raises(ValueError, match="binary"):
        _greedy.train(bins, [0, .5], [0], [0], objective="Logloss")
    with pytest.raises(ValueError, match="mix numeric"):
        _greedy.train(bins, [0, 1], [0, 0], [0, 1], candidate_types=[0, 1])
    with pytest.raises(ValueError, match="below 255"):
        _greedy.train(bins, [0, 1], [0], [255])
