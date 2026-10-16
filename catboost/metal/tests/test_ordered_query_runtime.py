"""Private Ordered query ABI checked against independent grouped equations.

The oracle evaluates query residual means, weighted softmax and supplied pair
incidence directly in NumPy; it never calls CatBoost fitting or the production
Ordered controller. Every prefix cursor has its own point and feature bank.
"""
import ctypes as ct
import math

import numpy as np
import pytest

from catboost_metal import _ordered
from catboost_metal._native import (
    AppendFeatureOptions, PairOptions, QueryOptions, StepInfo, StructureInfo,
    _f32, _u8, _u32,
)
from test_ordered_training import apple_silicon, prohibit_cpu_training


@pytest.fixture(scope="module")
def query_library():
    # Bind this additive ABI directly: these tests must remain independent of
    # the public Session's objective parsing and snapshot orchestration.
    lib = ct.CDLL(str(_ordered.build_library()))
    u8, u32, f32 = ct.POINTER(ct.c_uint8), ct.POINTER(ct.c_uint32), ct.POINTER(ct.c_float)
    text = [ct.c_char_p, ct.c_size_t]
    prefix = [ct.POINTER(_ordered.Params), ct.c_uint32, ct.c_uint64, u8]
    tail = [u32, ct.c_double, ct.POINTER(ct.c_void_p)]
    signatures = {
        "create_query_banked": prefix + [f32, f32, f32, u32, u32, u8, u32, ct.POINTER(QueryOptions)] + tail,
        "create_pair_banked": prefix + [f32, u32, u32, u8, u32, ct.POINTER(PairOptions), u32, u32, f32] + tail,
        "state_shape": [ct.c_void_p, u32, u32],
        "copy_state": [ct.c_void_p, ct.c_uint32, ct.c_uint32, u32, f32],
        "restore_cursors": [ct.c_void_p, ct.c_uint32, f32],
        "copy_predictions": [ct.c_void_p, f32],
        "begin_tree": [ct.c_void_p, ct.c_uint32],
        "grow_tree": [ct.c_void_p, ct.POINTER(StructureInfo)],
        "finish_tree": [ct.c_void_p, ct.POINTER(StepInfo), u32, u32, u32, u8, f32, f32],
        "step": [ct.c_void_p, ct.c_uint32, ct.POINTER(StepInfo), u32, u32, u32, u8, f32, f32],
        "append_features": [ct.c_void_p, ct.POINTER(AppendFeatureOptions), ct.POINTER(u8),
                            u32, u32, u8, u32, f32, u8, u8, u32],
        "set_feature_activity": [ct.c_void_p, ct.c_uint32, u8],
        "set_backtracking": [ct.c_void_p, ct.c_uint32],
    }
    for name, args in signatures.items():
        operation = getattr(lib, "cbm_ordered_session_" + name)
        operation.argtypes, operation.restype = args + text, ct.c_int
    lib.cbm_ordered_session_close.argtypes = [ct.c_void_p]
    lib.cbm_ordered_session_close.restype = None
    return lib


def query_problem(features=1):
    sizes = np.asarray([3, 4, 5, 3, 6, 4, 5], np.uint32)
    offsets = np.r_[0, np.cumsum(sizes)].astype(np.uint32)
    group_orders = [[0, 1, 2, 3, 4, 5, 6], [2, 5, 1, 6, 3, 0, 4], [4, 1, 6, 0, 5, 2, 3]]
    permutations = np.asarray([
        [row for group in order for row in range(offsets[group], offsets[group + 1])]
        for order in group_orders
    ], np.uint32)
    row = np.arange(offsets[-1], dtype=np.int64)
    banks = np.empty((3, features, len(row)), np.uint8)
    for bank in range(3):
        for feature in range(features):
            banks[bank, feature] = ((row * (bank + 1) + row // (feature + 2) + feature + bank) % 3 != 0)
    targets = np.asarray(.2 + ((row * 7) % 11) / 5., np.float32)
    weights = np.asarray(2. ** ((row % 3) - 1), np.float32)
    weights[8::11] = 0
    initial = np.asarray(.17 * np.sin(row) + .09 * (row % 4), np.float32)
    winners, losers, pair_weights = [], [], []
    for group, (begin, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        winner = int(begin + group % (end - begin))
        for loser in range(begin, end):
            if loser != winner:
                winners.append(winner); losers.append(loser)
                pair_weights.append(.25 * (1 + (loser + group) % 5))
        # A duplicate and a zero-weight reverse edge retain literal edge mass.
        winners.extend([winner, int(begin + (winner - begin + 1) % (end - begin))])
        losers.extend([int(begin + (winner - begin + 1) % (end - begin)), winner])
        pair_weights.extend([.125, 0.])
    return dict(sizes=sizes, offsets=offsets, group_orders=group_orders, permutations=permutations,
                banks=np.ascontiguousarray(banks), targets=targets, weights=weights, initial=initial,
                winners=np.asarray(winners, np.uint32), losers=np.asarray(losers, np.uint32),
                pair_weights=np.asarray(pair_weights, np.float32), beta=1.3, lambda_=.07, growth=1.7)


def query_params(data, objective, method=0, score=0, normalize=False, leaf_iterations=3, depth=1):
    p = _ordered.Params()
    for name, value in dict(rows=len(data["targets"]), features=data["banks"].shape[1],
                            candidates=data["banks"].shape[1], iterations=3, depth=depth,
                            objective=objective, score_function=score, leaf_method=method,
                            leaf_iterations=leaf_iterations, permutations=len(data["permutations"]),
                            min_fold_size=4, normalize=int(normalize), learning_rate=.13,
                            l2=2.5, bias=0., fold_growth=data["growth"], objective_param=0.).items():
        setattr(p, name, value)
    return p


class QueryRuntime:
    def __init__(self, lib, params, data, *, missing_offsets=False, missing_options=False, missing_targets=False):
        self.lib, self.params, self.data = lib, params, data
        self.handle = ct.c_void_p()
        cf = np.arange(params.candidates, dtype=np.uint32)
        cb, types = np.zeros(params.candidates, np.uint32), np.zeros(params.candidates, np.uint8)
        prefix = [ct.byref(params), len(data["banks"]), data["banks"].size, _u8(data["banks"])]
        candidates = [_u32(cf), _u32(cb), _u8(types), _u32(data["permutations"])]
        offsets = None if missing_offsets else _u32(data["offsets"])
        tail = [offsets, data["growth"], ct.byref(self.handle)]
        if params.objective == 14:
            options = PairOptions(len(data["winners"]), len(data["sizes"]), 0, 0)
            args = prefix + [_f32(data["initial"])] + candidates + [
                None if missing_options else ct.byref(options), _u32(data["winners"]),
                _u32(data["losers"]), _f32(data["pair_weights"])] + tail
            name = "create_pair_banked"
        else:
            options = QueryOptions(len(data["sizes"]), data["beta"], data["lambda_"], 0)
            args = prefix + [None if missing_targets else _f32(data["targets"]),
                             _f32(data["weights"]), _f32(data["initial"])] + candidates + [
                                 None if missing_options else ct.byref(options)] + tail
            name = "create_query_banked"
        error = ct.create_string_buffer(4096)
        result = getattr(lib, "cbm_ordered_session_" + name)(*args, error, len(error))
        if result:
            assert not self.handle.value, "failed constructor retained a session"
            raise RuntimeError(error.value.decode())
        assert self.handle.value

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.lib.cbm_ordered_session_close(self.handle)
        self.handle = ct.c_void_p()

    def call(self, name, *args):
        error = ct.create_string_buffer(4096)
        result = getattr(self.lib, "cbm_ordered_session_" + name)(self.handle, *args, error, len(error))
        if result:
            raise RuntimeError(error.value.decode())

    def state(self):
        tasks, count = ct.c_uint32(), ct.c_uint32()
        self.call("state_shape", ct.byref(tasks), ct.byref(count))
        descriptors, cursors = np.empty((tasks.value, 4), np.uint32), np.empty(count.value, np.float32)
        self.call("copy_state", tasks, count, _u32(descriptors), _f32(cursors))
        return descriptors, cursors

    def predictions(self):
        result = np.empty(self.params.rows, np.float32)
        self.call("copy_predictions", _f32(result))
        return result

    def grow(self):
        info = StructureInfo()
        self.call("grow_tree", ct.byref(info))
        return info

    def activity(self, values):
        active = np.asarray(values, np.uint8)
        self.call("set_feature_activity", len(active), _u8(active))

    def finish(self, selected=None):
        p, info, depth = self.params, StepInfo(), ct.c_uint32()
        features, borders = np.zeros(p.depth, np.uint32), np.zeros(p.depth, np.uint32)
        types = np.zeros(p.depth, np.uint8)
        values, weights = np.zeros(1 << p.depth, np.float32), np.zeros(1 << p.depth, np.float32)
        output = [ct.byref(info), ct.byref(depth), _u32(features), _u32(borders), _u8(types), _f32(values), _f32(weights)]
        self.call("finish_tree" if selected is None else "step", *([] if selected is None else [selected]), *output)
        return dict(depth=depth.value, features=features[:depth.value], borders=borders[:depth.value],
                    types=types[:depth.value], values=values[:1 << depth.value],
                    weights=weights[:1 << depth.value], loss=info.loss)

    def append(self, banks):
        banks = np.ascontiguousarray(banks, np.uint8)
        count, features, rows = banks.shape
        assert rows == self.params.rows
        options, first = AppendFeatureOptions(count, features, features, 256), ct.c_uint32()
        pointers = (ct.POINTER(ct.c_uint8) * count)(*[_u8(bank) for bank in banks])
        cf, cb = np.arange(features, dtype=np.uint32), np.zeros(features, np.uint32)
        self.call("append_features", ct.byref(options), pointers, _u32(cf), _u32(cb),
                  None, None, None, None, None, ct.byref(first))
        return first.value


def expected_descriptors(data, p):
    descriptors, offset = [], 0
    for permutation in range(max(p.permutations - 1, 1)):
        ends = np.cumsum(data["sizes"][data["group_orders"][permutation]]).tolist()
        # These deliberately tiny problems have CUDA's minimum prefix = 1.
        prefix = next(end for end in ends if end > 1)
        while True:
            requested = min(int(prefix * data["growth"]), p.rows)
            quality = next((end for end in ends if end > requested), p.rows)
            descriptors.append([prefix, quality, offset, permutation]); offset += quality
            if quality == p.rows:
                break
            prefix = quality
    descriptors.append([p.rows, p.rows, offset, p.permutations - 1])
    return np.asarray(descriptors, np.uint32)


def task_oracle(data, objective, order, point):
    """Return weighted gradient, diagonal curvature, original mass and loss."""
    order, point = np.asarray(order, np.int64), np.asarray(point, np.float64)
    gradient, hessian = np.zeros(len(order)), np.zeros(len(order))
    if objective == 14:
        mass, loss, denominator = np.zeros(len(order)), 0., 0.
        local = {row: index for index, row in enumerate(order)}
        for winner, loser, weight in zip(data["winners"], data["losers"], data["pair_weights"]):
            if int(winner) not in local:
                assert int(loser) not in local
                continue
            win, lose, weight = local[int(winner)], local[int(loser)], float(weight)
            difference = point[win] - point[lose]
            probability = 1. / (1. + np.exp(-difference))
            g, h = weight * (1. - probability), weight * probability * (1. - probability)
            gradient[win] += g; gradient[lose] -= g
            hessian[win] += h; hessian[lose] += h
            mass[win] += weight; mass[lose] += weight
            loss += weight * np.logaddexp(0., -difference); denominator += weight
        return gradient, hessian, mass, loss, denominator
    targets, mass = data["targets"][order].astype(float), data["weights"][order].astype(float)
    group_ids = np.searchsorted(data["offsets"][1:], order, side="right")
    loss, denominator = 0., 0.
    beta, lambda_ = float(np.float32(data["beta"])), float(np.float32(data["lambda_"]))
    for group in np.unique(group_ids):
        positions = np.flatnonzero(group_ids == group)
        assert len(positions) == data["sizes"][group], "oracle received a partial query"
        active = positions[mass[positions] > 0]
        if not len(active):
            continue
        y, w, raw = targets[active], mass[active], point[active]
        if objective == 12:
            residual = y - raw
            residual -= np.dot(w, residual) / w.sum()
            gradient[active], hessian[active] = w * residual, w
            loss += np.dot(w, residual ** 2); denominator += w.sum()
        else:
            logits = beta * raw + np.log(w)
            log_sum = np.logaddexp.reduce(logits)
            probability = np.exp(logits - log_sum)
            target_mass = np.dot(w, y)
            gradient[active] = beta * (w * y - target_mass * probability)
            hessian[active] = beta * target_mass * (beta * probability * (1. - probability) + lambda_)
            loss += np.dot(w * y, log_sum - logits); denominator += target_mass
    return gradient, hessian, mass, loss, denominator


def weak_score(data, p, descriptors, cursors, selected):
    numerator, norm = 0., 1e-20
    for prefix, quality, offset, bank in descriptors[:-1].astype(int):
        if bank != selected:
            continue
        order = data["permutations"][bank, :quality]
        gradient, hessian, weight, _, _ = task_oracle(data, p.objective, order, cursors[offset:offset + quality])
        denominator = hessian if p.score_function else weight
        children = data["banks"][bank, 0, order] > 0
        for side in (False, True):
            estimate = children[:prefix] == side
            quality_side = children[prefix:] == side
            mass = denominator[:prefix][estimate].sum()
            ridge = p.l2 * (mass if p.normalize else 1.)
            value = gradient[:prefix][estimate].sum() / (mass + ridge) if mass > 0 else 0.
            numerator += gradient[prefix:][quality_side].sum() * value
            norm += denominator[prefix:][quality_side].sum() * value ** 2
    assert norm > 1e-15, "test data must yield a valid forced split"
    return -numerator / math.sqrt(norm)


def leaf_oracle(data, p, descriptors, cursors, tree):
    updated, leaves = cursors.copy(), 1 << tree["depth"]
    all_values, all_weights = [], []
    for prefix, quality, offset, bank in descriptors.astype(int):
        order = data["permutations"][bank, :quality]
        ids = np.zeros(quality, np.uint32)
        for depth, (feature, border, kind) in enumerate(zip(tree["features"], tree["borders"], tree["types"])):
            bins = data["banks"][bank, feature, order]
            ids |= np.asarray(bins == border if kind else bins > border, np.uint32) << depth
        base = cursors[offset:offset + quality]
        values = np.zeros(leaves, np.float32)
        for _ in range(p.leaf_iterations):
            # All leaves participate in the query point BEFORE any update.
            point = np.asarray(base + values[ids], np.float32)
            gradient, hessian, mass, _, _ = task_oracle(data, p.objective, order[:prefix], point[:prefix])
            weights = np.bincount(ids[:prefix], weights=mass, minlength=leaves)
            g = np.bincount(ids[:prefix], weights=gradient, minlength=leaves)
            h = np.bincount(ids[:prefix], weights=mass if p.leaf_method else hessian, minlength=leaves)
            diagonal = h + p.l2 * (mass.sum() if p.normalize else 1.)
            next_values = values.astype(float)
            active = weights >= 1e-20
            next_values[~active] = 0.
            next_values[active] += g[active] / (diagonal[active] + 1e-20)
            values = next_values.astype(np.float32)
        if p.objective == 14:
            values = np.asarray(values - np.float32(values.astype(float).mean()), np.float32)
        all_values.append(values); all_weights.append(weights)
        updated[offset:offset + quality] = np.asarray(base + values[ids] * np.float32(p.learning_rate), np.float32)
    predictions = np.empty(p.rows, np.float32)
    _, quality, offset, bank = descriptors[-1].astype(int)
    predictions[data["permutations"][bank]] = updated[offset:offset + quality]
    _, _, _, loss, denominator = task_oracle(data, p.objective, np.arange(p.rows), predictions)
    metric = loss / denominator if denominator else 0.
    if p.objective == 12:
        metric = math.sqrt(metric)
    return updated, predictions, np.asarray(all_values), np.asarray(all_weights), metric


@pytest.mark.parametrize("objective", [12, 13, 14])
@pytest.mark.parametrize("method", [0, 1])
@pytest.mark.parametrize("score", [0, 1])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("selected", [0, 1])
def test_query_prefix_scores_and_joint_leaf_updates(query_library, objective, method, score, normalize, selected):
    data = query_problem()
    p = query_params(data, objective, method, score, normalize)
    expected = expected_descriptors(data, p)
    with QueryRuntime(query_library, p, data) as runtime:
        descriptors, cursors = runtime.state()
        np.testing.assert_array_equal(descriptors, expected)
        for _, quality, offset, bank in descriptors.astype(int):
            np.testing.assert_array_equal(cursors[offset:offset + quality], data["initial"][data["permutations"][bank, :quality]])
        for iteration in range(2):
            runtime.call("begin_tree", selected)
            score_expected = weak_score(data, p, descriptors, cursors, selected)
            split = runtime.grow()
            assert split.has_split and split.depth == 1 and split.feature == 0 and split.bin == 0
            np.testing.assert_allclose(split.score, score_expected, rtol=3e-5, atol=6e-6)
            tree = runtime.finish()
            updated, predictions, values, weights, metric = leaf_oracle(data, p, descriptors, cursors, tree)
            actual_descriptors, actual_cursors = runtime.state()
            np.testing.assert_array_equal(actual_descriptors, expected)
            np.testing.assert_allclose(actual_cursors, updated, rtol=3e-5, atol=6e-6, err_msg=f"all cursors after tree {iteration}")
            np.testing.assert_allclose(runtime.predictions(), predictions, rtol=3e-5, atol=6e-6)
            np.testing.assert_allclose(tree["values"], values[-1] * p.learning_rate, rtol=3e-5, atol=6e-6)
            np.testing.assert_allclose(tree["weights"], weights[-1], rtol=2e-6, atol=2e-6)
            np.testing.assert_allclose(tree["loss"], metric, rtol=3e-5, atol=6e-6)
            cursors = actual_cursors


@pytest.mark.parametrize("objective", [12, 13, 14])
@pytest.mark.parametrize("backtracking", [0, 1, 2])
def test_raw_step_incremental_and_cursor_restore_are_exact(query_library, objective, backtracking):
    data = query_problem()
    p = query_params(data, objective, leaf_iterations=4)
    with QueryRuntime(query_library, p, data) as original:
        original.call("set_backtracking", backtracking)
        first = original.finish(selected=1)
        descriptors, saved = original.state()
        second = original.finish(selected=0)
        final = original.state()[1]; predictions = original.predictions()
    with QueryRuntime(query_library, p, data) as incremental:
        incremental.call("set_backtracking", backtracking)
        incremental.call("begin_tree", 1); incremental.grow()
        actual = incremental.finish()
        for key in first:
            np.testing.assert_array_equal(actual[key], first[key])
        np.testing.assert_array_equal(incremental.state()[1], saved)
    with QueryRuntime(query_library, p, data) as restored:
        restored.call("set_backtracking", backtracking)
        restored.call("restore_cursors", len(saved), _f32(saved))
        np.testing.assert_array_equal(restored.state()[0], descriptors)
        actual = restored.finish(selected=0)
        for key in second:
            np.testing.assert_array_equal(actual[key], second[key])
        np.testing.assert_array_equal(restored.state()[1], final)
        np.testing.assert_array_equal(restored.predictions(), predictions)


@pytest.mark.parametrize("objective", [12, 13, 14])
def test_append_after_first_split_preserves_query_points(query_library, objective):
    data = query_problem(features=2)
    # The full task has three occupied coordinates and one empty leaf. Pair
    # centering must include that empty coordinate after the joint leaf walk.
    row = np.arange(len(data["targets"]))
    data["banks"][-1, 0] = row % 3 != 0
    data["banks"][-1, 1] = row % 3 == 2
    p = query_params(data, objective, depth=2)
    with QueryRuntime(query_library, p, data) as complete:
        descriptors, initial = complete.state()
        complete.activity([1, 0]); complete.call("begin_tree", 0)
        assert complete.grow().has_split
        complete.activity([0, 1]); assert complete.grow().has_split
        expected = complete.finish()
        assert expected["depth"] == 2
        cursors, predictions = complete.state()[1], complete.predictions()
        expected_cursors, expected_predictions, values, weights, metric = leaf_oracle(data, p, descriptors, initial, expected)
        np.testing.assert_allclose(cursors, expected_cursors, rtol=3e-5, atol=6e-6)
        np.testing.assert_allclose(predictions, expected_predictions, rtol=3e-5, atol=6e-6)
        np.testing.assert_allclose(expected["values"], values[-1] * p.learning_rate, rtol=3e-5, atol=6e-6)
        np.testing.assert_allclose(expected["weights"], weights[-1], rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(expected["loss"], metric, rtol=3e-5, atol=6e-6)
        assert expected["weights"][2] == 0
        if objective == 14:
            assert abs(values[-1, 2]) > 1e-4, "empty-leaf centering regression needs a nonzero mean"
            np.testing.assert_allclose(expected["values"].sum(), 0., atol=2e-7)
    reduced = dict(data, banks=np.ascontiguousarray(data["banks"][:, :1]))
    small = query_params(reduced, objective, depth=2)
    with QueryRuntime(query_library, small, reduced) as runtime:
        runtime.call("begin_tree", 0); assert runtime.grow().has_split
        assert runtime.append(data["banks"][:, 1:]) == 1
        runtime.activity([0, 1])
        assert runtime.append(np.empty((3, 0, p.rows), np.uint8)) == 2
        assert runtime.grow().has_split
        actual = runtime.finish()
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])
        np.testing.assert_array_equal(runtime.state()[1], cursors)
        np.testing.assert_array_equal(runtime.predictions(), predictions)


@pytest.mark.parametrize("objective", [12, 13, 14])
@pytest.mark.parametrize("missing", ["offsets", "options"])
def test_query_constructors_require_group_contract(query_library, objective, missing):
    data = query_problem(); p = query_params(data, objective)
    with pytest.raises(RuntimeError, match="(?i)(group|query|pair|options|offset)"):
        with QueryRuntime(query_library, p, data, **{"missing_" + missing: True}):
            pytest.fail("invalid query contract was accepted")


def test_query_softmax_rejects_negative_labels_and_missing_targets(query_library):
    data = query_problem(); p = query_params(data, 13)
    data["targets"][0] = -1
    with pytest.raises(RuntimeError, match="(?i)(target|nonnegative)"):
        with QueryRuntime(query_library, p, data):
            pytest.fail("negative QuerySoftMax label was accepted")
    data["targets"][0] = 1
    with pytest.raises(RuntimeError, match="(?i)(input|target|buffer)"):
        with QueryRuntime(query_library, p, data, missing_targets=True):
            pytest.fail("missing query labels were accepted")


def test_pair_constructor_rejects_cross_query_edge(query_library):
    data = query_problem(); p = query_params(data, 14)
    data["losers"][0] = data["offsets"][2]
    with pytest.raises(RuntimeError, match="(?i)(query|group)"):
        with QueryRuntime(query_library, p, data):
            pytest.fail("cross-query pair was accepted")
