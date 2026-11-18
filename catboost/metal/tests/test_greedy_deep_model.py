"""Deep Lossguide model export without Python recursion or CPU fitting."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from catboost import CatBoost, CatBoostClassifier, CatBoostRegressor

from catboost_metal._greedy_model import dumps_model_json, iter_model_json, model_json, tree_json


MISSING = np.iinfo(np.uint32).max


@pytest.fixture(autouse=True)
def forbid_cpu_fit(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Deep greedy export must not fit a CPU model.")
    for cls in (CatBoost, CatBoostRegressor, CatBoostClassifier):
        monkeypatch.setattr(cls, "fit", forbidden)
    monkeypatch.setattr(CatBoost, "_fit", forbidden)


def chain(depth, *, left=False, unique_borders=False):
    # Keep internal node IDs contiguous to verify that export uses topology,
    # rather than assuming the flat session IDs already have preorder layout.
    nodes = np.zeros((2 * depth + 1, 6), np.uint32)
    for index in range(depth):
        continuation = index + 1 if index + 1 < depth else 2 * depth
        children = [depth + index, continuation]
        if left:
            children.reverse()
        nodes[index] = [0, index if unique_borders else 0, 0, *children, MISSING]
        nodes[depth + index, 5] = index
    nodes[2 * depth, 5] = depth
    return SimpleNamespace(nodes=nodes, leaf_values=np.arange(depth + 1, dtype=np.float64),
                           leaf_weights=np.ones(depth + 1))


def document(depth, *, left=False, unique_borders=False):
    result = SimpleNamespace(trees=(chain(depth, left=left, unique_borders=unique_borders),), stats={})
    borders = [np.arange(depth, dtype=np.float32) if unique_borders else np.array([.5], np.float32)]
    return model_json(result, borders, bias=.25, grow_policy="Lossguide")


def shape(node):
    leaves, maximum, pending = 0, 0, [(node, 0)]
    while pending:
        current, depth = pending.pop()
        maximum = max(maximum, depth)
        if "value" in current:
            leaves += 1
        else:
            pending.extend(((current["left"], depth + 1), (current["right"], depth + 1)))
    return leaves, maximum


@pytest.mark.parametrize("depth", (30, 100))
@pytest.mark.parametrize("left", (False, True))
def test_deep_chains_roundtrip_json_and_cbm_readers(tmp_path, depth, left):
    exported = document(depth, left=left, unique_borders=not left)
    assert shape(exported["trees"][0]) == (depth + 1, depth)
    limit = sys.getrecursionlimit()
    serialized = dumps_model_json(exported)
    assert sys.getrecursionlimit() == limit
    assert json.loads(serialized) == exported
    path = tmp_path / "deep.json"
    path.write_text(serialized)
    model = CatBoost().load_model(path, format="json")
    raw = np.array([[-1.], [.5], [5.5], [depth + 1.]], np.float32)
    expected = np.array([depth, depth, 0, 0]) + .25 if left else np.array([0, 1, 6, depth]) + .25
    np.testing.assert_array_equal(model.predict(raw), expected)
    np.testing.assert_array_equal(model.get_tree_leaf_counts(), [depth + 1])
    assert len(model.get_leaf_values()) == depth + 1
    for format_ in ("json", "cbm"):
        path = tmp_path / ("roundtrip." + format_)
        model.save_model(path, format=format_)
        restored = CatBoost().load_model(path, format=format_)
        np.testing.assert_array_equal(restored.predict(raw), expected)
        np.testing.assert_array_equal(restored.get_tree_leaf_counts(), [depth + 1])


def test_depth_1500_uses_iterative_serialization_and_native_readers(tmp_path):
    depth = 1500
    original_limit = sys.getrecursionlimit()
    exported = document(depth)
    assert shape(exported["trees"][0]) == (depth + 1, depth)
    path = tmp_path / "deep-1500.json"
    path.write_text(dumps_model_json(exported))
    assert sys.getrecursionlimit() == original_limit
    # The native reader recursively reconstructs the tree. Exercise this much
    # deeper case in a separate process so a native stack failure is reported
    # as a failed check rather than taking the entire test runner down.
    code = """
import sys
import numpy as np
from catboost import CatBoost
def forbidden(*args, **kwargs):
    raise AssertionError('Deep model import must not train a CPU model')
CatBoost.fit = CatBoost._fit = forbidden
model = CatBoost().load_model(sys.argv[1], format='json')
expected = np.array([.25, 1500.25])
np.testing.assert_array_equal(model.predict([[0], [1]]), expected)
np.testing.assert_array_equal(model.get_tree_leaf_counts(), [1501])
for format_ in ('cbm', 'json'):
    path = sys.argv[2] + '.' + format_
    model.save_model(path, format=format_)
    restored = CatBoost().load_model(path, format=format_)
    np.testing.assert_array_equal(restored.predict([[0], [1]]), expected)
    np.testing.assert_array_equal(restored.get_tree_leaf_counts(), [1501])
"""
    completed = subprocess.run([sys.executable, "-c", code, str(path), str(tmp_path / "native-roundtrip")],
                               capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_maximum_leaf_bounded_depth_constructs_without_recursion():
    original_limit = sys.getrecursionlimit()
    exported = tree_json(chain(65535), [np.array([.5], np.float32)])
    assert shape(exported) == (65536, 65535)
    assert sys.getrecursionlimit() == original_limit
    with pytest.raises(ValueError, match="at most 65536 leaves"):
        tree_json(chain(65536), [np.array([.5], np.float32)])


def test_catboost_preorder_child_offset_overflow_is_rejected():
    # JSON import stores left-first preorder. This right child's offset is
    # 65536 and would truncate to zero in TNonSymmetricTreeStepNode's uint16.
    with pytest.raises(ValueError, match="preorder child offsets.*uint16"):
        tree_json(chain(32768, left=True), [np.array([.5], np.float32)])


def test_iterative_encoder_matches_json_for_supported_metadata():
    value = {"escaped \"key": [None, True, False, -2, 1.25, "α😀\n\t\\", {}, []],
             "tuple": ("yes", "no"), "nested": {"leaf": {"value": 0.0, "weight": 1.0}}}
    serialized = dumps_model_json(value)
    assert json.loads(serialized) == json.loads(json.dumps(value, allow_nan=False))
    assert "".join(iter_model_json(value)) == serialized
    shared = {"value": 3.}
    assert json.loads(dumps_model_json([shared, shared])) == [shared, shared]


@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
def test_iterative_encoder_rejects_nonfinite_values(value):
    with pytest.raises(ValueError, match="Out of range"):
        dumps_model_json({"value": value})


def test_iterative_encoder_rejects_cycles_and_nonstring_keys():
    value = []
    value.append(value)
    with pytest.raises(ValueError, match="Circular"):
        dumps_model_json(value)
    with pytest.raises(TypeError, match="keys must be strings"):
        dumps_model_json({0: "unsupported key"})
