"""Variable-tree trimming required by Metal best-model selection; no fitting."""
import json
import os

import numpy as np
import pytest
from catboost import CatBoost, CatBoostError


pytestmark = pytest.mark.skipif(os.environ.get('CATBOOST_NATIVE_METAL_TESTS') != '1',
                              reason='requires the rebuilt variable-tree truncation')


def forest(tmp_path, dimensions):
    def leaf(value, weight):
        values = [value + .03125 * dim for dim in range(dimensions)]
        return {'value': values[0] if dimensions == 1 else values, 'weight': weight}

    def split(left, right):
        return {'split': {'split_type': 'FloatFeature', 'float_feature_index': 0,
                         'border': .5, 'split_index': 0}, 'left': left, 'right': right}

    trees = [leaf(.3, 11), split(leaf(-.7, 2), leaf(.9, 9)),
             split(leaf(.4, 3), split(leaf(-.2, 1), leaf(1.1, 7))),
             split(split(leaf(.8, 2), leaf(-.3, 5)), leaf(-.6, 4)), leaf(.2, 11)]
    spec = {'features_info': {'float_features': [{'feature_index': 0, 'flat_feature_index': 0,
            'feature_id': 'x', 'borders': [.5], 'has_nans': False, 'nan_value_treatment': 'AsIs'}]},
            'scale_and_bias': [1.375, [.125 * (dim + 1) for dim in range(dimensions)]], 'trees': trees}
    path = tmp_path / 'input.json'
    path.write_text(json.dumps(spec))
    return CatBoost().load_model(path, format='json')


@pytest.mark.parametrize('dimensions', [1, 3])
@pytest.mark.parametrize('begin,end', [(0, 1), (0, 3), (0, 5), (1, 4), (2, 3), (4, 5)])
def test_trimming_rebases_nodes_leaves_weights_and_bias(tmp_path, dimensions, begin, end):
    model = forest(tmp_path, dimensions)
    x = np.array([[-2], [.5], [.51], [3]], np.float32)
    expected = model.predict(x, ntree_start=begin, ntree_end=end)
    counts = model.get_tree_leaf_counts()
    offset = np.r_[0, np.cumsum(counts)].astype(int)
    weights = model.get_leaf_weights()[offset[begin]:offset[end]].copy()
    values = model.get_leaf_values()[dimensions * offset[begin]:dimensions * offset[end]].copy()
    model.shrink(ntree_start=begin, ntree_end=end)
    assert model.tree_count_ == end - begin
    np.testing.assert_array_equal(model.get_tree_leaf_counts(), counts[begin:end])
    np.testing.assert_array_equal(model.get_leaf_weights(), weights)
    np.testing.assert_array_equal(model.get_leaf_values(), values)
    for kind in ('cbm', 'json'):
        path = tmp_path / ('shrunk.' + kind)
        model.save_model(path, format=kind)
        restored = CatBoost().load_model(path, format=kind)
        np.testing.assert_allclose(restored.predict(x), expected, atol=1e-14, rtol=1e-14)
        np.testing.assert_allclose(restored.predict(x, task_type='GPU'), expected, atol=1e-14, rtol=1e-14)


@pytest.mark.parametrize('begin,end', [(0, 0), (2, 2), (5, 5)])
def test_empty_tree_ranges_retain_only_applicable_bias(tmp_path, begin, end):
    model = forest(tmp_path, 1)
    model.shrink(ntree_start=begin, ntree_end=end)
    assert model.tree_count_ == 0 and model.get_leaf_values().size == 0
    scale, bias = model.get_scale_and_bias()
    assert scale == 1.375 and bias == (.125 if begin == 0 else 0.)


@pytest.mark.parametrize('begin,end', [(0, 6), (5, 6), (4, 3)])
def test_invalid_tree_range_rejected_before_mutating_model(tmp_path, begin, end):
    model = forest(tmp_path, 1)
    original = model.get_leaf_values().copy()
    with pytest.raises(CatBoostError):
        model.shrink(ntree_start=begin, ntree_end=end)
    assert model.tree_count_ == 5
    np.testing.assert_array_equal(model.get_leaf_values(), original)
