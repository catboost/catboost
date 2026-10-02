"""Group-exclusive native CTR histories and existing P4 cursor orchestration."""
import json
import os

import numpy as np
import pytest
from catboost import CatBoost, CatBoostRanker, CatBoostRegressor, CatBoostError, Pool
from catboost.utils import calculate_quantization_grid
from catboost_metal import _native
from catboost_metal._categorical import cat_feature_hashes
from test_ctrs_group import _reference

pytestmark = pytest.mark.skipif(os.environ.get('CATBOOST_NATIVE_METAL_QUERY_TESTS') != '1',
                               reason='requires rebuilt native grouped CTR adapter')
TYPES = ['Borders', 'Buckets', 'FloatTargetMeanValue', 'FeatureFreq']


@pytest.fixture(autouse=True)
def only_gpu(monkeypatch):
    original = CatBoost._fit
    def checked(self, *args, **kwargs):
        assert self.get_params().get('task_type') == 'GPU'
        return original(self, *args, **kwargs)
    monkeypatch.setattr(CatBoost, '_fit', checked)


class StopAfter:
    def after_iteration(self, info): return info.iteration < 2


def problem(loss='QueryRMSE', kind='Borders', permutations=1, history='Group'):
    rng = np.random.default_rng(34721)
    sizes = np.tile([3, 5, 8, 7, 12, 4, 9, 16], 3); rows = sizes.sum()
    groups = np.repeat(np.arange(len(sizes)), sizes)
    category = rng.choice(12, rows, p=np.arange(1, 13)/78)
    x = np.array([[f'category-{c}'] for c in category], object)
    y = (category % 2).astype(np.float32)
    if kind == 'FloatTargetMeanValue': y = np.float32(.125 + .75*y)
    weights = rng.uniform(.2, 2, rows).astype(np.float32); weights[::17] = 0
    options = dict(task_type='GPU', loss_function=loss, iterations=4, depth=2, learning_rate=.17,
        l2_leaf_reg=2., bootstrap_type='No', random_strength=0, score_function='L2', model_size_reg=0,
        boost_from_average=False, leaf_estimation_method='Gradient', leaf_estimation_iterations=2,
        leaf_estimation_backtracking='No', one_hot_max_size=1, max_ctr_complexity=1,
        simple_ctr=[f'{kind}:CtrBorderType=Uniform:CtrBorderCount=1:Prior=0.5'],
        ctr_target_border_count=1, ctr_history_unit=history, permutation_count=permutations,
        has_time=permutations == 1, random_seed=735, verbose=False, allow_writing_files=False)
    cls = CatBoostRegressor if loss == 'RMSE' else CatBoostRanker
    return cls, x, y, groups, weights, options


@pytest.mark.parametrize('loss', ['RMSE', 'QueryRMSE', 'QuerySoftMax'])
@pytest.mark.parametrize('kind', TYPES)
@pytest.mark.parametrize('history', ['Sample', 'Group'])
def test_native_ctr_forests_match_independent_exclusive_query_histories(tmp_path, loss, kind, history):
    cls, x, y, groups, weights, options = problem(loss, kind, history=history)
    pool = Pool(x, y, cat_features=[0], group_id=groups, weight=weights)
    model = cls().set_params(**options).fit(pool)
    hashes = cat_feature_hashes(x[:, 0]); order = np.arange(len(x), dtype=np.uint32)
    reference_groups = groups if history == 'Group' else order
    # Binary Buckets retains bucket one; Borders sees the same one-bit target.
    ctr_targets = y if kind == 'FloatTargetMeanValue' else (y > .5).astype(np.float32)
    values, _ = _reference(hashes, ctr_targets, order, reference_groups, kind,
                           1 if kind == 'Buckets' else 0, .5, 1.)
    values = values.astype(np.float32)
    borders = np.array(calculate_quantization_grid(values, 1, border_type='Uniform'), np.float32)
    bins = np.searchsorted(borders, values, side='left').astype(np.uint8)[None, :]
    native = dict(objective=loss, iterations=4, depth=2, learning_rate=.17, l2_leaf_reg=2., bias=0.,
        sample_weight=weights, leaf_estimation_method='Gradient', leaf_estimation_iterations=2,
        score_function='L2', leaf_estimation_backtracking='No', bootstrap_type='No', random_strength=0.)
    if loss != 'RMSE': native['group_offsets'] = np.r_[0, np.cumsum(np.bincount(groups))].astype(np.uint32)
    with _native.Session(bins, y, np.zeros(len(borders), np.uint32), np.arange(len(borders), dtype=np.uint32), **native) as session:
        trees = [session.step() for _ in range(4)]
    np.testing.assert_array_equal(model.get_tree_leaf_counts(), [1 << t.depth for t in trees])
    np.testing.assert_allclose(model.get_leaf_values(), np.concatenate([t.leaf_values for t in trees]), rtol=4e-5, atol=3e-7)
    np.testing.assert_allclose(model.get_leaf_weights(), np.concatenate([t.leaf_weights for t in trees]), rtol=4e-6, atol=4e-6)
    path = tmp_path/'ctr.json'; model.save_model(path, format='json')
    doc = json.loads(path.read_text())
    assert any(s['split_type'] == 'OnlineCtr' for t in doc['oblivious_trees'] for s in t['splits'] or [])
    for ctr in doc['features_info']['ctrs']:
        np.testing.assert_array_equal(np.array(ctr['borders'], np.float32), borders)


@pytest.mark.parametrize('loss', ['RMSE', 'QueryRMSE', 'QuerySoftMax', 'PairLogit'])
@pytest.mark.parametrize('kind', TYPES)
@pytest.mark.parametrize('permutations', [1, 4])
@pytest.mark.parametrize('prequantized', [False, True])
def test_native_group_ctr_permutations_snapshots_eval_and_standard_readers(tmp_path, loss, kind, permutations, prequantized):
    cls, x, y, groups, weights, options = problem(loss, kind, permutations)
    pool = Pool(x, y, cat_features=[0], group_id=groups, weight=weights)
    if prequantized: pool.quantize()
    heldout_x = x.copy(); heldout_x[::23, 0] = 'unseen'
    evaluation = Pool(heldout_x, y, cat_features=[0], group_id=groups, weight=weights)
    direct = cls().set_params(**options).fit(pool, eval_set=evaluation, use_best_model=False)
    assert direct.get_metadata()['metal_permutations'] == str(permutations)
    assert direct.get_all_params()['ctr_history_unit'] == 'Group'
    np.testing.assert_allclose(direct.get_test_eval(), direct.predict(heldout_x, task_type='GPU'), atol=3e-7, rtol=4e-6)
    saved = options | dict(save_snapshot=True, snapshot_interval=0, snapshot_file='grouped.snapshot',
        allow_writing_files=True, train_dir=str(tmp_path))
    assert cls().set_params(**saved).fit(pool, eval_set=evaluation, use_best_model=False,
        callbacks=[StopAfter()]).tree_count_ == 2
    resumed = cls().set_params(**saved).fit(pool, eval_set=evaluation, use_best_model=False)
    for name in ('get_leaf_values', 'get_leaf_weights', 'get_test_eval'):
        np.testing.assert_array_equal(getattr(resumed, name)(), getattr(direct, name)())
    assert resumed.evals_result_ == direct.evals_result_
    for fmt in ('cbm', 'json'):
        path = tmp_path/('grouped.' + fmt); direct.save_model(path, format=fmt)
        loaded = cls().load_model(path, format=fmt)
        np.testing.assert_allclose(loaded.predict(heldout_x, task_type='GPU'), direct.predict(heldout_x, task_type='GPU'), rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(loaded.predict(heldout_x, task_type='GPU'), direct.predict(heldout_x), atol=3e-7, rtol=4e-6)
    # A valid neighboring query boundary changes histories and snapshot identity.
    changed = groups.copy(); changed[np.count_nonzero(groups == 0)] = 0
    with pytest.raises(CatBoostError, match='(?i)snapshot.*differ|differ.*snapshot'):
        cls().set_params(**saved).fit(Pool(x, y, cat_features=[0], group_id=changed, weight=weights),
            eval_set=evaluation, use_best_model=False)


@pytest.mark.parametrize('loss', ['QueryRMSE', 'QuerySoftMax'])
def test_groupwise_loss_default_selects_group_histories(loss):
    cls, x, y, groups, weights, options = problem(loss)
    explicit = cls().set_params(**options).fit(Pool(x, y, cat_features=[0], group_id=groups, weight=weights))
    options.pop('ctr_history_unit')
    default = cls().set_params(**options).fit(Pool(x, y, cat_features=[0], group_id=groups, weight=weights))
    assert default.get_all_params()['ctr_history_unit'] == 'Group'
    np.testing.assert_array_equal(default.get_leaf_values(), explicit.get_leaf_values())


@pytest.mark.parametrize('kind', TYPES)
@pytest.mark.parametrize('groups', [None, 'singletons'])
def test_trivial_group_history_matches_sample_history(kind, groups):
    cls, x, y, _, weights, options = problem('RMSE', kind)
    ids = None if groups is None else np.arange(len(x))
    pool = Pool(x, y, cat_features=[0], group_id=ids, weight=weights)
    group = cls().set_params(**options).fit(pool)
    sample = cls().set_params(**(options | dict(ctr_history_unit='Sample'))).fit(pool)
    np.testing.assert_array_equal(group.get_leaf_values(), sample.get_leaf_values())
    np.testing.assert_array_equal(group.get_leaf_weights(), sample.get_leaf_weights())
