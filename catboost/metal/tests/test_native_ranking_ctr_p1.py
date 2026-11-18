"""Native P1 CTR banks for generated and full-matrix ranking targets."""
import json
import os

import numpy as np
import pytest
from catboost import CatBoost, CatBoostError, CatBoostRanker, Pool
from catboost.utils import calculate_quantization_grid
from catboost_metal import _pair_matrix, _query_cross_entropy, _yeti, _yeti_pair
from catboost_metal._categorical import cat_feature_hashes
from test_ctrs_group import _reference
from test_native_grouped_ctrs import problem as categorical_problem, TYPES, StopAfter
from test_native_ranking_onehot import CASES

pytestmark = pytest.mark.skipif(os.environ.get('CATBOOST_NATIVE_METAL_QUERY_TESTS') != '1',
                               reason='requires rebuilt native ranking CTR adapter')


@pytest.fixture(autouse=True)
def only_gpu_training(monkeypatch):
    original = CatBoost._fit
    def checked(self, *args, **kwargs):
        assert self.get_params().get('task_type') == 'GPU'
        return original(self, *args, **kwargs)
    monkeypatch.setattr(CatBoost, '_fit', checked)


def problem(loss, method, kind='Borders', bootstrap='No', history='Group'):
    _, x, y, groups, weights, options = categorical_problem(loss, kind, 1, history)
    options.update(score_function='NewtonL2', leaf_estimation_method=method,
        leaf_estimation_iterations=1 if method == 'Simple' else 3, bootstrap_type=bootstrap,
        bayesian_matrix_reg=.2, depth=2, iterations=4)
    if loss == 'YetiRank': options.pop('bayesian_matrix_reg')
    if loss in ('YetiRank', 'YetiRankPairwise'): options['loss_function'] += ':permutations=7;decay=.85'
    if loss == 'QueryCrossEntropy': options['loss_function'] += ':alpha=.7;raw_values_scale=0,0:.8 4,2:1.6'
    if bootstrap == 'Bernoulli': options['subsample'] = .7
    offsets = np.r_[0, np.cumsum(np.bincount(groups))].astype(np.uint32)
    po = dict(cat_features=[0], group_id=groups, weight=weights)
    if loss == 'PairLogitPairwise':
        edges = np.array([(i, j) for a, b in zip(offsets[:-1], offsets[1:])
            for i in range(a, b) for j in range(a, b) if y[i] > y[j]], np.uint32)
        po.update(pairs=edges, pairs_weight=np.linspace(.3, 1.7, len(edges), dtype=np.float32))
    return x, y, groups, po, options, offsets


def oracle(x, y, groups, po, options, offsets, kind, history):
    loss = options['loss_function'].partition(':')[0]
    hashes = cat_feature_hashes(x[:, 0]); order = np.arange(len(x), dtype=np.uint32)
    values, _ = _reference(hashes, y if kind == 'FloatTargetMeanValue' else (y > .5).astype(np.float32),
        order, groups if history == 'Group' else order, kind, 1 if kind == 'Buckets' else 0, .5, 1.)
    grid = np.array(calculate_quantization_grid(values.astype(np.float32), 1, border_type='Uniform'), np.float32)
    bins = np.searchsorted(grid, values.astype(np.float32), side='left').astype(np.uint8)[None, :]
    args = {name: options[name] for name in ('iterations', 'depth', 'learning_rate', 'l2_leaf_reg',
        'leaf_estimation_method', 'leaf_estimation_iterations', 'leaf_estimation_backtracking',
        'bootstrap_type', 'random_seed', 'random_strength', 'score_function')}
    args.update(bins=bins, candidate_features=np.zeros(len(grid), np.uint32), candidate_bins=np.arange(len(grid), dtype=np.uint32),
        group_offsets=offsets, sample_weight=po['weight'])
    if options['bootstrap_type'] == 'Bernoulli': args['subsample'] = .7
    if loss != 'YetiRank': args['non_diagonal_regularization'] = .2
    if loss == 'PairLogitPairwise':
        args.update(pair_winners=po['pairs'][:, 0], pair_losers=po['pairs'][:, 1], pair_weights=po['pairs_weight'])
        session = _pair_matrix.Session(**args)
    elif loss == 'QueryCrossEntropy':
        scales = _query_cross_entropy.select_scales('0,0:.8 4,2:1.6', y, offsets)
        session = _query_cross_entropy.Session(targets=y, alpha=.7, query_scales=scales, **args)
    else:
        cls = _yeti.TrainingSession if loss == 'YetiRank' else _yeti_pair.TrainingSession
        session = cls(targets=y, permutations=7, decay=.85, **args)
    with session:
        trees = [session.step() for _ in range(options['iterations'])]
    return trees, grid


@pytest.mark.parametrize('loss,method', CASES)
@pytest.mark.parametrize('kind', TYPES)
@pytest.mark.parametrize('bootstrap', ['No', 'Bernoulli'])
def test_native_p1_ranking_ctrs_match_independent_history_forests(tmp_path, loss, method, kind, bootstrap):
    x, y, groups, po, options, offsets = problem(loss, method, kind, bootstrap)
    pool = Pool(x, y, **po); pool.quantize()
    direct = CatBoostRanker().set_params(**options).fit(pool, eval_set=pool, use_best_model=False)
    trees, grid = oracle(x, y, groups, po, options, offsets, kind, 'Group')
    np.testing.assert_array_equal(direct.get_tree_leaf_counts(), [1 << t.depth for t in trees])
    np.testing.assert_allclose(direct.get_leaf_values(), np.concatenate([t.leaf_values for t in trees]), rtol=4e-5, atol=6e-7)
    np.testing.assert_allclose(direct.get_leaf_weights(), np.concatenate([t.leaf_weights for t in trees]), rtol=5e-6, atol=5e-6)
    saved = options | dict(save_snapshot=True, snapshot_interval=0, snapshot_file='ctr.snapshot',
        allow_writing_files=True, train_dir=str(tmp_path))
    assert CatBoostRanker().set_params(**saved).fit(pool, eval_set=pool, use_best_model=False,
        callbacks=[StopAfter()]).tree_count_ == 2
    resumed = CatBoostRanker().set_params(**saved).fit(pool, eval_set=pool, use_best_model=False)
    for name in ('get_leaf_values', 'get_leaf_weights', 'get_test_eval'):
        np.testing.assert_array_equal(getattr(resumed, name)(), getattr(direct, name)())
    assert resumed.evals_result_ == direct.evals_result_
    prediction_x = x.copy(); prediction_x[::13, 0] = 'unseen'
    for fmt in ('cbm', 'json'):
        path = tmp_path/('ranking-ctr.' + fmt); direct.save_model(path, format=fmt)
        loaded = CatBoostRanker().load_model(path, format=fmt)
        np.testing.assert_allclose(loaded.predict(prediction_x, task_type='GPU'), direct.predict(prediction_x), rtol=4e-6, atol=3e-7)
        if fmt == 'json':
            doc = json.loads(path.read_text())
            assert any(s['split_type'] == 'OnlineCtr' for t in doc['oblivious_trees'] for s in t['splits'] or [])
            for ctr in doc['features_info']['ctrs']: np.testing.assert_array_equal(np.array(ctr['borders'], np.float32), grid)


@pytest.mark.parametrize('loss,method', CASES)
@pytest.mark.parametrize('history', ['Sample', 'Group'])
def test_explicit_p1_without_has_time_replays_and_checks_p4_availability(tmp_path, loss, method, history):
    x, y, groups, po, options, offsets = problem(loss, method, history=history)
    options.update(has_time=False, permutation_count=1)
    pool = Pool(x, y, **po)
    direct = CatBoostRanker().set_params(**options).fit(pool, eval_set=pool, use_best_model=False)
    assert direct.get_metadata()['metal_permutations'] == '1'
    saved = options | dict(save_snapshot=True, snapshot_interval=0, snapshot_file='shuffled-ctr.snapshot',
        allow_writing_files=True, train_dir=str(tmp_path))
    CatBoostRanker().set_params(**saved).fit(pool, eval_set=pool, use_best_model=False, callbacks=[StopAfter()])
    resumed = CatBoostRanker().set_params(**saved).fit(pool, eval_set=pool, use_best_model=False)
    np.testing.assert_array_equal(resumed.get_leaf_values(), direct.get_leaf_values())
    np.testing.assert_array_equal(resumed.get_test_eval(), direct.get_test_eval())
    assert resumed.evals_result_ == direct.evals_result_
    four = CatBoostRanker().set_params(**(options | dict(permutation_count=4))).fit(pool)
    assert four.get_metadata()['metal_permutations'] == '4'


@pytest.mark.parametrize('loss,method', CASES)
def test_has_time_collapses_requested_multiple_ctr_datasets(loss, method):
    x, y, _, po, options, _ = problem(loss, method)
    pool = Pool(x, y, **po)
    one = CatBoostRanker().set_params(**options).fit(pool)
    four = CatBoostRanker().set_params(**(options | dict(permutation_count=4))).fit(pool)
    assert four.get_metadata()['metal_permutations'] == '1'
    np.testing.assert_array_equal(one.get_leaf_values(), four.get_leaf_values())


@pytest.mark.parametrize('loss,method', CASES)
def test_ctr_initial_models_baselines_and_snapshot_continuation(tmp_path, loss, method):
    x, y, _, po, options, _ = problem(loss, method, 'FloatTargetMeanValue', 'Bernoulli')
    pool = Pool(x, y, **po)
    initial = CatBoostRanker().set_params(**(options | dict(iterations=2))).fit(pool)
    pool.set_baseline(np.linspace(-.1, .2, len(x), dtype=np.float32))
    direct = CatBoostRanker().set_params(**options).fit(pool, init_model=initial, eval_set=pool, use_best_model=False)
    saved = options | dict(save_snapshot=True, snapshot_interval=0, snapshot_file='initial.snapshot',
        allow_writing_files=True, train_dir=str(tmp_path))
    partial = CatBoostRanker().set_params(**saved).fit(pool, init_model=initial, eval_set=pool,
        use_best_model=False, callbacks=[StopAfter()])
    assert partial.tree_count_ == 4
    resumed = CatBoostRanker().set_params(**saved).fit(pool, init_model=initial, eval_set=pool, use_best_model=False)
    assert resumed.tree_count_ == 6
    for name in ('get_leaf_values', 'get_leaf_weights', 'get_test_eval'):
        np.testing.assert_array_equal(getattr(resumed, name)(), getattr(direct, name)())
    assert resumed.evals_result_ == direct.evals_result_
    np.testing.assert_allclose(np.array(direct.get_test_eval()),
        direct.predict(x, task_type='GPU') + pool.get_baseline().ravel(), atol=4e-7, rtol=4e-6)


@pytest.mark.parametrize('loss', ['YetiRank', 'YetiRankPairwise', 'PairLogitPairwise', 'QueryCrossEntropy'])
def test_ranking_ctr_model_size_penalty_changes_feature_selection(tmp_path, loss):
    rng = np.random.default_rng(48231)
    category = np.repeat(np.arange(8), np.arange(10, 90, 10)); rng.shuffle(category)
    y = (category >= 4).astype(np.float32); numeric = y + rng.normal(0, 1, len(y))
    x = [[f'kind-{c}', n] for c, n in zip(category, numeric)]
    groups = np.repeat(np.arange(len(y)//12), 12)
    options = problem(loss, 'Newton')[4] | dict(iterations=1, depth=1,
        simple_ctr=['FeatureFreq:CtrBorderType=Uniform:CtrBorderCount=15:Prior=0.5'])
    selected = []
    for strength in (0, 100):
        model = CatBoostRanker().set_params(**(options | dict(model_size_reg=strength))).fit(
            Pool(x, y, cat_features=[0], group_id=groups))
        path = tmp_path/f'penalty-{strength}.json'; model.save_model(path, format='json')
        selected.append(json.loads(path.read_text())['oblivious_trees'][0]['splits'][0]['split_type'])
    assert selected == ['OnlineCtr', 'FloatFeature']
