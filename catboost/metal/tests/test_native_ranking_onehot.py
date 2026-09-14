"""Native categorical ranking against fixed-bin resident Metal forests."""
import json
import os

import numpy as np
import pytest
from catboost import CatBoost, CatBoostError, CatBoostRanker, Pool
from catboost_metal import _pair_matrix, _query_cross_entropy, _yeti, _yeti_pair
from catboost_metal._categorical import cat_feature_hashes

pytestmark = pytest.mark.skipif(os.environ.get('CATBOOST_NATIVE_METAL_QUERY_TESTS') != '1',
                               reason='requires rebuilt native categorical ranking')
CASES = [('YetiRank', 'Newton')] + [(loss, method)
    for loss in ('YetiRankPairwise', 'PairLogitPairwise') for method in ('Simple', 'Newton', 'Gradient')
] + [('QueryCrossEntropy', method) for method in ('Simple', 'Newton')]


@pytest.fixture(autouse=True)
def only_gpu_training(monkeypatch):
    original = CatBoost._fit
    def checked(self, *args, **kwargs):
        assert self.get_params().get('task_type') == 'GPU'
        return original(self, *args, **kwargs)
    monkeypatch.setattr(CatBoost, '_fit', checked)


class StopAfter:
    def after_iteration(self, info): return info.iteration < 2


def problem(loss, method='Newton', kind='No'):
    rng = np.random.default_rng(91541); rows = 144
    categories = np.tile(np.arange(4), rows // 4); rng.shuffle(categories)
    other = rng.integers(0, 3, rows); numeric = rng.integers(0, 4, (rows, 2))
    x = np.empty((rows, 4), object)
    x[:, 0] = numeric[:, 0]; x[:, 3] = numeric[:, 1]
    x[:, 1] = np.array(['foo', 'bar', 'α😀', 'last'])[categories]
    x[:, 2] = np.array(['x', 'y', 'z'])[other]
    y = np.float32(.05 + .25 * categories + .06 * other + .005 * numeric[:, 0])
    weights = rng.uniform(.2, 2, rows).astype(np.float32); weights[::19] = 0
    baseline = rng.normal(0, .1, rows).astype(np.float32)
    groups = np.repeat(np.arange(rows // 12), 12)
    offsets = np.arange(0, rows + 1, 12, dtype=np.uint32)
    edges = np.array([(i, j) for g in range(0, rows, 12) for i in range(g, g + 12)
                      for j in range(g, g + 12) if y[i] > y[j]], np.uint32)
    ew = rng.uniform(.2, 1.6, len(edges)).astype(np.float32)
    pool_options = dict(cat_features=[1, 2], feature_names=['n0', 'kind', 'other', 'n1'],
                        group_id=groups, weight=weights, baseline=baseline)
    if loss == 'PairLogitPairwise': pool_options.update(pairs=edges, pairs_weight=ew)
    options = dict(task_type='GPU', loss_function=loss, iterations=4, depth=3, learning_rate=.17,
        l2_leaf_reg=3., leaf_estimation_method=method, leaf_estimation_iterations=1 if method == 'Simple' else 3,
        leaf_estimation_backtracking='No', bootstrap_type=kind, random_seed=718, random_strength=0.,
        score_function='NewtonL2', one_hot_max_size=6, has_time=True, verbose=False, allow_writing_files=False)
    if loss in ('YetiRank', 'YetiRankPairwise'): options['loss_function'] += ':permutations=7;decay=.85'
    if loss == 'QueryCrossEntropy': options['loss_function'] += ':alpha=.7'
    if kind == 'Bernoulli': options['subsample'] = .7
    return x, y, pool_options, options, offsets


def resident(x, y, pool_options, options, offsets):
    # Independent CatBoost hash reader and fixed numeric borders reconstruct the
    # native equality bank; this never calls a training implementation on CPU.
    bins = [np.array(x[:, f], np.uint8) for f in (0, 3)]
    features = [0] * 3 + [1] * 3; borders = list(range(3)) * 2; types = [0] * 6
    for index, f in enumerate((1, 2), 2):
        hashes = cat_feature_hashes(x[:, f]); values = np.unique(hashes)
        bins.append(np.searchsorted(values, hashes).astype(np.uint8))
        features += [index] * len(values); borders += list(range(len(values))); types += [1] * len(values)
    args = {name: options[name] for name in ('iterations', 'depth', 'learning_rate', 'l2_leaf_reg',
        'leaf_estimation_method', 'leaf_estimation_iterations', 'leaf_estimation_backtracking',
        'bootstrap_type', 'random_seed', 'random_strength', 'score_function')}
    args.update(bins=np.array(bins, np.uint8), candidate_features=np.array(features, np.uint32),
        candidate_bins=np.array(borders, np.uint32), candidate_types=np.array(types, np.uint8),
        group_offsets=offsets, sample_weight=pool_options['weight'], initial_predictions=pool_options['baseline'])
    if options['bootstrap_type'] == 'Bernoulli': args['subsample'] = .7
    loss = options['loss_function'].partition(':')[0]
    if loss == 'PairLogitPairwise':
        edges = pool_options['pairs']
        args.update(pair_winners=edges[:, 0], pair_losers=edges[:, 1], pair_weights=pool_options['pairs_weight'])
        session = _pair_matrix.Session(**args)
    elif loss == 'QueryCrossEntropy': session = _query_cross_entropy.Session(targets=y, alpha=.7, **args)
    else:
        cls = _yeti.TrainingSession if loss == 'YetiRank' else _yeti_pair.TrainingSession
        session = cls(targets=y, permutations=7, decay=.85, **args)
    with session:
        trees = [session.step() for _ in range(options['iterations'])]
        return trees, session.predictions()


@pytest.mark.parametrize('loss,method', CASES)
@pytest.mark.parametrize('kind', ['No', 'Bernoulli'])
@pytest.mark.parametrize('prequantized', [False, True])
def test_native_onehot_ranking_forests_replay_and_standard_readers(tmp_path, loss, method, kind, prequantized):
    x, y, po, options, offsets = problem(loss, method, kind)
    pool = Pool(x, y, **po)
    if prequantized:
        borders = tmp_path / 'borders.tsv'
        borders.write_text(''.join(f'{f}\t{b + .5}\n' for f in (0, 3) for b in range(3)))
        pool.quantize(input_borders=str(borders))
    evaluation_x = x.copy(); evaluation_x[::17, 1] = 'eval-only'; evaluation_x[::23, 2] = 'eval-other'
    evaluation = Pool(evaluation_x, y, **po)
    direct = CatBoostRanker().set_params(**options).fit(pool, eval_set=evaluation, use_best_model=False)
    assert direct.get_metadata()['metal_backend'] == 'METAL'
    assert direct.get_cat_feature_indices() == [1, 2]
    assert direct.feature_names_ == po['feature_names']
    # Eval cursor includes the supplied baseline; ordinary model application does not.
    np.testing.assert_allclose(np.array(direct.get_test_eval()),
        direct.predict(evaluation_x, task_type='GPU') + po['baseline'], rtol=2e-6, atol=3e-7)
    if prequantized:
        trees, prediction = resident(x, y, po, options, offsets)
        np.testing.assert_array_equal(direct.get_tree_leaf_counts(), [1 << t.depth for t in trees])
        np.testing.assert_allclose(direct.get_leaf_values(), np.concatenate([t.leaf_values for t in trees]), rtol=4e-5, atol=5e-7)
        np.testing.assert_allclose(direct.get_leaf_weights(), np.concatenate([t.leaf_weights for t in trees]), rtol=4e-6, atol=4e-6)
        np.testing.assert_allclose(direct.predict(x, task_type='GPU') + po['baseline'], prediction, rtol=4e-5, atol=5e-7)
    saved = options | dict(save_snapshot=True, snapshot_interval=0, snapshot_file='category.snapshot',
                            allow_writing_files=True, train_dir=str(tmp_path))
    assert CatBoostRanker().set_params(**saved).fit(pool, eval_set=evaluation, use_best_model=False,
        callbacks=[StopAfter()]).tree_count_ == 2
    resumed = CatBoostRanker().set_params(**saved).fit(pool, eval_set=evaluation, use_best_model=False)
    for name in ('get_leaf_values', 'get_leaf_weights', 'get_test_eval'):
        np.testing.assert_array_equal(getattr(resumed, name)(), getattr(direct, name)())
    assert resumed.evals_result_ == direct.evals_result_
    predict_x = evaluation_x.copy(); predict_x[::7, 1] = 'prediction-only'
    gpu = direct.predict(predict_x, task_type='GPU')
    np.testing.assert_allclose(gpu, direct.predict(predict_x), rtol=3e-6, atol=3e-7)
    for fmt in ('cbm', 'json'):
        path = tmp_path / ('onehot.' + fmt); direct.save_model(path, format=fmt)
        loaded = CatBoostRanker().load_model(path, format=fmt)
        np.testing.assert_allclose(loaded.predict(predict_x, task_type='GPU'), gpu, rtol=1e-13, atol=1e-16)
        if fmt == 'json':
            document = json.loads(path.read_text())
            assert any(s['split_type'] == 'OneHotFeature' for t in document['oblivious_trees'] for s in t['splits'] or [])
            assert not document['features_info'].get('ctrs')
    changed = x.copy(); changed[0, 1] = x[1, 1] if x[0, 1] != x[1, 1] else 'last-other'
    with pytest.raises(CatBoostError, match='(?i)snapshot.*differ|differ.*snapshot'):
        CatBoostRanker().set_params(**saved).fit(Pool(changed, y, **po), eval_set=evaluation, use_best_model=False)


@pytest.mark.parametrize('loss', ['YetiRank', 'YetiRankPairwise', 'PairLogitPairwise', 'QueryCrossEntropy'])
@pytest.mark.parametrize('prequantized', [False, True])
def test_onall_threshold_gates_multiple_ctr_permutations_but_allows_ignored_features(tmp_path, loss, prequantized):
    x, y, po, options, _ = problem(loss)
    pool = Pool(x, y, **po)
    options.update(has_time=False, permutation_count=4)
    if prequantized: pool.quantize()
    evaluation_x = x.copy(); evaluation_x[0, 1] = 'new'
    evaluation = Pool(evaluation_x, y, **po)
    # Four training values plus an eval-only value require threshold five.
    CatBoostRanker().set_params(**(options | dict(one_hot_max_size=5))).fit(pool, eval_set=evaluation)
    for limit, validation in ((4, evaluation), (2, None)):
        ctr = CatBoostRanker().set_params(**(options | dict(one_hot_max_size=limit))).fit(pool, eval_set=validation)
        assert ctr.get_metadata()['metal_permutations'] == '4'
    model = CatBoostRanker().set_params(**(options | dict(one_hot_max_size=2, ignored_features=[1, 2]))).fit(pool)
    assert np.isfinite(model.predict(x, task_type='GPU')).all()


@pytest.mark.parametrize('method', ['Simple', 'Newton', 'Gradient'])
@pytest.mark.parametrize('prequantized', [False, True])
def test_unlabeled_explicit_pairs_need_no_categorical_target_history(tmp_path, method, prequantized):
    x, y, po, options, _ = problem('PairLogitPairwise', method)
    unlabeled = Pool(x, **po); labeled = Pool(x, y, **po)
    if prequantized:
        unlabeled.quantize(); labeled.quantize()
    a = CatBoostRanker().set_params(**options).fit(unlabeled)
    b = CatBoostRanker().set_params(**options).fit(labeled)
    np.testing.assert_array_equal(a.get_leaf_values(), b.get_leaf_values())
    np.testing.assert_array_equal(a.get_leaf_weights(), b.get_leaf_weights())
    np.testing.assert_array_equal(a.predict(x, task_type='GPU'), b.predict(x, task_type='GPU'))


@pytest.mark.parametrize('loss', ['YetiRank', 'YetiRankPairwise', 'PairLogitPairwise', 'QueryCrossEntropy'])
@pytest.mark.parametrize('constant', [False, True])
def test_categorical_only_and_constant_columns_keep_correct_model_layout(tmp_path, loss, constant):
    x, y, po, options, _ = problem(loss)
    if constant:
        x[:, 2] = 'constant'
        pool = Pool(x, y, **po)
    else:
        x = x[:, [1, 2]]; po.update(cat_features=[0, 1], feature_names=['kind', 'other'])
        pool = Pool(x, y, **po)
    pool.quantize()
    model = CatBoostRanker().set_params(**options).fit(pool)
    assert model.feature_names_ == po['feature_names']
    path = tmp_path / 'only.json'; model.save_model(path, format='json')
    doc = json.loads(path.read_text())
    assert any(s['split_type'] == 'OneHotFeature' for t in doc['oblivious_trees'] for s in t['splits'] or [])
    assert not doc['features_info'].get('ctrs')
    np.testing.assert_allclose(model.predict(x, task_type='GPU'), model.predict(x), atol=2e-7)


@pytest.mark.parametrize('loss', ['YetiRank', 'YetiRankPairwise', 'PairLogitPairwise', 'QueryCrossEntropy'])
def test_onehot_uint8_limit_counts_all_categories(tmp_path, loss):
    n = 512; codes = np.arange(n) % 256
    x = [[f'kind-{c}'] for c in codes]; y = (codes % 7).astype(np.float32) / 7
    group = np.repeat(np.arange(8), 64)
    pool = Pool(x, y, cat_features=[0], group_id=group)
    options = problem(loss)[3] | dict(one_hot_max_size=256, iterations=2, depth=1, has_time=False, permutation_count=4)
    model = CatBoostRanker().set_params(**options).fit(pool)
    path = tmp_path / 'limit.json'; model.save_model(path, format='json')
    assert all(s['split_type'] == 'OneHotFeature' for t in json.loads(path.read_text())['oblivious_trees'] for s in t['splits'])
    np.testing.assert_allclose(model.predict(x, task_type='GPU'), model.predict(x), atol=2e-7)
    evaluation = Pool([['new']] + x[1:], y, cat_features=[0], group_id=group)
    ctr = CatBoostRanker().set_params(**options).fit(pool, eval_set=evaluation)
    assert ctr.get_metadata()['metal_permutations'] == '4'
    with pytest.raises(CatBoostError, match='maximum value of one-hot-encoding is 256'):
        CatBoostRanker().set_params(**(options | dict(one_hot_max_size=257))).fit(pool, eval_set=evaluation)


@pytest.mark.parametrize('loss', ['YetiRank', 'YetiRankPairwise', 'PairLogitPairwise', 'QueryCrossEntropy'])
def test_onehot_initial_model_and_baseline_resume_are_exact(tmp_path, loss):
    x, y, po, options, _ = problem(loss, kind='Bernoulli'); pool = Pool(x, y, **po)
    initial = CatBoostRanker().set_params(**(options | dict(iterations=2))).fit(pool)
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
