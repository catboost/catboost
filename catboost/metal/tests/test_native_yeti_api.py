"""Standard CatBoost GPU entry point for classic YetiRank and native recovery."""
import os
import numpy as np
import pytest
from catboost import CatBoost, CatBoostError, CatBoostRanker, Pool
from catboost_metal import _yeti
from catboost_metal._training import _shared_metric
from test_yeti_rank_training import problem
from test_yeti_rank_lifecycle import data

pytestmark = pytest.mark.skipif(os.environ.get('CATBOOST_NATIVE_METAL_TESTS') != '1',
                              reason='requires rebuilt native YetiRank trainer')


@pytest.fixture(autouse=True)
def only_gpu_training(monkeypatch):
    original = CatBoost._fit
    def fit(self, *args, **kwargs):
        assert self.get_params().get('task_type') == 'GPU'
        return original(self, *args, **kwargs)
    monkeypatch.setattr(CatBoost, '_fit', fit)


def params(**options):
    values = dict(task_type='GPU', loss_function='YetiRank:permutations=7;decay=0.85', iterations=6,
                  depth=3, learning_rate=.13, random_seed=817, bootstrap_type='No', random_strength=0,
                  leaf_estimation_iterations=3, has_time=True, verbose=False, allow_writing_files=False,
                  metric_period=1)
    values.update(options)
    return values


@pytest.mark.parametrize('score', ['L2', 'Cosine', 'NewtonL2', 'NewtonCosine', 'SolarL2', 'LOOL2', 'SatL2'])
@pytest.mark.parametrize('leaf_iterations', [1, 3])
def test_native_controller_matches_resident_tree_oracle_with_fixed_quantization(tmp_path, score, leaf_iterations):
    args = problem(iterations=3, depth=3, score_function=score, random_seed=817,
                   learning_rate=.13, leaf_estimation_iterations=leaf_iterations)
    x = args['bins'].T.astype(np.float32)
    groups = np.repeat(np.arange(3), np.diff(args['group_offsets']))
    pool = Pool(x, args['targets'], group_id=groups, weight=args['sample_weight'], baseline=args['initial_predictions'])
    borders = tmp_path / 'borders.tsv'
    borders.write_text(''.join(f'{f}\t{b + .5}\n' for f in range(3) for b in range(3)))
    pool.quantize(input_borders=str(borders))
    model = CatBoostRanker(**params(iterations=3, depth=3, score_function=score,
                                   l2_leaf_reg=.2, leaf_estimation_iterations=leaf_iterations)).fit(pool)
    with _yeti.TrainingSession(**args) as oracle:
        trees = [oracle.step() for _ in range(3)]
        expected_predictions = oracle.predictions()
    np.testing.assert_array_equal(model.get_tree_leaf_counts(), [1 << tree.depth for tree in trees])
    np.testing.assert_allclose(model.get_leaf_values(), np.concatenate([tree.leaf_values for tree in trees]), atol=2e-7, rtol=2e-6)
    np.testing.assert_allclose(model.get_leaf_weights(), np.concatenate([tree.leaf_weights for tree in trees]), atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(model.predict(x, task_type='GPU') + args['initial_predictions'], expected_predictions,
                               atol=2e-7, rtol=2e-6)
    assert model.get_metadata()['metal_yeti_centering'] == 'all_rows'


class StopAfter:
    def after_iteration(self, info): return info.iteration < 2


@pytest.mark.parametrize('kind', ['No', 'Bayesian', 'Bernoulli', 'Poisson', 'MVS'])
@pytest.mark.parametrize('leaf_iterations', [1, 3])
def test_native_snapshot_continues_the_same_stochastic_forest(tmp_path, kind, leaf_iterations):
    x, y, groups, weights = data(); pool = Pool(x, y, group_id=groups, weight=weights)
    options = params(bootstrap_type=kind, leaf_estimation_iterations=leaf_iterations, random_strength=.7)
    if kind in ('Bernoulli', 'Poisson', 'MVS'): options['subsample'] = .7
    fit = dict(eval_set=pool, use_best_model=False)
    direct = CatBoostRanker(**options).fit(pool, **fit)
    saved = dict(options, allow_writing_files=True, train_dir=str(tmp_path), save_snapshot=True,
                 snapshot_file='yeti.snapshot', snapshot_interval=0)
    assert CatBoostRanker(**saved).fit(pool, **fit, callbacks=[StopAfter()]).tree_count_ == 2
    restored = CatBoostRanker(**saved).fit(pool, **fit)
    np.testing.assert_array_equal(restored.get_leaf_values(), direct.get_leaf_values())
    np.testing.assert_array_equal(restored.get_leaf_weights(), direct.get_leaf_weights())
    np.testing.assert_array_equal(restored.predict(x, task_type='GPU'), direct.predict(x, task_type='GPU'))
    assert restored.evals_result_ == direct.evals_result_


@pytest.mark.parametrize('metric', ['PFound', 'NDCG:top=5', 'MAP:top=5;border=0.5'])
def test_native_quality_best_model_weights_and_export(tmp_path, metric):
    x, y, groups, weights = data(); pool = Pool(x, y, group_id=groups, weight=weights)
    model = CatBoostRanker(**params(eval_metric=metric)).fit(pool, eval_set=pool, use_best_model=True)
    key = next(k for k in model.evals_result_['validation'] if k.partition(':')[0] == metric.partition(':')[0])
    history = model.evals_result_['validation'][key]
    assert model.best_iteration_ == int(np.argmax(history))
    assert model.tree_count_ == model.best_iteration_ + 1
    offsets = np.r_[np.flatnonzero(np.r_[True, groups[1:] != groups[:-1]]), len(y)].astype(np.uint32)
    expected = _shared_metric(metric, model.get_test_eval(), y, weights, offsets)
    assert history[model.best_iteration_] == pytest.approx(expected, abs=2e-8)
    for fmt in ['cbm', 'json']:
        path = tmp_path / ('yeti.' + fmt); model.save_model(path, format=fmt)
        loaded = CatBoostRanker().load_model(path, format=fmt)
        np.testing.assert_allclose(loaded.predict(x, task_type='GPU'), model.predict(x), atol=2e-7)


def test_native_object_and_group_weights_are_combined_once():
    x, y, groups, weights = data()
    group_weight = np.linspace(.5, 2., 10).astype(np.float32)[groups]
    separate = Pool(x, y, group_id=groups, group_weight=group_weight); separate.set_weight(weights)
    combined = Pool(x, y, group_id=groups, weight=np.float32(weights * group_weight))
    first = CatBoostRanker(**params()).fit(separate)
    second = CatBoostRanker(**params()).fit(combined)
    np.testing.assert_array_equal(first.get_leaf_values(), second.get_leaf_values())
    np.testing.assert_array_equal(first.get_leaf_weights(), second.get_leaf_weights())


def test_native_initial_model_and_baseline_snapshot(tmp_path):
    x, y, groups, weights = data(); pool = Pool(x, y, group_id=groups, weight=weights)
    initial = CatBoostRanker(**params(iterations=2)).fit(pool)
    pool.set_baseline(np.linspace(-.1, .2, len(y)))
    options = params(iterations=5, random_strength=.5)
    direct = CatBoostRanker(**options).fit(pool, init_model=initial, eval_set=pool, use_best_model=False)
    saved = dict(options, allow_writing_files=True, train_dir=str(tmp_path), save_snapshot=True,
                 snapshot_interval=0, snapshot_file='initial-yeti.snapshot')
    assert CatBoostRanker(**saved).fit(pool, init_model=initial, callbacks=[StopAfter()],
                                      eval_set=pool, use_best_model=False).tree_count_ == 4
    restored = CatBoostRanker(**saved).fit(pool, init_model=initial, eval_set=pool, use_best_model=False)
    assert restored.tree_count_ == 7
    np.testing.assert_array_equal(restored.get_leaf_values(), direct.get_leaf_values())
    np.testing.assert_array_equal(restored.get_test_eval(), direct.get_test_eval())


@pytest.mark.parametrize('options', [dict(loss_function='YetiRank:mode=NDCG'),
    dict(loss_function='YetiRank:permutations=0'), dict(loss_function='YetiRank:decay=-1'),
    dict(leaf_estimation_method='Gradient'), dict(leaf_estimation_backtracking='Armijo'),
    dict(boosting_type='Ordered', score_function='L2')])
def test_native_unsupported_options_rejected(options):
    x, y, groups, _ = data()
    with pytest.raises(CatBoostError):
        CatBoostRanker(**params(**options)).fit(Pool(x, y, group_id=groups))


@pytest.mark.parametrize('depth', [0, 16])
def test_native_snapshot_rng_counts_zero_depth_and_duplicate_search_stops(tmp_path, depth):
    x, y, groups, _ = data()
    # One split candidate forces a repeated-winner stop at depth16.
    x = (x[:, :1] > 0).astype(np.float32)
    pool = Pool(x, y, group_id=groups)
    options = params(iterations=4, depth=depth)
    full = CatBoostRanker(**options).fit(pool)
    assert max(full.get_tree_leaf_counts()) <= 2
    saved = dict(options, allow_writing_files=True, train_dir=str(tmp_path), save_snapshot=True,
                 snapshot_interval=0, snapshot_file='depth-yeti.snapshot')
    CatBoostRanker(**saved).fit(pool, callbacks=[StopAfter()])
    restored = CatBoostRanker(**saved).fit(pool)
    np.testing.assert_array_equal(restored.get_leaf_values(), full.get_leaf_values())


def test_native_snapshot_rejects_corrupted_host_draw_count(tmp_path):
    import struct
    x, y, groups, _ = data(); pool = Pool(x, y, group_id=groups)
    saved = params(allow_writing_files=True, train_dir=str(tmp_path), save_snapshot=True,
                   snapshot_interval=0, snapshot_file='corrupt-yeti.snapshot')
    CatBoostRanker(**saved).fit(pool, callbacks=[StopAfter()])
    path = tmp_path / 'corrupt-yeti.snapshot'
    raw = bytearray(path.read_bytes()); tag = b'Metal YetiRank random v1'
    position = raw.index(tag) + len(tag)
    count = struct.unpack_from('<Q', raw, position)[0]
    struct.pack_into('<Q', raw, position, count + 1)
    path.write_bytes(raw)
    with pytest.raises(CatBoostError, match='YetiRank snapshot random draw count'):
        CatBoostRanker(**saved).fit(pool)
