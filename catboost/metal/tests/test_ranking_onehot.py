"""Standalone categorical ranking, shared model semantics and strict recovery."""
import json
import platform

import numpy as np
import pytest
from catboost import CatBoost, CatBoostRanker, Pool
from catboost_metal import CatBoostMetalRanker, _native
from catboost_metal._categorical import cat_feature_hashes
from test_native_ranking_onehot import CASES, problem

pytestmark = pytest.mark.skipif(platform.system() != 'Darwin' or platform.machine() != 'arm64',
                               reason='requires Apple GPU')
ALL_CASES = CASES + [(loss, method) for loss in ('QueryRMSE', 'QuerySoftMax', 'PairLogit')
                    for method in ('Newton', 'Gradient')]
LOSSES = ['YetiRank', 'YetiRankPairwise', 'PairLogitPairwise', 'QueryCrossEntropy',
          'QueryRMSE', 'QuerySoftMax', 'PairLogit']


@pytest.fixture(autouse=True)
def no_cpu_training(monkeypatch):
    def forbidden(*args, **kwargs): raise AssertionError('CatBoost fitting is forbidden in standalone tests')
    monkeypatch.setattr(CatBoost, '_fit', forbidden)


def inputs(loss, method='Newton', kind='No'):
    x, y, po, options, _ = problem(loss, method, kind)
    for key in ('task_type', 'has_time', 'verbose', 'allow_writing_files'): options.pop(key)
    options.update(cat_features=[1, 2], border_count=32)
    fit = dict(group_id=po['group_id'], sample_weight=po['weight'], use_best_model=False)
    if loss in ('PairLogit', 'PairLogitPairwise'):
        edge_po = problem('PairLogitPairwise')[2]
        fit.update(pairs=edge_po['pairs'], pairs_weight=edge_po['pairs_weight'],
            eval_pairs=edge_po['pairs'], eval_pairs_weight=edge_po['pairs_weight'])
    evaluation = x.copy(); evaluation[::17, 1] = 'eval-only'
    fit['eval_set'] = (evaluation, y, po['group_id'], po['weight'])
    return x, y, options, fit


@pytest.mark.parametrize('loss,method', ALL_CASES)
@pytest.mark.parametrize('kind', ['No', 'Bernoulli'])
def test_standalone_categorical_forests_evaluate_resume_and_export(tmp_path, loss, method, kind):
    x, y, options, fit = inputs(loss, method, kind)
    model = CatBoostMetalRanker(**options).fit(x, y, **fit)
    assert model._layout.permutation_count == 1 and not model._layout.ctrs
    assert len(model.loss_history_) >= options['iterations']
    np.testing.assert_allclose(model.predict(x, task_type='METAL'), model.training_predictions_, rtol=4e-6, atol=3e-7)
    eval_x = fit['eval_set'][0]
    np.testing.assert_allclose(model.predict(eval_x, task_type='METAL'), model.predict(eval_x), rtol=4e-6, atol=3e-7)
    saved = dict(save_snapshot=True, snapshot_interval=0, snapshot_file=tmp_path/'onehot.snapshot')
    partial = CatBoostMetalRanker(**(options | dict(iterations=2))).fit(x, y, **fit, **saved)
    assert partial.tree_count_ == 2
    resumed = CatBoostMetalRanker(**options).fit(x, y, **fit, **saved)
    for name in ('training_predictions_', 'tree_depths_'):
        np.testing.assert_array_equal(getattr(resumed, name), getattr(model, name))
    assert resumed.evals_result_ == model.evals_result_
    np.testing.assert_array_equal(resumed._result.leaf_values, model._result.leaf_values)
    prediction_x = eval_x.copy(); prediction_x[::7, 1] = 'prediction-only'
    for fmt in ('cbm', 'json'):
        path = tmp_path/('model.' + fmt); model.save_model(path, format=fmt)
        loaded = CatBoostRanker().load_model(path, format=fmt)
        np.testing.assert_allclose(loaded.predict(prediction_x), model.predict(prediction_x, task_type='METAL'), rtol=4e-6, atol=3e-7)
        if fmt == 'json':
            doc = json.loads(path.read_text()); parameters = doc['model_info']['params']
            if isinstance(parameters, str): parameters = json.loads(parameters)
            assert parameters['cat_feature_params']['one_hot_max_size'] == 6
            assert parameters['loss_function']['type'] == loss
            assert any(s['split_type'] == 'OneHotFeature' for t in doc['oblivious_trees'] for s in t['splits'] or [])
            assert not doc['features_info'].get('ctrs')


@pytest.mark.parametrize('loss', LOSSES)
def test_validation_hashes_count_towards_cuda_onehot_threshold_before_training(monkeypatch, loss):
    x, y, options, fit = inputs(loss)
    # Training has four category hashes; the fifth is validation-only.
    CatBoostMetalRanker(**(options | dict(one_hot_max_size=5))).fit(x, y, **fit)
    import catboost_metal.ranker as ranker
    monkeypatch.setattr(ranker, 'run_training', lambda *a, **kw: pytest.fail('Unsupported CTR reached training'))
    with pytest.raises(ValueError, match='learn plus validation categories'):
        CatBoostMetalRanker(**(options | dict(one_hot_max_size=4))).fit(x, y, **fit)
    with pytest.raises(ValueError, match='learn plus validation categories'):
        CatBoostMetalRanker(**(options | dict(one_hot_max_size=2))).fit(x, y, **fit)


@pytest.mark.parametrize('loss', LOSSES)
def test_snapshot_binds_original_dictionary_even_when_dense_bins_are_identical(tmp_path, loss):
    x, y, options, fit = inputs(loss)
    for key in ('eval_set', 'eval_pairs', 'eval_pairs_weight'): fit.pop(key, None)
    saved = dict(save_snapshot=True, snapshot_interval=0, snapshot_file=tmp_path/'dictionary.snapshot')
    original = CatBoostMetalRanker(**(options | dict(iterations=2))).fit(x, y, **fit, **saved)
    old = sorted(set(x[:, 1]), key=lambda value: int(cat_feature_hashes([value])[0]))
    new = sorted([f'renamed-{i}' for i in range(len(old))], key=lambda value: int(cat_feature_hashes([value])[0]))
    mapping = dict(zip(old, new)); changed = x.copy(); changed[:, 1] = [mapping[v] for v in x[:, 1]]
    changed_model = CatBoostMetalRanker(**(options | dict(iterations=2))).fit(changed, y, **fit)
    np.testing.assert_array_equal(changed_model._layout.transform(changed), original._layout.transform(x))
    with pytest.raises(ValueError, match='does not match'):
        CatBoostMetalRanker(**options).fit(changed, y, **fit, **saved)


@pytest.mark.parametrize('loss', LOSSES)
def test_snapshot_binds_eval_only_category_hashes(tmp_path, loss):
    x, y, options, fit = inputs(loss)
    saved = dict(save_snapshot=True, snapshot_interval=0, snapshot_file=tmp_path/'eval.snapshot')
    original = CatBoostMetalRanker(**(options | dict(iterations=2))).fit(x, y, **fit, **saved)
    changed = fit['eval_set'][0].copy(); changed[::17, 1] = 'different-eval-only'
    np.testing.assert_array_equal(original._layout.transform(changed), original._layout.transform(fit['eval_set'][0]))
    with pytest.raises(ValueError, match='does not match'):
        CatBoostMetalRanker(**options).fit(x, y, **(fit | dict(eval_set=(changed, *fit['eval_set'][1:]))), **saved)


@pytest.mark.parametrize('loss', LOSSES)
def test_named_dataframes_and_integer_string_categories_share_native_hashes(loss):
    pd = pytest.importorskip('pandas')
    x, y, options, fit = inputs(loss)
    x[:, 1] = [str(i % 4) if i % 2 else i % 4 for i in range(len(x))]
    options['cat_features'] = ['kind', 'other']; names = ['number', 'kind', 'other', 'second']
    frame = pd.DataFrame(x, columns=names)
    evaluation = frame.copy(); evaluation.iloc[::17, 1] = 'eval-only'
    fit['eval_set'] = (evaluation, *fit['eval_set'][1:])
    a = CatBoostMetalRanker(**options).fit(frame, y, **fit)
    b = CatBoostMetalRanker(**(options | dict(cat_features=[1, 2]))).fit(frame.to_numpy(), y,
        **(fit | dict(eval_set=(evaluation.to_numpy(), *fit['eval_set'][1:]))))
    np.testing.assert_array_equal(a.training_predictions_, b.training_predictions_)
    assert a.feature_names_ == names
    np.testing.assert_allclose(a.predict(evaluation, task_type='METAL'), a.predict(evaluation), atol=3e-7)
    with pytest.raises(ValueError, match='names/order'):
        CatBoostMetalRanker(**options).fit(frame, y, **(fit | dict(eval_set=(evaluation[names[::-1]], *fit['eval_set'][1:]))))


@pytest.mark.parametrize('loss', LOSSES)
def test_standalone_categorical_pool_rejection_points_to_native_adapter(loss):
    x, y, options, fit = inputs(loss)
    pool = Pool(x, y, cat_features=[1, 2], group_id=fit['group_id'])
    with pytest.raises(ValueError, match='numeric Pool'):
        CatBoostMetalRanker(**options).fit(pool)


@pytest.mark.parametrize('loss,method', [('PairLogit', 'Newton'), ('PairLogit', 'Gradient'),
    ('PairLogitPairwise', 'Simple'), ('PairLogitPairwise', 'Newton'), ('PairLogitPairwise', 'Gradient')])
def test_unlabeled_standalone_onehot_pairs_preserve_literal_pair_mass(loss, method):
    x, y, options, fit = inputs(loss, method)
    labeled = CatBoostMetalRanker(**options).fit(x, y, **fit)
    unlabeled = CatBoostMetalRanker(**options).fit(x, None,
        **(fit | dict(eval_set=(fit['eval_set'][0], None, *fit['eval_set'][2:]))))
    np.testing.assert_array_equal(unlabeled._result.leaf_values, labeled._result.leaf_values)
    np.testing.assert_array_equal(unlabeled._result.leaf_weights, labeled._result.leaf_weights)
    np.testing.assert_array_equal(unlabeled.training_predictions_, labeled.training_predictions_)


@pytest.mark.parametrize('loss', LOSSES)
def test_categorical_best_model_matches_shared_metric_and_trimmed_gpu_reader(tmp_path, loss):
    from catboost_metal._training import _shared_metric
    x, y, options, fit = inputs(loss)
    fit['use_best_model'] = True
    model = CatBoostMetalRanker(**(options | dict(iterations=12, eval_metric='PFound'))).fit(x, y, **fit)
    history = model.evals_result_['validation']['PFound']
    assert model.best_iteration_ == int(np.argmax(history))
    assert model.tree_count_ == model.best_iteration_ + 1
    point = model.predict(fit['eval_set'][0], task_type='METAL')
    expected = _shared_metric('PFound', point, y, None if loss == 'PairLogit' else fit['sample_weight'], model.group_offsets_)
    assert history[model.best_iteration_] == pytest.approx(expected, abs=2e-7)
    path = tmp_path/'best.cbm'; model.save_model(path)
    loaded = CatBoostRanker().load_model(path)
    assert loaded.tree_count_ == model.tree_count_
    np.testing.assert_allclose(loaded.predict(fit['eval_set'][0]), point, atol=3e-7)


@pytest.mark.parametrize('kind', ['No', 'Bernoulli'])
def test_middle_hash_is_an_equality_split_not_an_ordinal_threshold(tmp_path, kind):
    spellings = sorted(['zero', 'one', 'two'], key=lambda x: int(cat_feature_hashes([x])[0]))
    code = np.tile(np.arange(3), 40); x = np.array([[spellings[i]] for i in code], object)
    y = (code == 1).astype(np.float32); groups = np.repeat(np.arange(20), 6)
    model = CatBoostMetalRanker(loss_function='QueryRMSE', iterations=1, depth=1,
        cat_features=[0], one_hot_max_size=3, learning_rate=1, l2_leaf_reg=0,
        bootstrap_type=kind, subsample=1 if kind == 'Bernoulli' else None, random_strength=0, score_function='L2').fit(x, y, group_id=groups)
    path = tmp_path/'equality.json'; model.save_model(path, format='json')
    split = json.loads(path.read_text())['oblivious_trees'][0]['splits'][0]
    signed_hash = np.array(cat_feature_hashes([spellings[1]]), np.uint32).view(np.int32)[0]
    assert split['split_type'] == 'OneHotFeature' and split['value'] == int(signed_hash)
    np.testing.assert_allclose(model.training_predictions_, y - np.float32(1 / 3), atol=1e-7)
    np.testing.assert_allclose(model.predict(x, task_type='METAL'), y - np.float32(1 / 3), atol=1e-7)
