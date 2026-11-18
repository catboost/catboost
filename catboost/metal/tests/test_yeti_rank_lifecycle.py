"""Actual-GPU classic YetiRank public fitting, export, selection and recovery."""
import json
import platform
import numpy as np
import pytest
from catboost_metal import CatBoostMetalRanker
from catboost_metal._training import _shared_metric

pytestmark = pytest.mark.skipif(platform.system() != 'Darwin' or platform.machine() != 'arm64', reason='Apple GPU required')


@pytest.fixture(autouse=True)
def no_cpu_training(monkeypatch):
    from catboost import CatBoost
    def forbidden(*a, **k): raise AssertionError('No CPU CatBoost training')
    monkeypatch.setattr(CatBoost, '_fit', forbidden)


def data():
    rng = np.random.default_rng(817)
    x = rng.normal(size=(103, 4)).astype(np.float32)
    y = np.clip(.5 + .3*x[:, 0] - .2*x[:, 1], 0, 1).astype(np.float32)
    group = np.repeat(np.arange(10), [10]*9 + [13])
    weight = rng.uniform(.3, 2., len(y)).astype(np.float32)
    return x, y, group, weight


def model(**options):
    defaults = dict(loss_function='YetiRank:permutations=7;decay=0.8;mode=Classic', iterations=6,
                    depth=3, leaf_estimation_iterations=3, random_seed=791, random_strength=.7)
    defaults.update(options)
    return CatBoostMetalRanker(**defaults)


@pytest.mark.parametrize('kind', ['No', 'Bayesian', 'Bernoulli', 'Poisson', 'MVS'])
@pytest.mark.parametrize('leaf_iterations', [1, 3])
def test_snapshot_resume_preserves_every_tree_and_metric(tmp_path, kind, leaf_iterations):
    x, y, groups, weights = data()
    config = dict(bootstrap_type=kind, leaf_estimation_iterations=leaf_iterations)
    if kind in ('Bernoulli', 'Poisson', 'MVS'): config['subsample'] = .7
    fit = dict(group_id=groups, sample_weight=weights, eval_set=(x[::-1], y[::-1], groups[::-1]), use_best_model=False)
    full = model(**config).fit(x, y, **fit)
    path = tmp_path / 'yeti.npz'
    partial = model(**config).fit(x, y, **fit, save_snapshot=True, snapshot_file=path,
                                 callback=lambda info: info.iteration < 2)
    assert partial.tree_count_ == 2
    resumed = model(**config).fit(x, y, **fit, save_snapshot=True, snapshot_file=path)
    for name in ('depths', 'split_features', 'split_bins', 'leaf_values', 'leaf_weights', 'predictions', 'rmse'):
        np.testing.assert_array_equal(getattr(full._result, name), getattr(resumed._result, name))
    assert full.evals_result_ == resumed.evals_result_
    assert full.training_stats_['yeti_rng'] == resumed.training_stats_['yeti_rng']
    np.testing.assert_array_equal(full.predict(x, task_type='GPU'), resumed.predict(x, task_type='GPU'))


@pytest.mark.parametrize('metric', ['PFound', 'NDCG:top=5;type=Exp;denominator=LogPosition', 'MAP:top=5;border=0.5'])
def test_real_quality_metrics_best_selection_and_normal_model_export(tmp_path, metric):
    from catboost import CatBoostRanker
    x, y, groups, weights = data()
    trained = model(eval_metric=metric).fit(x, y, group_id=groups, sample_weight=weights,
                                          eval_set=(x, y, groups, weights), use_best_model=True)
    history = trained.evals_result_['validation'][metric]
    assert trained.best_iteration_ == int(np.argmax(history))
    assert trained.tree_count_ == trained.best_iteration_ + 1
    expected = _shared_metric(metric, trained.training_predictions_, y, weights, trained.group_offsets_)
    assert history[trained.best_iteration_] == pytest.approx(expected, abs=2e-7)
    assert trained.evals_result_['learn']['PFound'][0] > .6
    assert trained.best_score_['learn']['PFound'] == max(trained.evals_result_['learn']['PFound'])
    for fmt, suffix in [('cbm', 'cbm'), ('json', 'json')]:
        path = tmp_path / ('model.' + suffix)
        trained.save_model(path, format=fmt)
        loaded = CatBoostRanker().load_model(path, format=fmt)
        np.testing.assert_allclose(loaded.predict(x), trained.predict(x, task_type='GPU'), atol=2e-7)


@pytest.mark.parametrize('change', ['missing', 'word', 'index', 'completed', 'bootstrap'])
def test_corrupt_rng_snapshot_is_rejected_before_gpu_session(tmp_path, monkeypatch, change):
    from catboost_metal import _yeti
    x, y, groups, _ = data(); path = tmp_path / 'yeti.npz'
    model(iterations=2).fit(x, y, group_id=groups, save_snapshot=True, snapshot_file=path)
    with np.load(path, allow_pickle=False) as saved: arrays = {k: saved[k].copy() for k in saved.files}
    header = json.loads(str(arrays['metadata']))
    if change == 'missing': del header['stats']['yeti_rng']
    elif change == 'word': header['stats']['yeti_rng']['words'][0] ^= 1
    elif change == 'index': header['stats']['yeti_rng']['index'] = 314
    elif change == 'completed': header['stats']['yeti_rng']['completed_iterations'] = 3
    else: header['stats']['yeti_rng']['bootstrap_initialized'] = True
    arrays['metadata'] = np.asarray(json.dumps(header))
    np.savez(path, **arrays)
    def forbidden(*a, **k): raise AssertionError('Invalid snapshot reached GPU session')
    monkeypatch.setattr(_yeti.TrainingSession, '__init__', forbidden)
    with pytest.raises(ValueError, match='YetiRank snapshot RNG'):
        model(iterations=3).fit(x, y, group_id=groups, save_snapshot=True, snapshot_file=path)


@pytest.mark.parametrize('options', [dict(loss_function='YetiRank:mode=NDCG'),
    dict(loss_function='YetiRank:permutations=1.5'), dict(loss_function='YetiRank:decay=-1'),
    dict(leaf_estimation_method='Gradient'), dict(leaf_estimation_backtracking='Armijo')])
def test_unsupported_options_are_rejected(options):
    with pytest.raises(ValueError): model(**options)


def test_default_leaf_settings_and_explicit_centering_are_recorded():
    x, y, groups, _ = data()
    trained = CatBoostMetalRanker(loss_function='YetiRank', iterations=2, yeti_legacy_prefix_centering=True)
    assert trained.l2_leaf_reg == 0
    assert trained.leaf_estimation_iterations == 1
    assert trained.eval_metric == 'PFound'
    trained.fit(x, y, group_id=groups)
    assert trained.training_stats_['yeti_centering'] == 'legacy_prefix'
