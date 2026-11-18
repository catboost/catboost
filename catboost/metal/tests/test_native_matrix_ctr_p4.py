"""Native full-matrix categorical P4, standard models and exact snapshots."""
import json
import os
import numpy as np
import pytest
from catboost import CatBoost, CatBoostRanker, CatBoostError, Pool
from test_native_ranking_ctr_p1 import problem
from test_native_grouped_ctrs import TYPES, StopAfter

pytestmark = pytest.mark.skipif(os.environ.get('CATBOOST_NATIVE_METAL_QUERY_TESTS') != '1',
                               reason='requires rebuilt native full-matrix P4 adapter')
CASES = [('PairLogitPairwise', method, mode) for method in ('Simple', 'Newton', 'Gradient')
         for mode in (['No'] if method == 'Simple' else ['No', 'AnyImprovement', 'Armijo'])] + [
         ('QueryCrossEntropy', method, mode) for method in ('Simple', 'Newton')
         for mode in (['No'] if method == 'Simple' else ['No', 'AnyImprovement', 'Armijo'])]


@pytest.fixture(autouse=True)
def only_gpu(monkeypatch):
    original = CatBoost._fit
    def checked(self, *args, **kwargs):
        assert self.get_params().get('task_type') == 'GPU'
        return original(self, *args, **kwargs)
    monkeypatch.setattr(CatBoost, '_fit', checked)


@pytest.mark.parametrize('loss,method,mode', CASES)
@pytest.mark.parametrize('kind', TYPES)
@pytest.mark.parametrize('prequantized', [False, True])
def test_native_matrix_p4_ctrs_resume_metrics_and_model_readers(tmp_path, loss, method, mode, kind, prequantized):
    x, y, _, po, options, _ = problem(loss, method, kind, 'Bernoulli' if prequantized else 'No')
    options.update(has_time=False, permutation_count=4, leaf_estimation_backtracking=mode)
    pool = Pool(x, y, **po)
    if prequantized: pool.quantize()
    evaluation_x = x.copy(); evaluation_x[::17, 0] = 'unseen'
    evaluation = Pool(evaluation_x, y, **po)
    direct = CatBoostRanker().set_params(**options).fit(pool, eval_set=evaluation, use_best_model=False)
    assert direct.get_metadata()['metal_permutations'] == '4'
    np.testing.assert_allclose(direct.get_test_eval(), direct.predict(evaluation_x, task_type='GPU'), atol=4e-7, rtol=4e-6)
    saved = options | dict(save_snapshot=True, snapshot_interval=0, snapshot_file='matrix-p4.snapshot',
        allow_writing_files=True, train_dir=str(tmp_path))
    assert CatBoostRanker().set_params(**saved).fit(pool, eval_set=evaluation, use_best_model=False,
        callbacks=[StopAfter()]).tree_count_ == 2
    resumed = CatBoostRanker().set_params(**saved).fit(pool, eval_set=evaluation, use_best_model=False)
    for name in ('get_leaf_values', 'get_leaf_weights', 'get_test_eval'):
        np.testing.assert_array_equal(getattr(resumed, name)(), getattr(direct, name)())
    assert resumed.evals_result_ == direct.evals_result_
    for fmt in ('cbm', 'json'):
        path = tmp_path/('matrix-p4.' + fmt); direct.save_model(path, format=fmt)
        loaded = CatBoostRanker().load_model(path, format=fmt)
        np.testing.assert_allclose(loaded.predict(evaluation_x, task_type='GPU'), direct.predict(evaluation_x), atol=4e-7, rtol=4e-6)
        if fmt == 'json':
            document = json.loads(path.read_text())
            assert any(s['split_type'] == 'OnlineCtr' for t in document['oblivious_trees'] for s in t['splits'] or [])
    changed = x.copy(); changed[0, 0] = 'new-original-category'
    with pytest.raises(CatBoostError, match='(?i)snapshot.*differ|differ.*snapshot'):
        CatBoostRanker().set_params(**saved).fit(Pool(changed, y, **po), eval_set=evaluation, use_best_model=False)


@pytest.mark.parametrize('loss,method,mode', CASES)
def test_native_matrix_p4_initial_model_and_baseline_continue_exactly(tmp_path, loss, method, mode):
    x, y, _, po, options, _ = problem(loss, method, 'FloatTargetMeanValue', 'Bernoulli')
    options.update(has_time=False, permutation_count=4, leaf_estimation_backtracking=mode)
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
