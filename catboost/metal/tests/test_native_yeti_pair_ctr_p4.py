"""Native categorical YetiRankPairwise multi-cursor stochastic training and recovery."""
import json
import os
from pathlib import Path
import numpy as np
import pytest
from catboost import CatBoost, CatBoostRanker, CatBoostError, Pool
from test_native_ranking_ctr_p1 import problem
from test_native_grouped_ctrs import TYPES, StopAfter

pytestmark = pytest.mark.skipif(os.environ.get('CATBOOST_NATIVE_METAL_QUERY_TESTS') != '1',
                               reason='requires rebuilt native YetiRankPairwise P4 adapter')
SAMPLERS = [('No','Object'),('Bayesian','Object'),('Bernoulli','Object'),('Bernoulli','Group')]


@pytest.fixture(autouse=True)
def only_gpu(monkeypatch):
    original = CatBoost._fit
    def checked(self, *args, **kwargs):
        assert self.get_params().get('task_type') == 'GPU'
        return original(self, *args, **kwargs)
    monkeypatch.setattr(CatBoost, '_fit', checked)


def inputs(kind,sampler,method,history='Group',unit='Object'):
    x,y,groups,po,options,offsets = problem('YetiRankPairwise',method,kind,history=history)
    options.update(has_time=False,permutation_count=4,bootstrap_type=sampler,sampling_unit=unit,random_strength=.7)
    if sampler == 'Bernoulli': options['subsample'] = .7
    return x,y,groups,po,options,offsets


@pytest.mark.parametrize('kind', TYPES)
@pytest.mark.parametrize('sampler,unit', SAMPLERS)
@pytest.mark.parametrize('method', ['Simple','Newton','Gradient'])
@pytest.mark.parametrize('prequantized', [False,True])
def test_native_yeti_pair_p4_ctrs_restore_oracle_rng_cursors_metrics_and_readers(tmp_path, kind, sampler, unit, method, prequantized):
    x,y,_,po,options,_ = inputs(kind,sampler,method,'Sample' if prequantized else 'Group',unit)
    pool = Pool(x,y,**po)
    if prequantized: pool.quantize()
    evaluation_x = x.copy(); evaluation_x[::17,0] = 'unseen'
    evaluation = Pool(evaluation_x,y,**po)
    direct = CatBoostRanker().set_params(**options).fit(pool,eval_set=evaluation,use_best_model=False)
    assert direct.get_metadata()['metal_permutations'] == '4'
    assert direct.get_metadata()['metal_yeti_pair_rng'] == 'item_iteration_dataset_domains_v2'
    np.testing.assert_allclose(direct.get_test_eval(),direct.predict(evaluation_x,task_type='GPU'),atol=4e-7,rtol=4e-6)
    saved = options | dict(save_snapshot=True,snapshot_interval=0,snapshot_file='yeti-pair-p4.snapshot',
        allow_writing_files=True,train_dir=str(tmp_path))
    assert CatBoostRanker().set_params(**saved).fit(pool,eval_set=evaluation,use_best_model=False,
        callbacks=[StopAfter()]).tree_count_ == 2
    resumed = CatBoostRanker().set_params(**saved).fit(pool,eval_set=evaluation,use_best_model=False)
    for name in ('get_leaf_values','get_leaf_weights','get_test_eval'):
        np.testing.assert_array_equal(getattr(resumed,name)(),getattr(direct,name)())
    assert resumed.evals_result_ == direct.evals_result_
    for fmt in ('cbm','json'):
        path = tmp_path/('yeti-pair-p4.'+fmt); direct.save_model(path,format=fmt)
        loaded = CatBoostRanker().load_model(path,format=fmt)
        np.testing.assert_allclose(loaded.predict(evaluation_x,task_type='GPU'),direct.predict(evaluation_x),atol=4e-7,rtol=4e-6)
        if fmt == 'json':
            document = json.loads(path.read_text())
            assert any(s['split_type']=='OnlineCtr' for t in document['oblivious_trees'] for s in t['splits'] or [])
    changed = x.copy(); changed[0,0] = 'new-original-category'
    with pytest.raises(CatBoostError,match='(?i)snapshot.*differ|differ.*snapshot'):
        CatBoostRanker().set_params(**saved).fit(Pool(changed,y,**po),eval_set=evaluation,use_best_model=False)
    with pytest.raises(CatBoostError,match='(?i)snapshot.*differ|differ.*snapshot'):
        CatBoostRanker().set_params(**(saved | dict(permutation_count=2))).fit(pool,eval_set=evaluation,use_best_model=False)


@pytest.mark.parametrize('sampler,unit', SAMPLERS)
@pytest.mark.parametrize('method', ['Simple','Newton','Gradient'])
def test_native_yeti_pair_p4_initial_model_baseline_and_snapshot_resume(tmp_path, sampler, unit, method):
    x,y,_,po,options,_ = inputs('FloatTargetMeanValue',sampler,method,unit=unit)
    pool = Pool(x,y,**po)
    initial = CatBoostRanker().set_params(**(options | dict(iterations=2))).fit(pool)
    pool.set_baseline(np.linspace(-.1,.2,len(x),dtype=np.float32))
    direct = CatBoostRanker().set_params(**options).fit(pool,init_model=initial,eval_set=pool,use_best_model=False)
    saved = options | dict(save_snapshot=True,snapshot_interval=0,snapshot_file='initial.snapshot',
        allow_writing_files=True,train_dir=str(tmp_path))
    partial = CatBoostRanker().set_params(**saved).fit(pool,init_model=initial,eval_set=pool,
        use_best_model=False,callbacks=[StopAfter()])
    assert partial.tree_count_ == 4
    resumed = CatBoostRanker().set_params(**saved).fit(pool,init_model=initial,eval_set=pool,use_best_model=False)
    assert resumed.tree_count_ == 6
    for name in ('get_leaf_values','get_leaf_weights','get_test_eval'):
        np.testing.assert_array_equal(getattr(resumed,name)(),getattr(direct,name)())
    assert resumed.evals_result_ == direct.evals_result_
    np.testing.assert_allclose(np.array(direct.get_test_eval()),
        direct.predict(x,task_type='GPU')+pool.get_baseline().ravel(),atol=4e-7,rtol=4e-6)


@pytest.mark.parametrize('kind', TYPES)
@pytest.mark.parametrize('sampler', ['No','Bernoulli'])
@pytest.mark.parametrize('method', ['Simple','Newton','Gradient'])
@pytest.mark.parametrize('history', ['Sample','Group'])
def test_native_p4_forest_matches_independent_ctr_histories_and_resident_oracle(kind, sampler, method, history):
    from catboost.utils import calculate_quantization_grid
    from catboost_metal import _yeti_pair
    from catboost_metal._categorical import cat_feature_hashes
    from catboost_metal._data import cuda_history_order
    from test_ctrs_group import _reference
    x,y,_,po,options,_ = inputs(kind,sampler,method,history)
    options['random_strength'] = 0
    model = CatBoostRanker().set_params(**options).fit(Pool(x,y,**po))
    # Compile yeti_pool_shuffle_probe.cpp against the checked-out fast/shuffle
    # headers to regenerate this fixture. It applies the existing shared Pool
    # preprocessing order; no model fitting is involved in fixture generation.
    golden = json.loads(Path(__file__).with_name('yeti_pool_shuffle_golden.json').read_text())
    order = np.array(golden['order']); sizes = np.array(golden['group_sizes'])
    assert sorted(order) == list(range(len(x))) and sizes.sum() == len(x)
    offsets = np.r_[0,np.cumsum(sizes)].astype(np.uint32)
    groups = np.repeat(np.arange(len(sizes)),sizes)
    hashes = cat_feature_hashes(x[order,0]); targets = y[order]; weights = po['weight'][order]
    matrices = []; borders = None
    for permutation in range(4):
        query_order = cuda_history_order(len(sizes),permutation)
        row_order = np.concatenate([np.arange(offsets[g],offsets[g+1],dtype=np.uint32) for g in query_order])
        values,_ = _reference(hashes,targets if kind=='FloatTargetMeanValue' else (targets>.5).astype(np.float32),
            row_order,groups if history=='Group' else np.arange(len(x)),kind,1 if kind=='Buckets' else 0,.5,1.)
        values = values.astype(np.float32)
        if borders is None: borders = np.array(calculate_quantization_grid(values,1,border_type='Uniform'),np.float32)
        matrices.append(np.searchsorted(borders,values,side='left').astype(np.uint8)[None,:])
    args = {name:options[name] for name in ('iterations','depth','learning_rate','l2_leaf_reg',
        'leaf_estimation_iterations','leaf_estimation_method','bootstrap_type','random_seed','random_strength','score_function')}
    if sampler == 'Bernoulli': args['subsample'] = .7
    with _yeti_pair.TrainingSession(matrices[0],targets,np.zeros(len(borders),np.uint32),np.arange(len(borders),dtype=np.uint32),
            group_offsets=offsets,sample_weight=weights,permutations=7,decay=.85,non_diagonal_regularization=.2,**args) as oracle:
        oracle.configure_permutations(matrices); trees = [oracle.step() for _ in range(options['iterations'])]
    np.testing.assert_array_equal(model.get_tree_leaf_counts(),[1<<tree.depth for tree in trees])
    np.testing.assert_allclose(model.get_leaf_values(),np.concatenate([tree.leaf_values for tree in trees]),atol=6e-7,rtol=5e-5)
    np.testing.assert_allclose(model.get_leaf_weights(),np.concatenate([tree.leaf_weights for tree in trees]),atol=6e-6,rtol=5e-6)
