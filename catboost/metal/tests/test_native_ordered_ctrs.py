"""Native FeatureParallel simple CTR histories, static penalties and recovery."""
import json
import os
from pathlib import Path
import numpy as np
import pytest
from catboost import CatBoostClassifier, CatBoostRegressor, CatBoostError, Pool
from catboost_metal import _ordered
from catboost_metal._categorical import cat_feature_hashes
from catboost_metal._ordered_rng import cuda_ordered_group_history_order
from test_native_greedy_ctrs import inputs as greedy_inputs, readers, only_gpu
from test_native_greedy_api import OBJECTIVES, StopAfter
from test_native_grouped_ctrs import TYPES
from test_ctrs_group import _reference

pytestmark=pytest.mark.skipif(os.environ.get('CATBOOST_NATIVE_METAL_TESTS')!='1',reason='requires rebuilt native Ordered CTR adapter')
LOSSES=list(OBJECTIVES)+[('Lq:q=2.7','Newton')]


def inputs(loss='RMSE',method='Newton',kind='Borders',sampling='No',count=4,history='Sample'):
    cls,x,y,po,config=greedy_inputs('SymmetricTree',loss,method,kind,sampling,count,history)
    config.update(boosting_type='Ordered',score_function='Cosine',min_fold_size=16,fold_len_multiplier=1.7,
        fold_permutation_block=3)
    return cls,x,y,po,config


@pytest.mark.parametrize('loss,method',LOSSES)
@pytest.mark.parametrize('score',['Cosine','NewtonCosine'])
@pytest.mark.parametrize('count',[1,4])
def test_scalar_ctr_objectives_and_model_readers(tmp_path,loss,method,score,count):
    cls,x,y,po,config=inputs(loss,method,count=count);config['score_function']=score
    evaluation=x.copy();evaluation[::19,0]='unknown'
    model=cls().set_params(**config).fit(Pool(x,y,**po),eval_set=Pool(evaluation,y,**po),use_best_model=False)
    assert model.get_metadata()['metal_backend']=='METAL'
    assert model.get_metadata()['metal_permutations']==str(count)
    np.testing.assert_allclose(model.get_test_eval(),model.predict(evaluation,prediction_type='RawFormulaVal',task_type='GPU'),atol=7e-7,rtol=5e-6)
    start=0
    for leaves in model.get_tree_leaf_counts():
        assert model.get_leaf_weights()[start:start+leaves].sum()==pytest.approx(po['weight'].sum(dtype=float),rel=4e-6)
        start+=leaves
    readers(model,cls,evaluation,po,tmp_path)


@pytest.mark.parametrize('kind',TYPES)
@pytest.mark.parametrize('sampling',['No','Bayesian','Bernoulli','Poisson','MVS'])
@pytest.mark.parametrize('quantized',[False,True])
@pytest.mark.parametrize('history',['Sample','Group'])
def test_ctr_prefix_snapshot_restores_all_histories(tmp_path,kind,sampling,quantized,history):
    cls,x,y,po,config=inputs(kind=kind,sampling=sampling,history=history,count=7)
    config.update(random_strength=.35)
    pool=Pool(x,y,**po)
    if quantized:pool.quantize()
    heldout=x.copy();heldout[::17,0]='unseen';evaluation=Pool(heldout,y,**po)
    direct=cls().set_params(**config).fit(pool,eval_set=evaluation,use_best_model=False)
    saved=config|dict(save_snapshot=True,snapshot_interval=0,snapshot_file='ordered-ctr.snapshot',allow_writing_files=True,train_dir=str(tmp_path))
    assert cls().set_params(**saved).fit(pool,eval_set=evaluation,use_best_model=False,callbacks=[StopAfter(2)]).tree_count_==2
    resumed=cls().set_params(**saved).fit(pool,eval_set=evaluation,use_best_model=False)
    for name in ('get_leaf_values','get_leaf_weights','get_test_eval'):
        np.testing.assert_array_equal(getattr(resumed,name)(),getattr(direct,name)())
    assert resumed.evals_result_==direct.evals_result_
    changed=x.copy();changed[0,0]='changed-original-category'
    with pytest.raises(CatBoostError,match='(?i)snapshot.*differ|differ.*snapshot'):
        cls().set_params(**saved).fit(Pool(changed,y,**po),eval_set=evaluation,use_best_model=False)


@pytest.mark.parametrize('kind',TYPES)
@pytest.mark.parametrize('history',['Sample','Group'])
@pytest.mark.parametrize('count',[1,4,7])
def test_native_forests_match_independent_ctr_values_and_resident_prefixes(kind,history,count):
    from catboost.utils import calculate_quantization_grid
    cls,x,y,po,config=inputs(kind=kind,history=history,count=count,sampling='Bernoulli')
    model=cls().set_params(**config).fit(Pool(x,y,**po))
    if count==1:
        order=np.arange(len(x));sizes=np.bincount(po['group_id'])
    else:
        golden=json.loads(Path(__file__).with_name('yeti_pool_shuffle_golden.json').read_text())
        order=np.array(golden['order']);sizes=np.array(golden['group_sizes'])
    hashes=cat_feature_hashes(x[order,0]);targets=y[order];weights=po['weight'][order]
    groups=np.repeat(np.arange(len(sizes)),sizes);banks=[];borders=None
    for p in range(count):
        row_order=cuda_ordered_group_history_order(sizes,p,3)
        values,_=_reference(hashes,targets if kind=='FloatTargetMeanValue' else (targets>.5).astype(np.float32),
            row_order,groups if history=='Group' else np.arange(len(x)),kind,1 if kind=='Buckets' else 0,.5,1.)
        values=values.astype(np.float32)
        if borders is None:borders=np.array(calculate_quantization_grid(values,1,border_type='Uniform'),np.float32)
        banks.append(np.searchsorted(borders,values,side='left').astype(np.uint8)[None,:])
    args={k:config[k] for k in ('iterations','depth','learning_rate','l2_leaf_reg','leaf_estimation_method',
        'leaf_estimation_iterations','bootstrap_type','random_seed','random_strength','score_function','subsample',
        'min_fold_size','fold_len_multiplier','fold_permutation_block','model_size_reg')}
    args['fold_len_multiplier']=float(np.float32(args['fold_len_multiplier']))
    with _ordered.Session(banks[0],targets,np.zeros(len(borders),np.uint32),np.arange(len(borders),dtype=np.uint32),
        sample_weight=weights,permutation_count=count,permutation_bins=banks,group_sizes=sizes,
        ctr_unique_values=[len(np.unique(hashes))],**args) as oracle:
        for _ in range(config['iterations']):oracle.step()
        result=oracle.result()
    np.testing.assert_array_equal(model.get_tree_leaf_counts(),1<<result.depths)
    for name in ('leaf_values','leaf_weights'):
        expected=np.concatenate([getattr(result,name)[i,:1<<d] for i,d in enumerate(result.depths)])
        np.testing.assert_allclose(getattr(model,'get_'+name)(),expected,atol=7e-7 if name=='leaf_values' else 7e-6,rtol=5e-5)


@pytest.mark.parametrize('mode',['No','AnyImprovement','Armijo'])
def test_ordered_ctr_initial_models_baselines_and_best_model(tmp_path,mode):
    cls,x,y,po,config=inputs('Logloss',sampling='Bernoulli');config['leaf_estimation_backtracking']=mode
    pool=Pool(x,y,**po);initial=cls().set_params(**(config|dict(iterations=2))).fit(pool)
    pool.set_baseline(np.linspace(-.1,.2,len(x),dtype=np.float32))
    direct=cls().set_params(**config).fit(pool,init_model=initial,eval_set=pool,use_best_model=False)
    saved=config|dict(save_snapshot=True,snapshot_interval=0,snapshot_file='initial.snapshot',allow_writing_files=True,train_dir=str(tmp_path))
    cls().set_params(**saved).fit(pool,init_model=initial,eval_set=pool,use_best_model=False,callbacks=[StopAfter(2)])
    resumed=cls().set_params(**saved).fit(pool,init_model=initial,eval_set=pool,use_best_model=False)
    for name in ('get_leaf_values','get_leaf_weights','get_test_eval'):
        np.testing.assert_array_equal(getattr(resumed,name)(),getattr(direct,name)())
    assert resumed.evals_result_==direct.evals_result_
    np.testing.assert_allclose(direct.get_test_eval(),direct.predict(x,prediction_type='RawFormulaVal',task_type='GPU')+pool.get_baseline().ravel(),atol=7e-7,rtol=5e-6)


def test_has_time_collapses_ctr_and_fold_history_count():
    cls,x,y,po,config=inputs(count=7)
    a=cls().set_params(**(config|dict(has_time=True))).fit(Pool(x,y,**po))
    b=cls().set_params(**(config|dict(has_time=True,permutation_count=1))).fit(Pool(x,y,**po))
    assert a.get_metadata()['metal_permutations']=='1'
    np.testing.assert_array_equal(a.get_leaf_values(),b.get_leaf_values())
