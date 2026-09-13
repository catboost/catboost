"""Native greedy simple CTR P1/P4: histories, forests, recovery and readers."""
import json
import os
from pathlib import Path
import numpy as np
import pytest
from catboost import CatBoost, CatBoostClassifier, CatBoostRegressor, CatBoostError, Pool
from catboost_metal import _greedy
from catboost_metal._categorical import cat_feature_hashes
from catboost_metal._data import cuda_history_order, cuda_search_permutation
from test_native_greedy_api import POLICIES, OBJECTIVES, StopAfter, sampler
from test_native_ranking_ctr_p1 import problem as ranking_problem
from test_native_grouped_ctrs import TYPES

pytestmark=pytest.mark.skipif(os.environ.get('CATBOOST_NATIVE_METAL_TESTS')!='1',reason='requires rebuilt native greedy CTR adapter')


@pytest.fixture(autouse=True)
def only_gpu(monkeypatch):
    original=CatBoost._fit
    def checked(self,*args,**kwargs):
        assert self.get_params().get('task_type')=='GPU'
        return original(self,*args,**kwargs)
    monkeypatch.setattr(CatBoost,'_fit',checked)


def inputs(policy='Depthwise',loss='RMSE',method='Newton',kind='Borders',sampling='No',count=4,history='Sample'):
    x,y,_,po,config,_=ranking_problem('YetiRank:permutations=7;decay=0.85','Newton',kind,'No')
    config.update(loss_function=loss,grow_policy=policy,has_time=count==1,permutation_count=count,
        leaf_estimation_method=method,leaf_estimation_iterations=3,leaf_estimation_backtracking='No',
        ctr_history_unit=history,model_size_reg=.8,**sampler(sampling))
    config.pop('sampling_unit',None)
    if policy=='Lossguide':config['max_leaves']=5
    cls=CatBoostClassifier if loss in ('Logloss','CrossEntropy') else CatBoostRegressor
    if loss=='Logloss':y=(y>.5).astype(np.float32)
    return cls,x,y,po,config


def readers(model,cls,x,po,tmp_path):
    prediction=model.predict(Pool(x,**{k:v for k,v in po.items() if k!='weight'}),prediction_type='RawFormulaVal',task_type='GPU')
    for fmt in ('cbm','json'):
        path=tmp_path/('greedy-ctr.'+fmt);model.save_model(path,format=fmt)
        loaded=cls().load_model(path,format=fmt)
        np.testing.assert_allclose(loaded.predict(x,prediction_type='RawFormulaVal',task_type='GPU'),prediction,atol=7e-7,rtol=5e-6)
        np.testing.assert_allclose(loaded.predict(x,prediction_type='RawFormulaVal'),prediction,atol=7e-7,rtol=5e-6)
        if fmt=='json' and np.any(model.get_tree_leaf_counts()>1):
            assert 'OnlineCtr' in path.read_text()


@pytest.mark.parametrize('policy',POLICIES)
@pytest.mark.parametrize('loss,method',OBJECTIVES)
@pytest.mark.parametrize('count',[1,4])
def test_eleven_registered_losses_accept_ctr_datasets_and_original_weights(tmp_path,policy,loss,method,count):
    cls,x,y,po,config=inputs(policy,loss,method,count=count)
    evaluation=x.copy();evaluation[::19,0]='unknown'
    model=cls().set_params(**config).fit(Pool(x,y,**po),eval_set=Pool(evaluation,y,**po),use_best_model=False)
    assert model.get_metadata()['metal_permutations']==str(count)
    np.testing.assert_allclose(model.get_test_eval(),model.predict(evaluation,prediction_type='RawFormulaVal',task_type='GPU'),atol=7e-7,rtol=5e-6)
    begin=0
    for count in model.get_tree_leaf_counts():
        assert np.sum(model.get_leaf_weights()[begin:begin+count])==pytest.approx(po['weight'].sum(dtype=float),rel=4e-6)
        begin+=count
    readers(model,cls,evaluation,po,tmp_path)


@pytest.mark.parametrize('policy',POLICIES)
@pytest.mark.parametrize('kind',TYPES)
@pytest.mark.parametrize('sampling',['No','Bayesian','Bernoulli','Poisson'])
@pytest.mark.parametrize('quantized',[False,True])
def test_ctr_p4_exact_snapshot_with_every_sampler_and_history_unit(tmp_path,policy,kind,sampling,quantized):
    cls,x,y,po,config=inputs(policy,kind=kind,sampling=sampling,history='Group' if quantized else 'Sample')
    config.update(random_strength=.35,score_function='Cosine')
    pool=Pool(x,y,**po)
    if quantized:pool.quantize()
    heldout=x.copy();heldout[::17,0]='unseen'
    evaluation=Pool(heldout,y,**po)
    direct=cls().set_params(**config).fit(pool,eval_set=evaluation,use_best_model=False)
    saved=config|dict(save_snapshot=True,snapshot_interval=0,snapshot_file='greedy-ctr.snapshot',allow_writing_files=True,train_dir=str(tmp_path))
    assert cls().set_params(**saved).fit(pool,eval_set=evaluation,use_best_model=False,callbacks=[StopAfter(2)]).tree_count_==2
    resumed=cls().set_params(**saved).fit(pool,eval_set=evaluation,use_best_model=False)
    for name in ('get_leaf_values','get_leaf_weights','get_test_eval'):
        np.testing.assert_array_equal(getattr(resumed,name)(),getattr(direct,name)())
    assert resumed.evals_result_==direct.evals_result_
    readers(direct,cls,heldout,po,tmp_path)
    changed=x.copy();changed[0,0]='changed-original-category'
    with pytest.raises(CatBoostError,match='(?i)snapshot.*differ|differ.*snapshot'):
        cls().set_params(**saved).fit(Pool(changed,y,**po),eval_set=evaluation,use_best_model=False)


@pytest.mark.parametrize('policy',POLICIES)
@pytest.mark.parametrize('mode',['No','AnyImprovement','Armijo'])
def test_ctr_p4_backtracking_initial_models_baselines_and_resume(tmp_path,policy,mode):
    cls,x,y,po,config=inputs(policy,'Logloss',sampling='Bernoulli');config['leaf_estimation_backtracking']=mode
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


@pytest.mark.parametrize('policy',POLICIES)
@pytest.mark.parametrize('loss',['Quantile:alpha=0.7','MAE','MAPE'])
def test_exact_ctr_p4_leaves_and_has_time_p1_fallback(tmp_path,policy,loss):
    cls,x,y,po,config=inputs(policy,loss,'Exact',sampling='Bernoulli')
    model=cls().set_params(**config).fit(Pool(x,y,**po));readers(model,cls,x,po,tmp_path)
    a=cls().set_params(**(config|dict(has_time=True))).fit(Pool(x,y,**po))
    b=cls().set_params(**(config|dict(has_time=True,permutation_count=1))).fit(Pool(x,y,**po))
    assert a.get_metadata()['metal_permutations']=='1'
    np.testing.assert_array_equal(a.get_leaf_values(),b.get_leaf_values())


@pytest.mark.parametrize('policy',POLICIES)
@pytest.mark.parametrize('kind',TYPES)
@pytest.mark.parametrize('history',['Sample','Group'])
def test_greedy_forests_match_independent_ctr_banks_and_skip_symmetric_size_penalties(policy,kind,history):
    from catboost.utils import calculate_quantization_grid
    from test_ctrs_group import _reference
    cls,x,y,po,config=inputs(policy,kind=kind,history=history,sampling='Bernoulli')
    model=cls().set_params(**config).fit(Pool(x,y,**po))
    unpenalized=cls().set_params(**(config|dict(model_size_reg=0))).fit(Pool(x,y,**po))
    np.testing.assert_array_equal(model.get_leaf_values(),unpenalized.get_leaf_values())
    golden=json.loads(Path(__file__).with_name('yeti_pool_shuffle_golden.json').read_text())
    order=np.array(golden['order']);sizes=np.array(golden['group_sizes']);offsets=np.r_[0,np.cumsum(sizes)]
    assert len(order)==len(x)
    hashes=cat_feature_hashes(x[order,0]);targets=y[order];weights=po['weight'][order]
    groups=np.repeat(np.arange(len(sizes)),sizes);banks=[];borders=None
    for p in range(4):
        query_order=cuda_history_order(len(sizes),p)
        row_order=np.concatenate([np.arange(offsets[g],offsets[g+1],dtype=np.uint32) for g in query_order])
        values,_=_reference(hashes,targets if kind=='FloatTargetMeanValue' else (targets>.5).astype(np.float32),
            row_order,groups if history=='Group' else np.arange(len(x)),kind,1 if kind=='Buckets' else 0,.5,1.)
        values=values.astype(np.float32)
        if borders is None:borders=np.array(calculate_quantization_grid(values,1,border_type='Uniform'),np.float32)
        banks.append(np.searchsorted(borders,values,side='left').astype(np.uint8)[None,:])
    args={k:config[k] for k in ('iterations','depth','grow_policy','learning_rate','l2_leaf_reg',
        'leaf_estimation_method','leaf_estimation_iterations','bootstrap_type','random_seed','random_strength','score_function','subsample')}
    if policy=='Lossguide':args['max_leaves']=config['max_leaves']
    with _greedy.Session(banks[0],targets,np.zeros(len(borders),np.uint32),np.arange(len(borders),dtype=np.uint32),
            sample_weight=weights,**args) as oracle:
        oracle.configure_permutations(banks);trees=[]
        for t in range(config['iterations']):
            oracle.select_permutation(cuda_search_permutation(config['random_seed'],t,4));trees.append(oracle.step())
    np.testing.assert_array_equal(model.get_tree_leaf_counts(),[len(t.leaf_values) for t in trees])
    at=0
    for tree in trees:
        count=len(tree.leaf_values)
        np.testing.assert_allclose(np.sort(model.get_leaf_values()[at:at+count]),np.sort(tree.leaf_values),atol=7e-7,rtol=5e-5)
        np.testing.assert_allclose(np.sort(model.get_leaf_weights()[at:at+count]),np.sort(tree.leaf_weights),atol=7e-6,rtol=5e-6)
        at+=count
