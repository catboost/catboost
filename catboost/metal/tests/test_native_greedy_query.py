"""Native GPU QueryRMSE/SoftMax greedy weights, categoricals and recovery."""
import json,os
import numpy as np
import pytest
from catboost import CatBoost,CatBoostRanker,CatBoostError,Pool
from test_native_greedy_api import POLICIES,StopAfter,sampler
from test_native_query_api import pool_for,params
from test_native_ranking_ctr_p1 import problem as ctr_problem
from test_native_grouped_ctrs import TYPES

pytestmark=pytest.mark.skipif(os.environ.get('CATBOOST_NATIVE_METAL_QUERY_TESTS')!='1',reason='rebuilt native query adapter required')
LOSSES=['QueryRMSE','QuerySoftMax:beta=0.7;lambda=0.03']

@pytest.fixture(autouse=True)
def gpu_only(monkeypatch):
    original=CatBoost._fit
    def checked(self,*a,**kw):
        assert self.get_params().get('task_type')=='GPU'
        return original(self,*a,**kw)
    monkeypatch.setattr(CatBoost,'_fit',checked)


def config(policy,loss,**extra):
    p=params(loss_function=loss,iterations=5,grow_policy=policy,leaf_estimation_iterations=3)
    if policy=='Lossguide':p['max_leaves']=6
    return p|extra


def readers(model,x,tmp_path):
    raw=model.predict(x,task_type='GPU')
    np.testing.assert_allclose(model.predict(x),raw,rtol=4e-6,atol=8e-7)
    for fmt in ['cbm','json']:
        p=tmp_path/('query.'+fmt);model.save_model(p,format=fmt)
        loaded=CatBoostRanker().load_model(p,format=fmt)
        np.testing.assert_allclose(loaded.predict(x,task_type='GPU'),raw,rtol=4e-6,atol=8e-7)
        np.testing.assert_allclose(loaded.predict(x),raw,rtol=4e-6,atol=8e-7)
        if fmt=='json': assert 'trees' in json.loads(p.read_text())


@pytest.mark.parametrize('policy',POLICIES)
@pytest.mark.parametrize('loss',LOSSES)
@pytest.mark.parametrize('method',['Newton','Gradient'])
@pytest.mark.parametrize('score',['L2','Cosine','NewtonL2','NewtonCosine','SolarL2','LOOL2','SatL2'])
def test_native_weighted_query_greedy_methods_scores_and_readers(tmp_path,policy,loss,method,score):
    pool,x,_,_,w=pool_for(loss,separate=True)
    merged,*_=pool_for(loss)
    p=config(policy,loss,score_function=score,leaf_estimation_method=method)
    model=CatBoostRanker().set_params(**p).fit(pool,eval_set=pool,use_best_model=False)
    same=CatBoostRanker().set_params(**p).fit(merged,eval_set=merged,use_best_model=False)
    np.testing.assert_array_equal(model.get_leaf_values(),same.get_leaf_values())
    np.testing.assert_array_equal(model.get_leaf_weights(),same.get_leaf_weights())
    np.testing.assert_array_equal(model.predict(x,task_type='GPU'),same.predict(x,task_type='GPU'))
    assert model.evals_result_==same.evals_result_
    at=0
    for count in model.get_tree_leaf_counts():
        assert sum(model.get_leaf_weights()[at:at+count])==pytest.approx(w.sum(dtype=float),rel=3e-6)
        at+=count
    assert model.get_all_params()['grow_policy']==policy
    np.testing.assert_allclose(model.get_test_eval(),model.predict(x,task_type='GPU'),rtol=4e-6,atol=8e-7)
    readers(model,x,tmp_path)


@pytest.mark.parametrize('policy',POLICIES)
@pytest.mark.parametrize('loss',LOSSES)
@pytest.mark.parametrize('mode',['No','AnyImprovement','Armijo'])
@pytest.mark.parametrize('sampling',['No','Bayesian','Bernoulli','Poisson'])
def test_numeric_query_exact_snapshot_baseline_and_initial_model(tmp_path,policy,loss,mode,sampling):
    pool,x,*_=pool_for(loss,separate=True)
    p=config(policy,loss,leaf_estimation_backtracking=mode,random_strength=.3,**sampler(sampling))
    initial=CatBoostRanker(**(p|{'iterations':2})).fit(pool)
    pool.set_baseline(np.linspace(-.1,.1,len(x),dtype=np.float32))
    direct=CatBoostRanker().set_params(**p).fit(pool,eval_set=pool,use_best_model=False,init_model=initial)
    saved=p|dict(save_snapshot=True,snapshot_interval=0,snapshot_file='query.snapshot',allow_writing_files=True,train_dir=str(tmp_path))
    CatBoostRanker().set_params(**saved).fit(pool,eval_set=pool,use_best_model=False,init_model=initial,callbacks=[StopAfter(2)])
    resumed=CatBoostRanker().set_params(**saved).fit(pool,eval_set=pool,use_best_model=False,init_model=initial)
    for name in ['get_leaf_values','get_leaf_weights','get_test_eval']:
        np.testing.assert_array_equal(getattr(direct,name)(),getattr(resumed,name)())
    assert direct.evals_result_==resumed.evals_result_
    readers(resumed,x,tmp_path)


@pytest.mark.parametrize('policy',POLICIES)
@pytest.mark.parametrize('loss',LOSSES)
@pytest.mark.parametrize('kind',['OneHot',*TYPES])
@pytest.mark.parametrize('count',[1,4])
def test_grouped_categorical_query_datasets_raw_quantized_snapshot(tmp_path,policy,loss,kind,count):
    actual_kind='Borders' if kind=='OneHot' else kind
    x,y,_,po,p,_=ctr_problem(loss,'Newton',actual_kind,'No')
    p.update(config(policy,loss),has_time=count==1,permutation_count=count,
        ctr_history_unit='Group' if count==4 else 'Sample',one_hot_max_size=255 if kind=='OneHot' else 1)
    heldout=x.copy();heldout[::17,0]='unknown'
    pool=Pool(x,y,**po);evaluation=Pool(heldout,y,**po)
    if count==4:pool.quantize()
    direct=CatBoostRanker().set_params(**p).fit(pool,eval_set=evaluation,use_best_model=False)
    saved=p|dict(save_snapshot=True,snapshot_interval=0,snapshot_file='categorical.snapshot',allow_writing_files=True,train_dir=str(tmp_path))
    CatBoostRanker().set_params(**saved).fit(pool,eval_set=evaluation,use_best_model=False,callbacks=[StopAfter(2)])
    resumed=CatBoostRanker().set_params(**saved).fit(pool,eval_set=evaluation,use_best_model=False)
    for name in ['get_leaf_values','get_leaf_weights','get_test_eval']:
        np.testing.assert_array_equal(getattr(direct,name)(),getattr(resumed,name)())
    assert direct.evals_result_==resumed.evals_result_
    if kind!='OneHot': assert direct.get_metadata()['metal_permutations']==str(count)
    readers(resumed,heldout,tmp_path)
    changed=x.copy();changed[0,0]='changed-original-category'
    with pytest.raises(CatBoostError,match='(?i)snapshot.*differ|differ.*snapshot'):
        CatBoostRanker().set_params(**saved).fit(Pool(changed,y,**po),eval_set=evaluation,use_best_model=False)
