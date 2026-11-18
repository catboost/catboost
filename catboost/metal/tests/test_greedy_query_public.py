"""Standalone greedy query ranker lifecycle and ordinary CatBoost readers."""
import json
import numpy as np
import pytest
from catboost import CatBoost,CatBoostRanker,Pool
from catboost_metal import CatBoostMetalRanker
from catboost_metal import _greedy
from catboost_metal._greedy_training import run_training
from test_native_query_api import dataset
from test_native_greedy_api import POLICIES,sampler
from test_querywise_training import problem
from test_greedy_querywise import OBJECTIVES,options
from catboost_metal._query_data import query_metric

@pytest.fixture(autouse=True)
def no_cpu_fit(monkeypatch):
    def forbidden(*a,**kw):raise AssertionError('Standalone acceptance must never fit CPU CatBoost')
    monkeypatch.setattr(CatBoost,'_fit',forbidden)


def data(objective,categorical=False):
    x,y,g,w,gw=dataset()
    if objective=='QuerySoftMax':y=np.exp(y/3).astype(np.float32)
    if categorical:
        x=x.astype(object);x[:,2]=np.array(['a','b','c'])[np.arange(len(x))%3]
    return x,y,g,w,gw


def config(policy,objective,method='Newton',sampling='No',categorical=False,**extra):
    p=dict(loss_function=objective+(':beta=0.7;lambda=0.03' if objective=='QuerySoftMax' else ''),
        grow_policy=policy,iterations=6,depth=3,learning_rate=.18,l2_leaf_reg=2.3,score_function='Cosine',
        random_strength=.3,random_seed=71,leaf_estimation_method=method,leaf_estimation_iterations=4,
        leaf_estimation_backtracking='Armijo',**sampler(sampling))
    if policy=='Lossguide':p['max_leaves']=6
    if categorical:p.update(cat_features=[2],one_hot_max_size=8)
    return p|extra


def readers(model,x,tmp_path):
    raw=model.predict(x,task_type='GPU')
    np.testing.assert_allclose(model.predict(x),raw,rtol=5e-6,atol=1e-6)
    for fmt in ['cbm','json']:
        path=tmp_path/('standalone-query.'+fmt);model.save_model(path,format=fmt)
        reader=CatBoostRanker().load_model(path,format=fmt)
        np.testing.assert_allclose(reader.predict(x),raw,rtol=5e-6,atol=1e-6)
        np.testing.assert_allclose(reader.predict(x,task_type='GPU'),raw,rtol=5e-6,atol=1e-6)
        if fmt=='json': assert 'trees' in json.loads(path.read_text())


@pytest.mark.parametrize('objective',OBJECTIVES)
@pytest.mark.parametrize('policy',POLICIES)
@pytest.mark.parametrize('method',['Newton','Gradient'])
@pytest.mark.parametrize('sampling',['No','Bayesian','Bernoulli','Poisson'])
@pytest.mark.parametrize('categorical',[False,True])
def test_public_query_ranker_weighted_recovery_and_readers(tmp_path,objective,policy,method,sampling,categorical):
    x,y,g,w,gw=data(objective,categorical);heldout=x.copy()
    if categorical:heldout[::19,2]='unknown'
    p=config(policy,objective,method,sampling,categorical)
    fit=dict(group_id=g,sample_weight=w,group_weight=gw,eval_set=(heldout,y,g,w,gw),use_best_model=False)
    direct=CatBoostMetalRanker(**p).fit(x,y,**fit)
    saved=dict(save_snapshot=True,snapshot_file=tmp_path/'query.npz',snapshot_interval=0)
    partial=CatBoostMetalRanker(**p).fit(x,y,**fit,**saved,callback=lambda info:info.iteration<2)
    assert partial.tree_count_==2
    resumed=CatBoostMetalRanker(**p).fit(x,y,**fit,**saved)
    assert resumed._result.resumed_iterations==2
    for a,b in zip(direct._result.trees,resumed._result.trees):
        np.testing.assert_array_equal(a.nodes,b.nodes);np.testing.assert_array_equal(a.leaf_values,b.leaf_values)
        np.testing.assert_array_equal(a.leaf_weights,b.leaf_weights)
    np.testing.assert_array_equal(direct.training_predictions_,resumed.training_predictions_)
    np.testing.assert_array_equal(direct.predict(heldout,task_type='GPU'),resumed.predict(heldout,task_type='GPU'))
    assert direct.evals_result_==resumed.evals_result_
    raw=direct.predict(heldout,task_type='GPU')
    value=query_metric(raw,y,w*gw,direct.group_offsets_,objective,.7,.03)
    assert next(iter(direct.evals_result_['validation'].values()))[-1]==pytest.approx(value,rel=8e-6,abs=1e-6)
    readers(resumed,heldout,tmp_path)
    changed=g.copy();changed[:3]=1000
    with pytest.raises(ValueError,match='(?i)snapshot.*(differ|match)|(differ|match).*snapshot'):
        CatBoostMetalRanker(**p).fit(x,y,**(fit|dict(group_id=changed)),**saved)


@pytest.mark.parametrize('objective',OBJECTIVES)
@pytest.mark.parametrize('policy',POLICIES)
@pytest.mark.parametrize('count',[1,4,7])
def test_grouped_lifecycle_ctr_banks_snapshot_metric_and_identity(tmp_path,objective,policy,count,monkeypatch):
    bins,y,w,off,raw,cf,cb=problem(objective)
    banks=np.stack([bins.copy() for _ in range(count)])
    for p in range(1,count):banks[p,0]=np.roll(bins[0],3*p)
    opts=options(objective,policy)|dict(group_offsets=off,sample_weight=w,initial_predictions=raw,
        permutation_bins=banks,eval_bins=bins,eval_targets=y,eval_weight=w,eval_group_offsets=off,
        leaf_estimation_backtracking='Armijo',random_strength=.4,random_seed=15,bias=0.,use_best_model=False)
    direct=run_training(bins,y,cf,cb,**opts)
    saved=dict(save_snapshot=True,snapshot_interval=0,snapshot_file=tmp_path/'banks.npz')
    run_training(bins,y,cf,cb,**opts,**saved,callback=lambda info:info.iteration<1)
    resumed=run_training(bins,y,cf,cb,**opts,**saved)
    np.testing.assert_array_equal(resumed.predictions,direct.predictions)
    np.testing.assert_array_equal(resumed.eval_predictions,direct.eval_predictions)
    assert resumed.evals_result==direct.evals_result
    for a,b in zip(resumed.trees,direct.trees):
        np.testing.assert_array_equal(a.nodes,b.nodes);np.testing.assert_array_equal(a.leaf_values,b.leaf_values)
    def forbidden():pytest.fail('snapshot identity must fail before loading Metal')
    monkeypatch.setattr(_greedy,'build_library',forbidden)
    with pytest.raises(ValueError,match='(?i)snapshot.*(differ|match)|(differ|match).*snapshot'):
        run_training(bins,y,cf,cb,**(opts|dict(query_beta=.6)),**saved)


@pytest.mark.parametrize('policy',POLICIES)
@pytest.mark.parametrize('metric',['QueryRMSE','NDCG:top=3','PFound'])
def test_ranking_metrics_subgroups_best_trimming_and_extended_resume(tmp_path,policy,metric):
    x,y,g,w,gw=data('QueryRMSE');y=(y-y.min())/(y.max()-y.min())
    p=config(policy,'QueryRMSE',iterations=9,eval_metric=metric,learning_rate=.4)
    sub=np.tile(np.array(['same','other','same','third','fourth','fifth']),20)
    fit=dict(group_id=g,subgroup_id=sub,sample_weight=w,group_weight=gw,
        eval_set=(x,y[::-1],g,w,gw,sub),use_best_model=True)
    direct=CatBoostMetalRanker(**p).fit(x,y,**fit)
    assert direct.tree_count_==direct.best_iteration_+1
    saved=dict(save_snapshot=True,snapshot_interval=0,snapshot_file=tmp_path/'trim.npz')
    CatBoostMetalRanker(**p).fit(x,y,**fit,**saved,callback=lambda info:info.iteration<3)
    resumed=CatBoostMetalRanker(**p).fit(x,y,**fit,**saved)
    np.testing.assert_array_equal(resumed.predict(x,task_type='GPU'),direct.predict(x,task_type='GPU'))
    assert resumed.evals_result_==direct.evals_result_
    p['iterations']=11
    extended=CatBoostMetalRanker(**p).fit(x,y,**fit,**saved)
    fresh=CatBoostMetalRanker(**p).fit(x,y,**fit)
    assert extended._result.resumed_iterations==9
    np.testing.assert_array_equal(extended.predict(x,task_type='GPU'),fresh.predict(x,task_type='GPU'))
    assert extended.evals_result_==fresh.evals_result_
