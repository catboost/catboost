"""Standalone PFound tracking, snapshot identity, best models and normal export."""
import json
import numpy as np
import pytest
from catboost import CatBoost,CatBoostRanker
from catboost_metal import CatBoostMetalRanker
from catboost_metal._training import _shared_metric
from test_yeti_pair_training import problem

@pytest.fixture(autouse=True)
def no_cpu_fit(monkeypatch):
    monkeypatch.setattr(CatBoost,'_fit',lambda *a,**kw:pytest.fail('CPU CatBoost training is forbidden'))


def setup(method='Simple',kind='Bernoulli',unit='Object'):
    args=problem();x=args['bins'].T.astype(np.float32);y=args['targets'];w=args['sample_weight']
    group=np.repeat(np.arange(len(args['group_offsets'])-1),np.diff(args['group_offsets']))
    options=dict(loss_function='YetiRankPairwise:permutations=7;decay=.85',iterations=6,depth=3,random_seed=781,
        leaf_estimation_method=method,leaf_estimation_iterations=1 if method=='Simple' else 3,bootstrap_type=kind,
        sampling_unit=unit,learning_rate=.1,l2_leaf_reg=3.,bayesian_matrix_reg=.2,border_count=8,random_strength=0.)
    if kind=='Bernoulli':options['subsample']=.7
    if kind=='Bayesian':options['bagging_temperature']=.7
    fit=dict(group_id=group,sample_weight=w,eval_set=(x,y,group,w),use_best_model=False)
    return args,x,y,w,group,options,fit


@pytest.mark.parametrize('method',['Simple','Newton','Gradient'])
@pytest.mark.parametrize('kind,unit',[('No','Object'),('Bayesian','Object'),('Bernoulli','Object'),('Bernoulli','Group')])
def test_public_generated_pair_snapshot_export_and_pfound_history(tmp_path,method,kind,unit):
    args,x,y,w,group,options,fit=setup(method,kind,unit)
    direct=CatBoostMetalRanker(**options).fit(x,y,**fit);snapshot=tmp_path/'pfound.snapshot'
    partial=CatBoostMetalRanker(**options).fit(x,y,**fit,save_snapshot=True,snapshot_file=snapshot,snapshot_interval=0,callback=lambda event:event.iteration<2)
    assert partial.tree_count_==2
    resumed=CatBoostMetalRanker(**options).fit(x,y,**fit,save_snapshot=True,snapshot_file=snapshot,snapshot_interval=0)
    np.testing.assert_array_equal(resumed.predict(x),direct.predict(x))
    for attr in ('get_leaf_values','get_leaf_weights'):np.testing.assert_array_equal(getattr(resumed._model,attr)(),getattr(direct._model,attr)())
    assert resumed.evals_result_==direct.evals_result_
    for i,value in enumerate(direct.evals_result_['learn']['PFound'],1):
        expected=_shared_metric('PFound',direct.predict(x,ntree_end=i),y,w,args['group_offsets'])
        assert value==pytest.approx(expected,abs=3e-7)
    assert direct.training_stats_['yeti_pair_rng']=='item_iteration_domains_v1'
    params=json.loads(direct._model.get_metadata()['params'])
    assert params['tree_learner_options']['bootstrap']['sampling_unit']==unit
    for fmt in ('cbm','json'):
        path=tmp_path/('pfound.'+fmt);resumed.save_model(path,format=fmt);loaded=CatBoostRanker().load_model(str(path),format=fmt)
        np.testing.assert_allclose(loaded.predict(x,task_type='GPU'),direct.predict(x),atol=1e-12,rtol=1e-12)
        np.testing.assert_array_equal(loaded.get_leaf_values().astype(np.float32),direct._model.get_leaf_values().astype(np.float32))
        np.testing.assert_array_equal(loaded.get_leaf_weights().astype(np.float32),direct._model.get_leaf_weights().astype(np.float32))


@pytest.mark.parametrize('extra',[{'sampling_unit':'Group'},{'loss_function':'YetiRankPairwise:permutations=8'},{'random_seed':789}])
def test_snapshot_rejects_changed_target_sampling_or_seed(tmp_path,extra):
    _,x,y,_,_,options,fit=setup();snap=tmp_path/'pfound.snapshot'
    CatBoostMetalRanker(**options).fit(x,y,**fit,save_snapshot=True,snapshot_file=snap,snapshot_interval=0,callback=lambda event:event.iteration<2)
    with pytest.raises(ValueError,match='(?i)(snapshot|fingerprint|configuration)'):
        CatBoostMetalRanker(**(options|extra)).fit(x,y,**fit,save_snapshot=True,snapshot_file=snap,snapshot_interval=0)


def test_pfound_best_model_uses_maximum_and_public_defaults():
    _,x,y,_,_,options,fit=setup();direct=CatBoostMetalRanker(**options).fit(x,y,**fit)
    best=CatBoostMetalRanker(**options).fit(x,y,**(fit|{'use_best_model':True}))
    selected=int(np.argmax(direct.evals_result_['validation']['PFound']))
    assert best.best_iteration_==selected and best.tree_count_==selected+1
    np.testing.assert_array_equal(best.predict(x),direct.predict(x,ntree_end=selected+1))
    default=CatBoostMetalRanker(loss_function='YetiRankPairwise')
    assert default.leaf_estimation_method=='Simple' and default.leaf_estimation_iterations==1 and default.l2_leaf_reg==0


@pytest.mark.parametrize('extra',[{'bootstrap_type':'Poisson'},{'bootstrap_type':'MVS'},{'sampling_unit':'Row'},
    {'leaf_estimation_backtracking':'AnyImprovement'},{'leaf_estimation_iterations':2},{'depth':0},{'depth':9}])
def test_public_rejects_unimplemented_or_invalid_combinations(extra):
    with pytest.raises(ValueError):CatBoostMetalRanker(loss_function='YetiRankPairwise',**extra)
