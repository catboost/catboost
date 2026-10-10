"""Standalone full-matrix ranker metadata, metrics and exact recoverable forests."""
import json
import numpy as np
import pytest
from catboost import CatBoost,CatBoostRanker,Pool
from catboost_metal import CatBoostMetalRanker,_pair_matrix
from catboost_metal._training import _shared_metric,run_training
from test_pair_lifecycle import problem,assert_same_model,pair_options
from test_ranking_metrics import expected_metric


@pytest.fixture(autouse=True)
def no_cpu_fit(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('No CPU CatBoost training')
    monkeypatch.setattr(CatBoost,'_fit',forbidden)


def params(**extra):
    return dict(loss_function='PairLogitPairwise',iterations=5,depth=3,learning_rate=.17,
        l2_leaf_reg=3.,bayesian_matrix_reg=.2,border_count=3,bootstrap_type='No',random_strength=0.,
        score_function='NewtonL2',leaf_estimation_iterations=3,leaf_estimation_backtracking='No',random_seed=718)|extra


def inputs():
    bins,y,features,borders,offsets,groups,pairs,edges=problem()
    y=np.linspace(0,1,len(y)).astype(np.float32);objects=np.linspace(.2,2,len(y)).astype(np.float32)
    return bins,y,features,borders,offsets,groups,pairs,edges,objects


@pytest.mark.parametrize('method,iterations',[('Newton',1),('Gradient',5)])
def test_public_defaults_match_cuda_pairwise_family(method,iterations):
    model=CatBoostMetalRanker(loss_function='PairLogitPairwise',leaf_estimation_method=method)
    assert model.leaf_estimation_iterations==iterations
    assert model.l2_leaf_reg==5 and model.bayesian_matrix_reg==pytest.approx(.1)
    assert model.bootstrap_type=='Bernoulli' and model.subsample==.5 and model.border_count==32
    assert model.random_strength==0 and model.leaf_estimation_backtracking=='AnyImprovement'


@pytest.mark.parametrize('kind',['No','Bayesian','Bernoulli','Poisson'])
@pytest.mark.parametrize('mode',['No','AnyImprovement','Armijo'])
def test_public_snapshot_callback_and_replay_are_exact(tmp_path,kind,mode):
    bins,y,_,_,_,groups,pairs,edges,objects=inputs();x=bins.T
    options=params(bootstrap_type=kind,leaf_estimation_backtracking=mode)
    if kind=='Bayesian':options['bagging_temperature']=.7
    elif kind!='No':options['subsample']=.7
    fit=dict(group_id=groups,sample_weight=objects,pairs=pairs,pairs_weight=edges,
        eval_set=(x,y,groups,objects),eval_pairs=pairs,eval_pairs_weight=edges,use_best_model=False)
    path=tmp_path/'pair-matrix.snapshot'
    stopped=CatBoostMetalRanker(**options).fit(x,y,**fit,save_snapshot=True,snapshot_file=path,
        snapshot_interval=0,callback=lambda info:info.iteration<2)
    assert stopped.tree_count_==2
    resumed=CatBoostMetalRanker(**options).fit(x,y,**fit,save_snapshot=True,snapshot_file=path)
    direct=CatBoostMetalRanker(**options).fit(x,y,**fit)
    assert_same_model(resumed._result,direct._result)
    assert resumed.evals_result_==direct.evals_result_ and resumed.training_stats_['resumed_iterations']==2
    with np.load(path,allow_pickle=False) as archive:
        assert all(archive[name].dtype.kind!='O' for name in archive.files)
    for fmt in ['cbm','json']:
        out=tmp_path/('pair-matrix.'+fmt);resumed.save_model(out,format=fmt)
        model=CatBoostRanker().load_model(out,format=fmt)
        np.testing.assert_allclose(model.predict(x,task_type='GPU'),resumed.predict(x,task_type='METAL'),atol=2e-7)
        assert model.get_all_params()['loss_function']=='PairLogitPairwise'
        assert model.get_all_params()['bayesian_matrix_reg']==pytest.approx(.2)


@pytest.mark.parametrize('selection',['PairLogit','PairLogit:use_weights=false','PairAccuracy',
    'NDCG:top=5','NDCG:top=5;use_weights=false','PFound','PFound:use_weights=false','MAP:top=5;border=.5'])
def test_pair_metrics_and_ranking_weights_use_the_correct_target_mass(selection):
    bins,y,_,_,offsets,groups,pairs,edges,objects=inputs();x=bins.T
    model=CatBoostMetalRanker(**params(eval_metric=selection)).fit(x,y,group_id=groups,sample_weight=objects,
        pairs=pairs,pairs_weight=edges,eval_set=(x,y,groups,objects),eval_pairs=pairs,eval_pairs_weight=edges,use_best_model=False)
    raw=model.predict(x,task_type='GPU')
    expected=_shared_metric(selection,raw,y,objects,offsets,(pairs[:,0],pairs[:,1],edges),pair_query_unit_weights=False)
    if selection.partition(':')[0] in ('NDCG','PFound','MAP'):
        independent=expected_metric(raw,y,offsets,objects,selection)
    else:
        difference=raw[pairs[:,0]]-raw[pairs[:,1]]
        terms=difference>0 if selection=='PairAccuracy' else np.logaddexp(0.,-difference)
        independent=np.average(terms,weights=np.ones(len(edges)) if 'use_weights=false' in selection else edges)
    assert expected==pytest.approx(independent,abs=1e-12)
    assert model.evals_result_['validation'][selection][-1]==pytest.approx(expected,abs=2e-7)
    assert model.training_stats_['metric_maximized']==not_minimized(selection)
    # This catches accidentally reusing ordinary PairLogit's unit query weights.
    if selection.partition(':')[0] in ('NDCG','PFound') and 'use_weights=false' not in selection:
        unit=_shared_metric(selection,raw,y,objects,offsets,(pairs[:,0],pairs[:,1],edges))
        assert abs(expected-unit)>1e-4


def not_minimized(name):return name.partition(':')[0]!='PairLogit'


def test_public_query_weight_product_once_and_original_leaf_metadata():
    bins,y,_,_,offsets,groups,pairs,edges,objects=inputs();x=bins.T
    gw=np.array([.4,1.3,2.1],np.float32)[groups]
    fit=dict(group_id=groups,pairs=pairs,pairs_weight=edges)
    separate=CatBoostMetalRanker(**params()).fit(x,y,sample_weight=objects,group_weight=gw,**fit)
    combined=CatBoostMetalRanker(**params()).fit(x,y,sample_weight=np.float32(objects*gw),**fit)
    assert_same_model(separate._result,combined._result)
    for row in separate._result.leaf_weights:assert row.sum()==pytest.approx(np.float32(objects*gw).sum(),rel=3e-6)


def test_stored_pool_pairs_and_weights_match_array_entrypoint():
    bins,y,_,_,_,groups,pairs,edges,objects=inputs();x=bins.T
    pool=Pool(x,y,group_id=groups,pairs=pairs,pairs_weight=edges,weight=objects)
    a=CatBoostMetalRanker(**params()).fit(pool)
    b=CatBoostMetalRanker(**params()).fit(x,y,group_id=groups,pairs=pairs,pairs_weight=edges,sample_weight=objects)
    assert_same_model(a._result,b._result)


@pytest.mark.parametrize('changed',['sample_weight','pairs_weight','bayesian_matrix_reg','pairs_order'])
def test_snapshot_rejects_changed_matrix_target_before_gpu(tmp_path,monkeypatch,changed):
    bins,y,_,_,_,groups,pairs,edges,objects=inputs();x=bins.T
    fit=dict(group_id=groups,pairs=pairs,pairs_weight=edges,sample_weight=objects,
        save_snapshot=True,snapshot_file=tmp_path/'identity.snapshot')
    CatBoostMetalRanker(**params(iterations=2)).fit(x,y,**fit)
    options=params()
    if changed=='bayesian_matrix_reg':options[changed]=.7
    elif changed=='pairs_order':fit['pairs']=pairs[::-1];fit['pairs_weight']=edges[::-1]
    else:fit[changed]=fit[changed]*2
    def forbidden(*a,**kw):raise AssertionError('Changed snapshot reached GPU')
    monkeypatch.setattr(_pair_matrix,'TrainingSession',forbidden)
    with pytest.raises(ValueError,match='does not match'):CatBoostMetalRanker(**options).fit(x,y,**fit)


def test_reversed_validation_edges_trim_model_but_preserve_resume_forest(tmp_path):
    x=np.array([[0],[1],[0],[1],[0],[1]],np.float32);groups=np.repeat([0,1,2],2)
    pairs=np.array([[1,0],[3,2],[5,4]],np.uint32);path=tmp_path/'stopped.snapshot'
    options=params(iterations=10,depth=1,border_count=1,learning_rate=.4)
    model=CatBoostMetalRanker(**options).fit(x,group_id=groups,pairs=pairs,
        eval_set=(x,np.zeros(6),groups),eval_pairs=pairs[:,::-1],early_stopping_rounds=2,
        use_best_model=True,save_snapshot=True,snapshot_file=path)
    assert model.best_iteration_==0 and model.tree_count_==1 and model.training_stats_['iterations_trained']==3
    with np.load(path,allow_pickle=False) as archive:assert len(archive['depths'])==3
    first=CatBoostMetalRanker(**(options|{'iterations':1})).fit(x,group_id=groups,pairs=pairs)
    np.testing.assert_array_equal(model.training_predictions_,first.training_predictions_)


@pytest.mark.parametrize('options',[{'depth':9},{'bootstrap_type':'MVS'},{'leaf_estimation_method':'Exact'},
    {'bayesian_matrix_reg':-1},{'bayesian_matrix_reg':float('nan')},{'bayesian_matrix_reg':True},
    {'boosting_type':'Ordered'},{'grow_policy':'Lossguide'}])
def test_invalid_or_unconnected_public_configuration_fails_before_fit(options):
    with pytest.raises(ValueError):CatBoostMetalRanker(**params(**options))
