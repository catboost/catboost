"""Standalone full QueryCrossEntropy lifecycle and CUDA-target GPU metrics."""
import numpy as np
import pytest
from catboost import CatBoost,CatBoostRanker,Pool
from catboost_metal import CatBoostMetalRanker
from catboost_metal import _query_cross_entropy as qce
from test_query_cross_entropy_training import problem,leaf_walk
from test_pair_lifecycle import assert_same_model


@pytest.fixture(autouse=True)
def no_cpu_fit(monkeypatch):
    def forbidden(*args,**kw):raise AssertionError('No CPU CatBoost fitting')
    monkeypatch.setattr(CatBoost,'_fit',forbidden)


def params(**extra):
    return dict(loss_function='QueryCrossEntropy:alpha=.7;raw_values_scale=0,0:.8 4,2:1.6',iterations=5,depth=3,
        learning_rate=.17,l2_leaf_reg=1.,bayesian_matrix_reg=.2,border_count=3,bootstrap_type='No',random_strength=0.,
        score_function='NewtonL2',leaf_estimation_iterations=3,leaf_estimation_backtracking='No',random_seed=718)|extra


def inputs():
    args=problem();groups=np.repeat(np.arange(8),np.diff(args['group_offsets']))
    return args,args['bins'].T,args['targets'],groups,args['sample_weight']


def test_public_defaults_choose_cuda_query_matrix_options():
    model=CatBoostMetalRanker(loss_function='QueryCrossEntropy')
    assert model.leaf_estimation_method=='Newton' and model.leaf_estimation_iterations==10
    assert model.l2_leaf_reg==1 and model.bayesian_matrix_reg==pytest.approx(.1)
    assert model.bootstrap_type=='Bernoulli' and model.subsample==.5 and model.border_count==32
    assert model.random_strength==0 and model.leaf_estimation_backtracking=='AnyImprovement'


@pytest.mark.parametrize('kind',['No','Bernoulli'])
@pytest.mark.parametrize('mode',['No','AnyImprovement','Armijo'])
def test_public_snapshot_callback_and_replay_preserve_full_forest(tmp_path,kind,mode):
    args,x,y,groups,weights=inputs();options=params(bootstrap_type=kind,leaf_estimation_backtracking=mode)
    if kind=='Bernoulli':options['subsample']=.7
    fit=dict(group_id=groups,sample_weight=weights,eval_set=(x,y,groups,weights),use_best_model=False)
    path=tmp_path/'qce.snapshot'
    stopped=CatBoostMetalRanker(**options).fit(x,y,**fit,save_snapshot=True,snapshot_file=path,snapshot_interval=0,callback=lambda info:info.iteration<2)
    assert stopped.tree_count_==2
    resumed=CatBoostMetalRanker(**options).fit(x,y,**fit,save_snapshot=True,snapshot_file=path)
    direct=CatBoostMetalRanker(**options).fit(x,y,**fit)
    assert_same_model(resumed._result,direct._result)
    assert resumed.evals_result_==direct.evals_result_ and resumed.training_stats_['resumed_iterations']==2
    with np.load(path,allow_pickle=False) as archive:assert all(archive[n].dtype.kind!='O' for n in archive.files)
    for fmt in ['cbm','json']:
        output=tmp_path/('qce.'+fmt);resumed.save_model(output,format=fmt)
        read=CatBoostRanker().load_model(output,format=fmt)
        np.testing.assert_allclose(read.predict(x,task_type='GPU'),resumed.predict(x,task_type='METAL'),atol=2e-7)
        assert qce.parse_description(read.get_all_params()['loss_function'])==qce.parse_description(options['loss_function'])
        assert read.get_all_params()['bayesian_matrix_reg']==pytest.approx(.2)


@pytest.mark.parametrize('metric',['QueryCrossEntropy:alpha=.2','QueryCrossEntropy:alpha=.2;use_weights=false',
    'QueryCrossEntropy:alpha=.2;raw_values_scale=0,0:9','NDCG:top=5','PFound','Logloss'])
def test_selection_metric_matches_target_scales_weights_and_direction(metric):
    args,x,y,groups,weights=inputs()
    model=CatBoostMetalRanker(**params(eval_metric=metric)).fit(x,y,group_id=groups,sample_weight=weights,
        eval_set=(x,y,groups,weights),use_best_model=False)
    predictions=model.training_predictions_
    if metric.startswith('QueryCrossEntropy'):
        scales=qce.select_scales('0,0:.8 4,2:1.6',y,args['group_offsets'])
        expected=qce.metric(predictions,y,weights,args['group_offsets'],alpha=.2,query_scales=scales)
        assert model.evals_result_['validation'][metric][-1]==pytest.approx(expected,abs=2e-7)
    assert model.training_stats_['metric_maximized']==(metric.startswith('NDCG') or metric=='PFound')


def test_pool_and_array_group_weight_products_match():
    args,x,y,groups,weights=inputs();gw=np.linspace(.3,2,8).astype(np.float32)[groups]
    split=CatBoostMetalRanker(**params()).fit(x,y,group_id=groups,sample_weight=weights,group_weight=gw)
    pool=Pool(x,y,group_id=groups,weight=np.float32(weights*gw))
    combined=CatBoostMetalRanker(**params()).fit(pool)
    assert_same_model(split._result,combined._result)
    for row in split._result.leaf_weights:assert row.sum()==pytest.approx(np.float32(weights*gw).sum(),rel=3e-6)


@pytest.mark.parametrize('changed',['alpha','raw_values_scale','sample_weight','bayesian_matrix_reg'])
def test_snapshot_rejects_changed_matrix_target_before_training(tmp_path,monkeypatch,changed):
    args,x,y,groups,weights=inputs();fit=dict(group_id=groups,sample_weight=weights,save_snapshot=True,snapshot_file=tmp_path/'identity.snapshot')
    CatBoostMetalRanker(**params(iterations=2)).fit(x,y,**fit);options=params()
    if changed=='alpha':options['loss_function']='QueryCrossEntropy:alpha=.5;raw_values_scale=0,0:.8 4,2:1.6'
    elif changed=='raw_values_scale':options['loss_function']='QueryCrossEntropy:alpha=.7;raw_values_scale=0,0:1.8'
    elif changed=='sample_weight':fit['sample_weight']=weights*2
    else:options['bayesian_matrix_reg']=.7
    def forbidden(*a,**kw):raise AssertionError('Mismatched snapshot reached training session')
    monkeypatch.setattr(qce,'TrainingSession',forbidden)
    with pytest.raises(ValueError,match='does not match'):CatBoostMetalRanker(**options).fit(x,y,**fit)


def test_reversed_validation_targets_stop_and_trim_but_keep_recovery_forest(tmp_path):
    x=np.array([[0],[1]]*12,np.float32);y=np.tile([0,1],12);groups=np.repeat(np.arange(12),2)
    options=params(iterations=12,depth=1,border_count=1,learning_rate=.4);path=tmp_path/'stopped.snapshot'
    model=CatBoostMetalRanker(**options).fit(x,y,group_id=groups,eval_set=(x,1-y,groups),early_stopping_rounds=2,
        use_best_model=True,save_snapshot=True,snapshot_file=path)
    assert model.best_iteration_==0 and model.tree_count_==1 and model.training_stats_['iterations_trained']==3
    with np.load(path,allow_pickle=False) as archive:assert len(archive['depths'])==3
    first=CatBoostMetalRanker(**(options|{'iterations':1})).fit(x,y,group_id=groups)
    np.testing.assert_array_equal(model.training_predictions_,first.training_predictions_)


@pytest.mark.parametrize('extra',[
    {'depth':9},{'bootstrap_type':'MVS'},{'bootstrap_type':'Bayesian'},{'score_function':'L2'},
    {'leaf_estimation_method':'Gradient'},{'leaf_estimation_method':'Exact'},{'bayesian_matrix_reg':-1},
    {'boosting_type':'Ordered'},{'grow_policy':'Lossguide'},{'loss_function':'QueryCrossEntropy:alpha=1.2'},
    {'loss_function':'QueryCrossEntropy:raw_values_scale=0,0:nan'},
])
def test_invalid_or_unconnected_public_options_fail_before_fit(extra):
    with pytest.raises(ValueError):CatBoostMetalRanker(**params(**extra))
