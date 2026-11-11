"""Standard CatBoost PairLogitPairwise GPU fitting, metrics, export and recovery."""
import os
import numpy as np
import pytest
from catboost import CatBoost,CatBoostRanker,CatBoostError,Pool
from catboost_metal import _pair_matrix
from catboost_metal._training import _shared_metric
from test_native_pair_api import data,StopAfter
from test_pairwise_matrix_training import problem

pytestmark=pytest.mark.skipif(os.environ.get('CATBOOST_NATIVE_METAL_QUERY_TESTS')!='1',reason='rebuilt native Metal adapter required')


def params(**extra):
    values=dict(task_type='GPU',loss_function='PairLogitPairwise',iterations=6,depth=3,learning_rate=.17,
        l2_leaf_reg=3.,border_count=16,random_seed=718,bootstrap_type='No',random_strength=0.,
        score_function='NewtonL2',leaf_estimation_iterations=3,leaf_estimation_backtracking='No',
        has_time=True,verbose=False,allow_writing_files=False,metric_period=1)
    values.update(extra);return values


@pytest.mark.parametrize('score',['L2','Cosine','NewtonL2','NewtonCosine','SolarL2','LOOL2','SatL2'])
@pytest.mark.parametrize('method',['Newton','Gradient'])
def test_native_full_forest_matches_resident_oracle_and_original_document_weights(tmp_path,score,method):
    args=problem(score_function=score,leaf_estimation_method=method,candidate_types=np.zeros(9,np.uint8),non_diagonal_regularization=.1)
    x=args['bins'].T.astype(np.float32);edges=np.column_stack((args['pair_winners'],args['pair_losers']))
    pool=Pool(x,np.linspace(0,1,len(x)),group_id=np.zeros(len(x),int),pairs=edges,pairs_weight=args['pair_weights'],
        baseline=args['initial_predictions'],weight=args['sample_weight'])
    borders=tmp_path/'borders.tsv';borders.write_text(''.join(f'{f}\t{b+.5}\n' for f in range(3) for b in range(3)))
    pool.quantize(input_borders=str(borders))
    model=CatBoostRanker(**params(iterations=3,score_function=score,leaf_estimation_method=method)).fit(pool)
    with _pair_matrix.Session(**args) as session:
        trees=[session.step() for _ in range(3)];expected=session.predictions()
    np.testing.assert_array_equal(model.get_tree_leaf_counts(),[1<<t.depth for t in trees])
    np.testing.assert_allclose(model.get_leaf_values(),np.concatenate([t.leaf_values for t in trees]),rtol=3e-5,atol=3e-7)
    np.testing.assert_allclose(model.get_leaf_weights(),np.concatenate([t.leaf_weights for t in trees]),rtol=3e-6,atol=3e-6)
    np.testing.assert_allclose(model.predict(x,task_type='GPU')+args['initial_predictions'],expected,rtol=3e-6,atol=3e-7)
    assert model.get_all_params()['task_type']=='GPU'
    assert model.get_metadata()['metal_backend']=='METAL'


@pytest.mark.parametrize('mode',['No','AnyImprovement','Armijo'])
@pytest.mark.parametrize('kind',['No','Bayesian','Bernoulli','Poisson'])
def test_native_snapshots_preserve_pairwise_forest_metrics_and_weights(tmp_path,mode,kind):
    x,y,groups,edges,ew=data();weights=np.linspace(.2,2,len(x)).astype(np.float32)
    pool=Pool(x,y,group_id=groups,pairs=edges,pairs_weight=ew,weight=weights,baseline=np.linspace(-.4,.4,len(x)))
    options=params(leaf_estimation_backtracking=mode,bootstrap_type=kind,leaf_estimation_iterations=5)
    if kind=='Bayesian':options['bagging_temperature']=.7
    elif kind!='No':options['subsample']=.7
    direct=CatBoostRanker(**options).fit(pool,eval_set=pool,use_best_model=False)
    saved=dict(options,save_snapshot=True,snapshot_interval=0,snapshot_file='pair-matrix.snapshot',
        allow_writing_files=True,train_dir=str(tmp_path))
    assert CatBoostRanker(**saved).fit(pool,eval_set=pool,use_best_model=False,callbacks=[StopAfter()]).tree_count_==3
    restored=CatBoostRanker(**saved).fit(pool,eval_set=pool,use_best_model=False)
    for get in ('get_leaf_values','get_leaf_weights','get_test_eval'):
        np.testing.assert_array_equal(getattr(restored,get)(),getattr(direct,get)())
    assert restored.evals_result_==direct.evals_result_
    assert restored.evals_result_['learn']['PairLogit'][-1]<restored.evals_result_['learn']['PairLogit'][0]
    changed=ew.copy();changed[0]*=2
    with pytest.raises(CatBoostError,match='(?i)snapshot.*differ'):
        CatBoostRanker(**saved).fit(Pool(x,y,group_id=groups,pairs=edges,pairs_weight=changed,weight=weights,
            baseline=np.linspace(-.4,.4,len(x))),eval_set=pool,use_best_model=False)


@pytest.mark.parametrize('metric',['PairLogit','NDCG:top=5','MAP:top=5;border=0.5','PFound'])
def test_native_metrics_best_model_and_standard_gpu_roundtrip(tmp_path,metric):
    x,_,groups,edges,ew=data();y=np.linspace(0,1,len(x));weights=np.linspace(.3,2,len(x)).astype(np.float32)
    pool=Pool(x,y,group_id=groups,pairs=edges,pairs_weight=ew,weight=weights)
    model=CatBoostRanker(**params(eval_metric=metric,custom_metric=['PairAccuracy'])).fit(pool,eval_set=pool,use_best_model=True)
    key=next(k for k in model.evals_result_['validation'] if k.partition(':')[0]==metric.partition(':')[0])
    history=model.evals_result_['validation'][key];best=int(np.argmin(history) if metric=='PairLogit' else np.argmax(history))
    assert model.best_iteration_==best and model.tree_count_==best+1
    raw=np.asarray(model.get_test_eval())
    if metric=='PairLogit':expected=np.average(np.logaddexp(0.,raw[edges[:,1]]-raw[edges[:,0]]),weights=ew)
    else:expected=_shared_metric(metric,raw,y,weights,np.arange(0,len(x)+1,6,dtype=np.uint32))
    assert history[best]==pytest.approx(expected,abs=2e-8)
    for fmt in ['cbm','json']:
        path=tmp_path/('pair-matrix.'+fmt);model.save_model(path,format=fmt)
        restored=CatBoostRanker().load_model(path,format=fmt)
        np.testing.assert_allclose(restored.predict(x,task_type='GPU'),model.predict(x),atol=2e-7)


def test_native_generated_pairs_and_unlabeled_supplied_pairs():
    x,y,groups,edges,ew=data()
    for pool in [Pool(x,y,group_id=groups),Pool(x,group_id=groups,pairs=edges,pairs_weight=ew)]:
        model=CatBoostRanker(**params()).fit(pool)
        assert model.tree_count_==6 and np.isfinite(model.predict(x,task_type='GPU')).all()


def test_native_document_group_weights_combined_once_and_never_replace_original_pair_weights():
    x,y,groups,edges,ew=data();objects=np.linspace(.2,2,len(x)).astype(np.float32)
    gw=np.linspace(.4,2,8).astype(np.float32)[groups]
    split=Pool(x,y,group_id=groups,pairs=edges,pairs_weight=ew,group_weight=gw);split.set_weight(objects)
    combined=Pool(x,y,group_id=groups,pairs=edges,pairs_weight=ew,weight=np.float32(objects*gw))
    a=CatBoostRanker(**params()).fit(split);b=CatBoostRanker(**params()).fit(combined)
    np.testing.assert_array_equal(a.get_leaf_values(),b.get_leaf_values())
    np.testing.assert_array_equal(a.get_leaf_weights(),b.get_leaf_weights())
    for row in a.get_leaf_weights().reshape(-1,8):assert row.sum()==pytest.approx(np.float32(objects*gw).sum(),rel=3e-6)


@pytest.mark.parametrize('mode',['No','AnyImprovement','Armijo'])
def test_native_initial_model_baseline_resume(tmp_path,mode):
    x,y,groups,edges,ew=data();pool=Pool(x,y,group_id=groups,pairs=edges,pairs_weight=ew)
    initial=CatBoostRanker(**params(iterations=2)).fit(pool)
    pool.set_baseline(np.linspace(-.2,.2,len(x)))
    options=params(iterations=5,leaf_estimation_backtracking=mode)
    direct=CatBoostRanker(**options).fit(pool,init_model=initial,eval_set=pool,use_best_model=False)
    saved=dict(options,save_snapshot=True,snapshot_interval=0,snapshot_file='initial.snapshot',
        allow_writing_files=True,train_dir=str(tmp_path))
    CatBoostRanker(**saved).fit(pool,init_model=initial,eval_set=pool,use_best_model=False,callbacks=[StopAfter()])
    restored=CatBoostRanker(**saved).fit(pool,init_model=initial,eval_set=pool,use_best_model=False)
    assert restored.tree_count_==7
    np.testing.assert_array_equal(restored.get_leaf_values(),direct.get_leaf_values())
    np.testing.assert_array_equal(restored.get_test_eval(),direct.get_test_eval())


def test_native_defaults_select_gpu_newton_pairwise_path():
    x,y,groups,edges,ew=data();pool=Pool(x,y,group_id=groups,pairs=edges,pairs_weight=ew)
    model=CatBoostRanker(task_type='GPU',loss_function='PairLogitPairwise',iterations=3,depth=2,
        verbose=False,allow_writing_files=False).fit(pool)
    options=model.get_all_params()
    assert options['leaf_estimation_method']=='Newton' and options['l2_leaf_reg']==5
    assert options['random_strength']==0 and options['bootstrap_type']=='Bernoulli'
    assert model.tree_count_==3
