"""Native QueryCrossEntropy: scaled GPU metric, full forests and recovery."""
import os
import numpy as np
import pytest
from catboost import CatBoostRanker,CatBoostError,Pool
from catboost_metal import _query_cross_entropy as qce
from test_query_cross_entropy_training import problem,target
from test_native_pair_api import StopAfter

pytestmark=pytest.mark.skipif(os.environ.get('CATBOOST_NATIVE_METAL_QUERY_TESTS')!='1',reason='rebuilt native Metal adapter required')


def params(**extra):
    return dict(task_type='GPU',loss_function='QueryCrossEntropy:alpha=.7;raw_values_scale=0,0:.8 4,2:1.6',
        iterations=5,depth=3,learning_rate=.17,l2_leaf_reg=1.,border_count=16,random_seed=718,
        bootstrap_type='No',random_strength=0.,score_function='NewtonL2',leaf_estimation_iterations=3,
        leaf_estimation_backtracking='No',has_time=True,verbose=False,allow_writing_files=False,metric_period=1)|extra


def inputs(**extra):
    args=problem(alpha=.7,candidate_types=np.zeros(9,np.uint8),non_diagonal_regularization=.1,**extra)
    x=args['bins'].T.astype(np.float32);groups=np.repeat(np.arange(len(args['group_offsets'])-1),np.diff(args['group_offsets']))
    args['query_scales']=qce.select_scales('0,0:.8 4,2:1.6',args['targets'],args['group_offsets'])
    pool=Pool(x,args['targets'],group_id=groups,weight=args['sample_weight'],baseline=args['initial_predictions'])
    return args,x,groups,pool


@pytest.mark.parametrize('score',['Cosine','NewtonL2','NewtonCosine','SolarL2','LOOL2','SatL2'])
@pytest.mark.parametrize('kind',['No','Bernoulli'])
def test_native_full_forest_matches_resident_target_and_gpu_metric(tmp_path,score,kind):
    args,x,groups,pool=inputs(score_function=score,bootstrap_type=kind)
    borders=tmp_path/'borders.tsv';borders.write_text(''.join(f'{f}\t{b+.5}\n' for f in range(3) for b in range(3)))
    pool.quantize(input_borders=str(borders))
    options=params(iterations=3,score_function=score,bootstrap_type=kind)
    if kind=='Bernoulli':options['subsample']=.7
    model=CatBoostRanker(**options).fit(pool,eval_set=pool,use_best_model=False)
    with qce.Session(**args) as session:
        trees=[session.step() for _ in range(3)];expected=session.predictions()
    np.testing.assert_allclose(model.get_leaf_values(),np.concatenate([t.leaf_values for t in trees]),rtol=3e-5,atol=3e-7)
    np.testing.assert_allclose(model.get_leaf_weights(),np.concatenate([t.leaf_weights for t in trees]),rtol=3e-6,atol=3e-6)
    np.testing.assert_allclose(model.predict(x,task_type='GPU')+args['initial_predictions'],expected,rtol=3e-6,atol=3e-7)
    for history in model.evals_result_.values():
        np.testing.assert_allclose(next(iter(history.values())),[t.loss for t in trees],rtol=3e-6,atol=3e-7)
    assert model.get_metadata()['metal_backend']=='METAL'


@pytest.mark.parametrize('kind',['No','Bernoulli'])
@pytest.mark.parametrize('mode',['No','AnyImprovement','Armijo'])
def test_native_snapshot_is_exact_with_scaled_loss_and_original_document_weights(tmp_path,kind,mode):
    args,x,groups,pool=inputs();options=params(bootstrap_type=kind,leaf_estimation_backtracking=mode)
    if kind=='Bernoulli':options['subsample']=.7
    direct=CatBoostRanker(**options).fit(pool,eval_set=pool,use_best_model=False)
    saved=options|dict(save_snapshot=True,snapshot_interval=0,snapshot_file='qce.snapshot',allow_writing_files=True,train_dir=str(tmp_path))
    assert CatBoostRanker(**saved).fit(pool,eval_set=pool,use_best_model=False,callbacks=[StopAfter()]).tree_count_==3
    restored=CatBoostRanker(**saved).fit(pool,eval_set=pool,use_best_model=False)
    np.testing.assert_array_equal(restored.get_leaf_values(),direct.get_leaf_values())
    np.testing.assert_array_equal(restored.get_leaf_weights(),direct.get_leaf_weights())
    np.testing.assert_array_equal(restored.get_test_eval(),direct.get_test_eval())
    assert restored.evals_result_==direct.evals_result_
    with pytest.raises(CatBoostError,match='(?i)snapshot.*differ'):
        CatBoostRanker(**(saved|{'loss_function':'QueryCrossEntropy:alpha=.7;raw_values_scale=0,0:1.8'})).fit(pool,eval_set=pool,use_best_model=False)


@pytest.mark.parametrize('metric',['QueryCrossEntropy:alpha=.2','QueryCrossEntropy:alpha=.2;use_weights=false',
    'QueryCrossEntropy:alpha=.2;raw_values_scale=0,0:9','NDCG:top=5','PFound'])
def test_native_best_model_and_scaled_metric_parameters_roundtrip(tmp_path,metric):
    args,x,groups,pool=inputs();model=CatBoostRanker(**params(eval_metric=metric)).fit(pool,eval_set=pool,use_best_model=True)
    key=metric if metric in model.evals_result_['validation'] else next(k for k in model.evals_result_['validation'] if k.partition(':')[0]==metric.partition(':')[0])
    history=model.evals_result_['validation'][key];best=int(np.argmin(history) if metric.startswith('QueryCrossEntropy') else np.argmax(history))
    assert model.best_iteration_==best and model.tree_count_==best+1
    if metric.startswith('QueryCrossEntropy'):
        # CUDA target fallback intentionally retains ORIGINAL target weights and
        # scales, even if use_weights/raw_values_scale differ in the metric.
        expected=qce.metric(model.get_test_eval(),args['targets'],args['sample_weight'],args['group_offsets'],alpha=.2,query_scales=args['query_scales'])
        assert history[best]==pytest.approx(expected,abs=1e-12)
    for fmt in ['cbm','json']:
        path=tmp_path/('qce.'+fmt);model.save_model(path,format=fmt)
        restored=CatBoostRanker().load_model(path,format=fmt)
        np.testing.assert_allclose(restored.predict(x,task_type='GPU'),model.predict(x),atol=2e-7)


@pytest.mark.parametrize('mode',['No','AnyImprovement','Armijo'])
def test_native_initial_model_baseline_snapshot_resume(tmp_path,mode):
    args,x,groups,pool=inputs();initial=CatBoostRanker(**params(iterations=2)).fit(pool)
    options=params(leaf_estimation_backtracking=mode)
    direct=CatBoostRanker(**options).fit(pool,init_model=initial,eval_set=pool,use_best_model=False)
    saved=options|dict(save_snapshot=True,snapshot_interval=0,snapshot_file='initial.snapshot',allow_writing_files=True,train_dir=str(tmp_path))
    CatBoostRanker(**saved).fit(pool,init_model=initial,eval_set=pool,use_best_model=False,callbacks=[StopAfter()])
    restored=CatBoostRanker(**saved).fit(pool,init_model=initial,eval_set=pool,use_best_model=False)
    assert restored.tree_count_==7
    np.testing.assert_array_equal(restored.get_leaf_values(),direct.get_leaf_values())
    np.testing.assert_array_equal(restored.get_test_eval(),direct.get_test_eval())


def test_native_group_times_document_weight_is_applied_once():
    args,x,groups,_=inputs();y=args['targets'];objects=args['sample_weight'];gw=np.linspace(.4,2,8).astype(np.float32)[groups]
    split=Pool(x,y,group_id=groups,group_weight=gw);split.set_weight(objects)
    combined=Pool(x,y,group_id=groups,weight=np.float32(objects*gw))
    a=CatBoostRanker(**params()).fit(split);b=CatBoostRanker(**params()).fit(combined)
    np.testing.assert_array_equal(a.get_leaf_values(),b.get_leaf_values())
    np.testing.assert_array_equal(a.get_leaf_weights(),b.get_leaf_weights())


@pytest.mark.parametrize('description',['0,0:.6 0,0:2','0,0:.6 4,2:1 4,2:2','4294967295,0:2 0,0:.6','256,256:2'])
def test_native_scale_table_defaults_duplicate_entries_and_large_unused_sizes(description):
    args,x,groups,pool=inputs();options=params(loss_function='QueryCrossEntropy:alpha=.7;raw_values_scale='+description)
    model=CatBoostRanker(**options).fit(pool,eval_set=pool,use_best_model=False)
    scales=qce.select_scales(description,args['targets'],args['group_offsets'])
    expected=qce.metric(model.get_test_eval(),args['targets'],args['sample_weight'],args['group_offsets'],alpha=.7,query_scales=scales)
    assert next(iter(model.evals_result_['validation'].values()))[-1]==pytest.approx(expected,abs=1e-12)


@pytest.mark.parametrize('extra',[
    {'depth':9},{'score_function':'L2'},{'bootstrap_type':'Bayesian'},{'bootstrap_type':'Poisson','subsample':.7},
    {'bootstrap_type':'MVS'},{'leaf_estimation_method':'Gradient'},{'leaf_estimation_method':'Exact'},
    {'grow_policy':'Lossguide'},{'loss_function':'QueryCrossEntropy:raw_values_scale=0,0:nan'},
])
def test_native_unsupported_or_invalid_configuration_is_rejected(extra):
    _,_,_,pool=inputs()
    with pytest.raises(CatBoostError):CatBoostRanker(**params(**extra)).fit(pool)


def test_native_query_size_limit_and_default_newton_configuration():
    args,x,groups,pool=inputs()
    model=CatBoostRanker(task_type='GPU',loss_function='QueryCrossEntropy',iterations=3,depth=2,
        verbose=False,allow_writing_files=False).fit(pool)
    p=model.get_all_params()
    assert p['leaf_estimation_method']=='Newton' and p['l2_leaf_reg']==1 and p['random_strength']==0
    assert p['bootstrap_type']=='Bernoulli' and p['border_count']==32
    # Shared CUDA data defaults reduce leaf iterations for a short, narrow run.
    assert p['leaf_estimation_iterations']==1
    with pytest.raises(CatBoostError,match='256'):
        CatBoostRanker(**params()).fit(Pool(np.arange(257)[:,None],np.arange(257)%2,group_id=np.zeros(257,int)))
