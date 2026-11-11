"""Native grouped scalar Ordered, category routing and weighted lifecycle."""
import os
import numpy as np
import pytest
from catboost import CatBoost,CatBoostRegressor,CatBoostClassifier,CatBoostError,Pool
from test_native_ordered_onehot import problem,resident,assert_readers,only_gpu,StopAfter,OBJECTIVES

pytestmark=pytest.mark.skipif(os.environ.get('CATBOOST_NATIVE_METAL_TESTS')!='1',reason='requires native grouped Ordered build')
SIZES=[1,3,37,2,61,7,5,76]


def grouped_problem(loss='RMSE',method='Newton',kind='No',count=4):
    cls,x,y,w,config,numeric=problem(loss,method,kind,count)
    groups=np.repeat(np.arange(len(SIZES),dtype=np.uint64)+(1<<40),SIZES)
    group_weights=np.repeat(np.linspace(.5,1.75,len(SIZES),dtype=np.float32),SIZES)
    return cls,x,y,w,config,numeric,groups,group_weights


def pool(x,y,w,groups,group_weights,cats=(1,3),baseline=None):
    result=Pool(x,y,cat_features=list(cats),group_id=groups,group_weight=group_weights,baseline=baseline)
    result.set_weight(w)
    return result


@pytest.mark.parametrize('loss,method',[*OBJECTIVES,('Lq:q=2.5','Newton')])
@pytest.mark.parametrize('score',['Cosine','NewtonCosine'])
def test_group_and_object_weights_match_private_prefix_runtime(tmp_path,loss,method,score):
    cls,x,y,w,config,numeric,groups,gw=grouped_problem(loss,method,count=1)
    config.update(has_time=True,score_function=score,fold_len_multiplier=1.3)
    baseline=np.linspace(-.1,.1,len(y),dtype=np.float32)
    learn=pool(x,y,w,groups,gw,baseline=baseline)
    borders=tmp_path/'borders.tsv';borders.write_text(''.join(f'{f}\t{b+.5}\n' for f in (0,2) for b in range(3)))
    learn.quantize(input_borders=str(borders))
    model=cls().set_params(**config).fit(learn,eval_set=learn,use_best_model=False)
    expected=resident(config,x,y,w*gw,numeric,baseline,SIZES)
    np.testing.assert_array_equal(model.get_tree_leaf_counts(),2**expected.depths)
    np.testing.assert_allclose(model.get_test_eval(),expected.predictions,atol=8e-7,rtol=6e-6)
    values=np.concatenate([expected.leaf_values[i,:1<<int(d)] for i,d in enumerate(expected.depths)])
    np.testing.assert_allclose(model.get_leaf_values(),values,atol=8e-7,rtol=6e-6)
    if expected.split_types.any():assert_readers(model,cls,x,tmp_path)


@pytest.mark.parametrize('kind',['No','Bayesian','Bernoulli','Poisson','MVS'])
@pytest.mark.parametrize('count',[1,4,7])
@pytest.mark.parametrize('quantized',[False,True])
def test_group_boundaries_weights_and_hashes_survive_exact_recovery(tmp_path,kind,count,quantized):
    cls,x,y,w,config,_,groups,gw=grouped_problem(kind=kind,count=count)
    config.update(iterations=6,random_strength=.7)
    gw=np.ones_like(gw)  # Isolate group-boundary identity when the snapshot is changed.
    learn=pool(x,y,w,groups,gw)
    if quantized:learn.quantize()
    heldout=x.copy();heldout[::17,1]='unseen';evaluation=pool(heldout,y,w,groups,gw)
    direct=cls().set_params(**config).fit(learn,eval_set=evaluation,use_best_model=False)
    saved=config|dict(save_snapshot=True,snapshot_file='groups.snapshot',snapshot_interval=0,
                      allow_writing_files=True,train_dir=str(tmp_path))
    assert cls().set_params(**saved).fit(learn,eval_set=evaluation,use_best_model=False,callbacks=[StopAfter(2)]).tree_count_==2
    resumed=cls().set_params(**saved).fit(learn,eval_set=evaluation,use_best_model=False)
    for key in ('get_leaf_values','get_leaf_weights','get_test_eval'):
        np.testing.assert_array_equal(getattr(resumed,key)(),getattr(direct,key)())
    assert resumed.evals_result_==direct.evals_result_
    assert_readers(resumed,cls,heldout,tmp_path)
    changed=groups.copy();changed[1]=changed[0]
    with pytest.raises(CatBoostError,match='(?i)snapshot.*differ|differ.*snapshot'):
        cls().set_params(**saved).fit(pool(x,y,w,changed,np.ones_like(w)),eval_set=evaluation,use_best_model=False)


@pytest.mark.parametrize('mode',['No','AnyImprovement','Armijo'])
@pytest.mark.parametrize('count',[1,4,7])
def test_grouped_binary_initial_model_baseline_and_trimmed_snapshots(tmp_path,mode,count):
    cls,x,y,w,config,_,groups,gw=grouped_problem('Logloss',kind='Bernoulli',count=count)
    config.update(leaf_estimation_backtracking=mode,iterations=7,eval_metric='Accuracy')
    y=np.where(y,'positive','negative');config['class_weights']={'negative':.7,'positive':1.8}
    learn=pool(x,y,w,groups,gw)
    initial=cls().set_params(**(config|dict(iterations=2))).fit(learn)
    learn.set_baseline(np.linspace(-.1,.2,len(y),dtype=np.float32))
    saved=config|dict(save_snapshot=True,snapshot_file='initial.snapshot',snapshot_interval=0,
                      allow_writing_files=True,train_dir=str(tmp_path))
    fit=dict(eval_set=learn,use_best_model=True,init_model=initial)
    assert cls().set_params(**saved).fit(learn,**fit,callbacks=[StopAfter(2)]).tree_count_<=4
    resumed=cls().set_params(**saved).fit(learn,**fit)
    direct=cls().set_params(**config).fit(learn,**fit)
    for key in ('get_leaf_values','get_leaf_weights','get_test_eval'):
        np.testing.assert_array_equal(getattr(resumed,key)(),getattr(direct,key)())
    assert resumed.evals_result_==direct.evals_result_
    np.testing.assert_allclose(resumed.predict_proba(x,task_type='GPU'),resumed.predict_proba(x),atol=8e-7,rtol=6e-6)


@pytest.mark.parametrize('count',[1,4])
def test_has_time_preserves_whole_pool_and_too_few_groups_fail(count):
    cls,x,y,w,config,_,groups,gw=grouped_problem(count=count)
    learn=pool(x,y,w,groups,gw)
    model=cls().set_params(**(config|dict(has_time=True))).fit(learn)
    identity=cls().set_params(**(config|dict(has_time=True,permutation_count=1))).fit(learn)
    assert model.get_metadata()['metal_permutations']=='1'
    np.testing.assert_array_equal(model.predict(x),identity.predict(x))
    with pytest.raises(CatBoostError,match='at least four groups'):
        cls().set_params(**config).fit(Pool(x,y,cat_features=[1,3],group_id=np.repeat([0,1],len(y)//2)))


@pytest.mark.parametrize('kind',['No','MVS'])
def test_empty_quality_prefix_returns_a_finite_constant_model(kind):
    x=np.arange(2000,dtype=np.float32).reshape(1000,2)%17;y=np.sin(x[:,0])
    config=dict(task_type='GPU',boosting_type='Ordered',has_time=True,iterations=3,depth=3,
                bootstrap_type=kind,random_strength=1,verbose=False,allow_writing_files=False,boost_from_average=False)
    if kind=='MVS':config['subsample']=.7
    model=CatBoostRegressor(**config).fit(Pool(x,y,group_id=np.repeat([0,1,2,3],[1,1,1,997])))
    np.testing.assert_array_equal(model.get_tree_leaf_counts(),[1,1,1])
    assert np.isfinite(model.predict(x,task_type='GPU')).all()
