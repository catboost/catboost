"""Grouped scalar Ordered through standalone arrays, DataFrames and raw Pools."""
import platform
import numpy as np
import pytest
from catboost import Pool
from catboost_metal import CatBoostMetalClassifier,CatBoostMetalRegressor,_ordered
from test_native_ordered_groups import grouped_problem,SIZES,pool
from test_native_ordered_onehot import OBJECTIVES
from test_greedy_onehot_public import no_cpu_fit,assert_readers

pytestmark=pytest.mark.skipif(platform.system()!='Darwin' or platform.machine()!='arm64',reason='Apple GPU required')


def problem(loss='RMSE',method='Newton',kind='No',count=4):
    _,x,y,w,config,_,groups,gw=grouped_problem(loss,method,kind,count)
    for key in ('task_type','has_time','verbose','allow_writing_files','metric_period'):config.pop(key,None)
    config.update(cat_features=[1,3],one_hot_max_size=7)
    cls=CatBoostMetalClassifier if loss in ('Logloss','CrossEntropy') else CatBoostMetalRegressor
    return cls,x,y,w,config,groups,gw


@pytest.mark.parametrize('loss,method',[*OBJECTIVES,('Lq:q=2.5','Newton')])
@pytest.mark.parametrize('count',[1,4,7])
def test_group_weights_and_prefix_forests_match_private_runtime(tmp_path,loss,method,count):
    cls,x,y,w,config,g,gw=problem(loss,method,count=count)
    model=cls(**config).fit(x,y,sample_weight=w,group_id=g,group_weight=gw)
    layout=model._layout;features=[];borders=[];types=[]
    for f in range(len(layout.borders)):
        choices=layout.categorical[f].candidate_bins if f in layout.categorical else range(len(layout.borders[f]))
        features.extend([f]*len(choices));borders.extend(choices);types.extend([int(f in layout.categorical)]*len(choices))
    keys=('iterations','depth','learning_rate','leaf_estimation_method','leaf_estimation_iterations',
          'leaf_estimation_backtracking','score_function','random_seed','permutation_count',
          'min_fold_size','fold_len_multiplier','fold_permutation_block')
    objective,_,parameter=loss.partition(':')
    expected=_ordered.train(layout.transform(x),y,features,borders,candidate_types=types,sample_weight=w*gw,
        group_sizes=SIZES,objective=objective,objective_param=float(parameter.split('=')[1]) if parameter else None,
        **{key:config[key] for key in keys})
    for key in ('depths','split_features','split_bins','split_types','leaf_values','leaf_weights','predictions'):
        np.testing.assert_array_equal(getattr(model._result,key),getattr(expected,key))
    x[::13,1]='unseen';assert_readers(model,x,tmp_path)


@pytest.mark.parametrize('kind',['No','Bayesian','Bernoulli','Poisson','MVS'])
@pytest.mark.parametrize('count',[1,4,7])
@pytest.mark.parametrize('classification',[False,True])
def test_grouped_validation_and_prefix_snapshot_recover_bitwise(tmp_path,kind,count,classification):
    cls,x,y,w,config,g,gw=problem('Logloss' if classification else 'RMSE',kind=kind,count=count)
    config['random_strength']=.7
    heldout=x.copy();heldout[::17,1]='unseen'
    fit=dict(sample_weight=w,group_id=g,group_weight=gw,eval_set=(heldout,y,w),
             eval_group_id=g,eval_group_weight=gw,use_best_model=False)
    full=cls(**config).fit(x,y,**fit)
    path=tmp_path/'snapshot.npz';saved=dict(save_snapshot=True,snapshot_file=path,snapshot_interval=0)
    assert cls(**config).fit(x,y,**fit,**saved,callback=lambda info:info.iteration<2).tree_count_==2
    restored=cls(**config).fit(x,y,**fit,**saved)
    np.testing.assert_array_equal(restored.training_predictions_,full.training_predictions_)
    np.testing.assert_array_equal(restored.loss_history_,full.loss_history_)
    assert restored.get_evals_result()==full.get_evals_result()
    assert_readers(restored,heldout,tmp_path)


@pytest.mark.parametrize('mode',['No','AnyImprovement','Armijo'])
@pytest.mark.parametrize('count',[1,4,7])
def test_labels_class_weights_and_trimmed_full_cursor_recovery(tmp_path,mode,count):
    cls,x,y,w,config,g,gw=problem('Logloss',kind='Bernoulli',count=count)
    config.update(iterations=8,leaf_estimation_backtracking=mode,eval_metric='Accuracy',
                  class_weights={'negative':.7,'positive':1.8})
    y=np.where(y,'positive','negative')
    fit=dict(sample_weight=w,group_id=g,group_weight=gw,eval_set=(x,y,w),
             eval_group_id=g,eval_group_weight=gw,use_best_model=True)
    saved=dict(save_snapshot=True,snapshot_file=tmp_path/'best.npz',snapshot_interval=0)
    model=cls(**config).fit(x,y,**fit,**saved)
    assert model.tree_count_==int(np.argmax(model.get_evals_result()['validation']['Accuracy']))+1
    with np.load(saved['snapshot_file'],allow_pickle=False) as archive:assert len(archive['depths'])==8
    restored=cls(**config).fit(x,y,**fit,**saved)
    np.testing.assert_array_equal(restored.training_predictions_,model.training_predictions_)
    assert model.classes_.tolist()==['negative','positive']
    np.testing.assert_allclose(model.predict_proba(x,task_type='GPU'),model.predict_proba(x),atol=8e-7,rtol=6e-6)


@pytest.mark.parametrize('count',[1,4])
def test_numeric_pool_exposes_and_applies_group_weights_once(count):
    cls,x,y,w,config,g,gw=problem(count=count);x=x[:,[0,2]].astype(np.float32);config['cat_features']=[]
    learn=pool(x,y,w,g,gw,cats=[])
    from_pool=cls(**config).fit(learn,eval_set=learn,use_best_model=False)
    arrays=cls(**config).fit(x,y,sample_weight=w,group_id=g,group_weight=gw,
                            eval_set=(x,y,w),eval_group_id=g,eval_group_weight=gw,use_best_model=False)
    np.testing.assert_array_equal(from_pool.training_predictions_,arrays.training_predictions_)
    assert from_pool.get_evals_result()==arrays.get_evals_result()
    learn.set_baseline(np.zeros(len(y)))
    with pytest.raises(ValueError,match='baselines'):
        cls(**config).fit(learn)


@pytest.mark.parametrize('count',[1,4])
def test_group_boundaries_alone_change_snapshot_identity(tmp_path,count):
    cls,x,y,w,config,g,gw=problem(count=count);gw[:]=1
    saved=dict(save_snapshot=True,snapshot_file=tmp_path/'identity.npz',snapshot_interval=0)
    cls(**config).fit(x,y,sample_weight=w,group_id=g,group_weight=gw,**saved,callback=lambda info:info.iteration<2)
    changed=g.copy();changed[1]=changed[0]
    with pytest.raises(ValueError,match='Snapshot does not match'):
        cls(**config).fit(x,y,sample_weight=w,group_id=changed,group_weight=gw,**saved)


@pytest.mark.parametrize('case',['missing','too_few','noncontiguous','weight','eval_missing','plain'])
def test_group_contracts_fail_clearly(case):
    cls,x,y,w,config,g,gw=problem()
    fit=dict(sample_weight=w,group_id=g,group_weight=gw)
    if case=='missing':fit.pop('group_id')
    elif case=='too_few':fit.update(group_id=np.repeat([0,1],len(y)//2),group_weight=np.ones(len(y)))
    elif case=='noncontiguous':g=g.copy();g[2]=g[0];fit['group_id']=g
    elif case=='weight':gw=gw.copy();gw[2]*=2;fit['group_weight']=gw
    elif case=='eval_missing':fit['eval_group_id']=g
    else:config['boosting_type']='Plain'
    with pytest.raises(ValueError):cls(**config).fit(x,y,**fit)


def test_grouped_dataframe_validation_preserves_names(tmp_path):
    import pandas as pd
    cls,x,y,w,config,g,gw=problem();config['cat_features']=['category','other']
    frame=pd.DataFrame(x,columns=['left','category','right','other'])
    model=cls(**config).fit(frame,y,sample_weight=w,group_id=g,group_weight=gw,
        eval_set=(frame,y,w),eval_group_id=g,eval_group_weight=gw,use_best_model=False)
    assert_readers(model,frame,tmp_path)
    with pytest.raises(ValueError,match='names/order'):
        cls(**config).fit(frame,y,group_id=g,eval_set=(frame[['left','other','right','category']],y),eval_group_id=g)
