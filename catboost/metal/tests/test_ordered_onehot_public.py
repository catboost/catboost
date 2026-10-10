"""Standalone Ordered categories through public fitting and full-state recovery."""
import platform
import numpy as np
import pytest
from catboost_metal import CatBoostMetalClassifier, CatBoostMetalRegressor, _ordered
from test_native_ordered_onehot import problem as native_problem
from test_native_greedy_api import OBJECTIVES
from test_greedy_onehot_public import assert_readers, no_cpu_fit

pytestmark=pytest.mark.skipif(platform.system()!='Darwin' or platform.machine()!='arm64',reason='Apple GPU required')


def problem(loss='RMSE',method='Newton',kind='No',count=4):
    _,x,y,w,config,_=native_problem(loss,method,kind,count)
    for name in ('task_type','has_time','allow_writing_files','metric_period','verbose'):config.pop(name,None)
    config.update(cat_features=[1,3],one_hot_max_size=7)
    cls=CatBoostMetalClassifier if loss in ('Logloss','CrossEntropy') else CatBoostMetalRegressor
    return cls,x,y,w,config


@pytest.mark.parametrize('loss,method',OBJECTIVES)
@pytest.mark.parametrize('count',[1,4])
def test_public_training_matches_private_ordered_forests(tmp_path,loss,method,count):
    cls,x,y,w,config=problem(loss,method,count=count)
    model=cls(**config).fit(x,y,sample_weight=w)
    layout=model._layout;features=[];borders=[];types=[]
    for f in range(len(layout.borders)):
        choices=layout.categorical[f].candidate_bins if f in layout.categorical else range(len(layout.borders[f]))
        features.extend([f]*len(choices));borders.extend(choices);types.extend([int(f in layout.categorical)]*len(choices))
    keys=('iterations','depth','learning_rate','leaf_estimation_method','leaf_estimation_iterations',
          'leaf_estimation_backtracking','score_function','random_seed','permutation_count',
          'min_fold_size','fold_len_multiplier','fold_permutation_block')
    objective,_,parameter=loss.partition(':')
    expected=_ordered.train(layout.transform(x),y,features,borders,candidate_types=types,sample_weight=w,
        objective=objective,objective_param=float(parameter.split('=')[1]) if parameter else None,
        **{k:config[k] for k in keys})
    for name in ('depths','split_features','split_bins','split_types','leaf_values','leaf_weights','predictions'):
        np.testing.assert_array_equal(getattr(model._result,name),getattr(expected,name))
    assert expected.split_types.any()
    x[::13,1]='unseen';assert_readers(model,x,tmp_path)
    assert model._layout.ctrs=={}


@pytest.mark.parametrize('kind',['No','Bayesian','Bernoulli','Poisson','MVS'])
@pytest.mark.parametrize('count',[1,4])
@pytest.mark.parametrize('classification',[False,True])
def test_all_samplers_restore_entire_prefix_state_and_validation(tmp_path,kind,count,classification):
    cls,x,y,w,config=problem('Logloss' if classification else 'RMSE',kind=kind,count=count)
    config['random_strength']=.7
    evaluation=x.copy();evaluation[::17,1]='unseen'
    fit=dict(sample_weight=w,eval_set=(evaluation,y,w),use_best_model=False)
    full=cls(**config).fit(x,y,**fit)
    path=tmp_path/'snapshot.npz';saved=dict(save_snapshot=True,snapshot_file=path,snapshot_interval=0)
    assert cls(**config).fit(x,y,**fit,**saved,callback=lambda info:info.iteration<2).tree_count_==2
    with np.load(path,allow_pickle=False) as archive:
        assert archive['ordered_cursors'].size>len(y)
        before=archive['ordered_cursors'].copy()
    restored=cls(**config).fit(x,y,**fit,**saved)
    for name in ('depths','split_features','split_bins','split_types','leaf_values','leaf_weights','predictions','rmse'):
        np.testing.assert_array_equal(getattr(restored._result,name),getattr(full._result,name))
    assert restored.get_evals_result()==full.get_evals_result()
    with np.load(path,allow_pickle=False) as archive:assert not np.array_equal(archive['ordered_cursors'],before)
    assert_readers(restored,evaluation,tmp_path)


@pytest.mark.parametrize('mode',['No','AnyImprovement','Armijo'])
@pytest.mark.parametrize('count',[1,4])
def test_backtracking_best_model_saves_untrimmed_state(tmp_path,mode,count):
    cls,x,y,w,config=problem('Logloss',kind='Bernoulli',count=count)
    config.update(iterations=8,leaf_estimation_backtracking=mode,eval_metric='Accuracy')
    path=tmp_path/'best.npz'
    saved=dict(save_snapshot=True,snapshot_file=path,snapshot_interval=0)
    model=cls(**config).fit(x,y,sample_weight=w,eval_set=(x,y,w),use_best_model=True,**saved)
    history=model.get_evals_result()['validation']['Accuracy']
    assert model.tree_count_==int(np.argmax(history))+1
    with np.load(path,allow_pickle=False) as archive:assert len(archive['depths'])==8
    restored=cls(**config).fit(x,y,sample_weight=w,eval_set=(x,y,w),use_best_model=True,**saved)
    np.testing.assert_array_equal(restored.training_predictions_,model.training_predictions_)
    np.testing.assert_allclose(model.predict_proba(x,task_type='GPU'),model.predict_proba(x),atol=6e-7,rtol=5e-6)


@pytest.mark.parametrize('changed',['learn','eval'])
@pytest.mark.parametrize('count',[1,4])
def test_snapshot_identity_includes_hashes_when_dense_bins_match(tmp_path,changed,count):
    cls,x,y,w,config=problem(count=count)
    evaluation=x.copy();evaluation[::7,1]='unseen-first'
    path=tmp_path/'identity.npz';saved=dict(save_snapshot=True,snapshot_file=path,snapshot_interval=0)
    original=x.copy()
    initial=cls(**config).fit(x,y,sample_weight=w,eval_set=(evaluation,y,w),use_best_model=False,
                             **saved,callback=lambda info:info.iteration<2)
    if changed=='eval':evaluation[evaluation[:,1]=='unseen-first',1]='unseen-second'
    else:
        from catboost_metal._categorical import cat_feature_hashes
        values=np.unique(x[:,1]);ordered=values[np.argsort(cat_feature_hashes(values))]
        names=np.array([f'renamed-{i}' for i in range(4)]);names=names[np.argsort(cat_feature_hashes(names))]
        mapping=dict(zip(ordered,names));x[:,1]=[mapping[v] for v in x[:,1]]
        evaluation=x.copy();evaluation[::7,1]='unseen-first'
        fresh=cls(**config).fit(x,y,sample_weight=w,eval_set=(evaluation,y,w),use_best_model=False)
        np.testing.assert_array_equal(initial._layout.transform(original),fresh._layout.transform(x))
    with pytest.raises(ValueError,match='Snapshot does not match'):
        cls(**config).fit(x,y,sample_weight=w,eval_set=(evaluation,y,w),use_best_model=False,**saved)


@pytest.mark.parametrize('count',[1,4])
def test_names_onall_threshold_and_constant_categories(tmp_path,count):
    import pandas as pd
    cls,x,y,w,config=problem(count=count);config['cat_features']=['category','other']
    frame=pd.DataFrame(x,columns=['left','category','right','other'])
    heldout=frame.copy();heldout.loc[0,'category']='unseen'
    model=cls(**config).fit(frame,y,sample_weight=w,eval_set=(heldout,y),use_best_model=False)
    assert_readers(model,heldout,tmp_path)
    with pytest.raises(ValueError,match='names/order'):
        cls(**config).fit(frame,y,eval_set=(heldout[['left','other','right','category']],y))
    ctr_model=cls(**(config|dict(one_hot_max_size=4))).fit(frame,y,eval_set=(heldout,y))
    assert ctr_model._layout.ctrs
    assert_readers(ctr_model,heldout,tmp_path)
    pure=np.column_stack((np.repeat('constant',len(x)),x[:,1],x[:,3]))
    model=cls(**(config|dict(cat_features=[0,1,2]))).fit(pure,y,sample_weight=w)
    assert_readers(model,pure,tmp_path)


@pytest.mark.parametrize('count',[1,4])
def test_original_class_labels_weights_and_empty_prediction(count):
    cls,x,y,w,config=problem('Logloss',count=count)
    labels=np.where(y,'positive','negative')
    config['class_weights']={'negative':.7,'positive':1.8}
    model=cls(**config).fit(x,labels,sample_weight=w)
    assert model.classes_.tolist()==['negative','positive']
    np.testing.assert_allclose(model.predict_proba(x,task_type='GPU'),model.predict_proba(x),atol=6e-7,rtol=5e-6)
    assert model.predict_proba(x[:0],task_type='GPU').shape==(0,2)
