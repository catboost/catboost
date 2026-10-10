"""Standalone one-hot greedy training, exact recovery and model interoperability."""
import json
import platform
import numpy as np
import pytest
from catboost_metal import CatBoostMetalClassifier, CatBoostMetalRegressor, _greedy
from test_native_greedy_onehot import problem as native_problem
from test_native_greedy_api import POLICIES, OBJECTIVES

pytestmark=pytest.mark.skipif(platform.system()!='Darwin' or platform.machine()!='arm64',reason='Apple GPU required')


@pytest.fixture(autouse=True)
def no_cpu_fit(monkeypatch):
    from catboost import CatBoost
    def forbidden(*args,**kwargs):raise AssertionError('CPU CatBoost fitting is forbidden')
    monkeypatch.setattr(CatBoost,'_fit',forbidden)


def problem(policy='Depthwise',loss='RMSE',method='Newton',kind='No'):
    _,x,y,w,config,_=native_problem(policy,loss,method,kind)
    for name in ('task_type','has_time','allow_writing_files','metric_period','verbose'):config.pop(name,None)
    config.update(cat_features=[1,3],one_hot_max_size=7)
    cls=CatBoostMetalClassifier if loss in ('Logloss','CrossEntropy') else CatBoostMetalRegressor
    return cls,x,y,w,config


def assert_readers(model,x,path):
    from catboost import CatBoost
    expected=model.predict(x,prediction_type='RawFormulaVal')
    np.testing.assert_allclose(model.predict(x,prediction_type='RawFormulaVal',task_type='GPU'),expected,atol=6e-7,rtol=5e-6)
    for fmt in ('cbm','json'):
        p=path/('greedy.'+fmt);model.save_model(p,format=fmt)
        loaded=CatBoost().load_model(p,format=fmt)
        np.testing.assert_allclose(loaded.predict(x,prediction_type='RawFormulaVal'),expected,atol=1e-12,rtol=1e-12)
        if fmt=='json':assert 'OneHotFeature' in p.read_text()


@pytest.mark.parametrize('policy',POLICIES)
@pytest.mark.parametrize('loss,method',OBJECTIVES)
def test_public_onehot_scalar_models_match_resident_routes_and_readers(tmp_path,policy,loss,method):
    cls,x,y,w,options=problem(policy,loss,method)
    model=cls(**options).fit(x,y,sample_weight=w)
    layout=model._layout
    features=[];borders=[];types=[]
    for f in range(len(layout.borders)):
        choices=layout.categorical[f].candidate_bins if f in layout.categorical else range(len(layout.borders[f]))
        features.extend([f]*len(choices));borders.extend(choices);types.extend([int(f in layout.categorical)]*len(choices))
    params={k:options[k] for k in ('iterations','depth','grow_policy','learning_rate','leaf_estimation_method',
        'leaf_estimation_iterations','leaf_estimation_backtracking','score_function','random_seed')}
    if policy=='Lossguide':params['max_leaves']=options['max_leaves']
    objective,_,parameter=loss.partition(':')
    expected=_greedy.train(layout.transform(x),y,features,borders,candidate_types=types,sample_weight=w,
        objective=objective,objective_param=float(parameter.split('=')[1]) if parameter else None,**params)
    np.testing.assert_array_equal(model.training_predictions_,expected.predictions)
    for actual,reference in zip(model._result.trees,expected.trees):
        np.testing.assert_array_equal(actual.nodes,reference.nodes)
        np.testing.assert_array_equal(actual.leaf_values,reference.leaf_values)
    x[::13,1]='unseen';assert_readers(model,x,tmp_path)
    assert model._layout.ctrs=={}


@pytest.mark.parametrize('policy',POLICIES)
@pytest.mark.parametrize('kind',['No','Bayesian','Bernoulli','Poisson'])
@pytest.mark.parametrize('classification',[False,True])
def test_public_onehot_all_samplers_restore_full_state_bitwise(tmp_path,policy,kind,classification):
    cls,x,y,w,config=problem(policy,'Logloss' if classification else 'RMSE',kind=kind)
    config['random_strength']=.7
    evaluation=x.copy();evaluation[::17,1]='unseen'
    fit=dict(sample_weight=w,eval_set=(evaluation,y,w),use_best_model=False)
    full=cls(**config).fit(x,y,**fit)
    path=tmp_path/'snapshot.npz'
    saved=dict(save_snapshot=True,snapshot_file=path,snapshot_interval=0)
    assert cls(**config).fit(x,y,**fit,**saved,callback=lambda info:info.iteration<2).tree_count_==2
    restored=cls(**config).fit(x,y,**fit,**saved)
    for name in ('training_predictions_','loss_history_'):
        np.testing.assert_array_equal(getattr(restored,name),getattr(full,name))
    assert restored.get_evals_result()==full.get_evals_result()
    for left,right in zip(restored._result.trees,full._result.trees):
        np.testing.assert_array_equal(left.nodes,right.nodes);np.testing.assert_array_equal(left.leaf_values,right.leaf_values)
    assert_readers(restored,evaluation,tmp_path)


@pytest.mark.parametrize('policy',POLICIES)
@pytest.mark.parametrize('mode',['No','AnyImprovement','Armijo'])
def test_public_binary_categories_backtracking_and_best_model(policy,mode):
    cls,x,y,w,config=problem(policy,'Logloss',kind='Bernoulli')
    config.update(iterations=8,leaf_estimation_backtracking=mode,eval_metric='Accuracy')
    model=cls(**config).fit(x,y,sample_weight=w,eval_set=(x,y,w),use_best_model=True)
    history=model.get_evals_result()['validation']['Accuracy']
    assert model.tree_count_==int(np.argmax(history))+1
    np.testing.assert_allclose(model.predict_proba(x,task_type='GPU'),model.predict_proba(x),atol=5e-7,rtol=5e-6)


@pytest.mark.parametrize('policy',POLICIES)
@pytest.mark.parametrize('changed',['learn','eval'])
def test_snapshot_identity_includes_original_hashes_even_when_dense_bins_match(tmp_path,policy,changed):
    cls,x,y,w,config=problem(policy);config.update(iterations=4,one_hot_max_size=8)
    evaluation=x.copy();evaluation[::7,1]='unseen-first'
    path=tmp_path/'identity.npz';saved=dict(save_snapshot=True,snapshot_file=path,snapshot_interval=0)
    cls(**config).fit(x,y,sample_weight=w,eval_set=(evaluation,y,w),use_best_model=False,**saved,callback=lambda info:info.iteration<2)
    if changed=='eval':evaluation[evaluation[:,1]=='unseen-first',1]='unseen-second'
    else:
        # A complete dictionary rename may preserve every dense training bin.
        from catboost_metal._categorical import cat_feature_hashes
        values=np.unique(x[:,1]);ordered=values[np.argsort(cat_feature_hashes(values))]
        names=np.array([f'renamed-{i}' for i in range(4)]);names=names[np.argsort(cat_feature_hashes(names))]
        mapping=dict(zip(ordered,names));x[:,1]=[mapping[v] for v in x[:,1]]
        evaluation=x.copy();evaluation[::7,1]='unseen-first'
        config['one_hot_max_size']=8
    with pytest.raises(ValueError,match='Snapshot does not match'):
        cls(**config).fit(x,y,sample_weight=w,eval_set=(evaluation,y,w),use_best_model=False,**saved)


@pytest.mark.parametrize('policy',POLICIES)
def test_dataframe_feature_identity_onall_threshold_and_pool_constraints(tmp_path,policy):
    import pandas as pd
    from catboost import Pool
    cls,x,y,w,config=problem(policy);config['cat_features']=['category','other']
    frame=pd.DataFrame(x,columns=['left','category','right','other'])
    heldout=frame.copy();heldout.loc[0,'category']='unseen'
    model=cls(**config).fit(frame,y,sample_weight=w,eval_set=(heldout,y),use_best_model=False)
    assert_readers(model,heldout,tmp_path)
    with pytest.raises(ValueError,match='names/order'):
        cls(**config).fit(frame,y,eval_set=(heldout[['left','other','right','category']],y))
    ctr=cls(**(config|dict(one_hot_max_size=4))).fit(frame,y,eval_set=(heldout,y))
    assert ctr._layout.ctrs and ctr._layout.permutation_count==4
    assert_readers(ctr,heldout,tmp_path)
    with pytest.raises(ValueError,match='cannot extract raw categories'):
        cls(**config).fit(Pool(frame,y,cat_features=['category','other']))
