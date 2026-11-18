"""Standalone greedy CTR lifecycle, all-bank snapshots and standard models."""
import json
import numpy as np
import pytest
from catboost import CatBoost
from catboost_metal import _greedy
from catboost_metal._data import cuda_search_permutation, prepare_features
from catboost_metal._greedy_inference import predict_bins
from catboost_metal._training import _fingerprint, _json
from test_greedy_onehot_public import problem as onehot_problem, POLICIES, OBJECTIVES


@pytest.fixture(autouse=True)
def no_cpu_fit(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('CPU CatBoost fitting is forbidden')
    monkeypatch.setattr(CatBoost,'_fit',forbidden)


def problem(policy='Depthwise',loss='RMSE',method='Newton',kind='No',ctr='Borders',count=4):
    cls,x,y,w,config=onehot_problem(policy,loss,method,kind)
    config.update(one_hot_max_size=1,ctr_type=ctr,ctr_border_count=5,permutation_count=count)
    return cls,x,y,w,config


def readers(model,x,path):
    raw=model.predict(x,prediction_type='RawFormulaVal')
    np.testing.assert_allclose(model.predict(x,prediction_type='RawFormulaVal',task_type='GPU'),raw,atol=7e-7,rtol=5e-6)
    for fmt in ('cbm','json'):
        p=path/('greedy-ctr.'+fmt);model.save_model(p,format=fmt)
        loaded=CatBoost().load_model(p,format=fmt)
        np.testing.assert_allclose(loaded.predict(x,prediction_type='RawFormulaVal'),raw,atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose(loaded.predict(x,prediction_type='RawFormulaVal',task_type='GPU'),raw,atol=7e-7,rtol=5e-6)
        if fmt=='json' and any(len(t.leaf_values)>1 for t in model._result.trees):
            assert 'OnlineCtr' in p.read_text()


def resident(model,y,w,config):
    layout=model._layout;features=[];borders=[];types=[]
    for f,grid in enumerate(layout.borders):
        choices=layout.categorical[f].candidate_bins if f in layout.categorical else range(len(grid))
        features.extend([f]*len(choices));borders.extend(choices);types.extend([int(f in layout.categorical)]*len(choices))
    args={k:config[k] for k in ('iterations','depth','grow_policy','learning_rate','leaf_estimation_method',
        'leaf_estimation_iterations','leaf_estimation_backtracking','score_function','random_seed','bootstrap_type','random_strength')}
    for k in ('max_leaves','l2_leaf_reg','bagging_temperature','subsample'):
        if k in config:args[k]=config[k]
    loss,_,param=config['loss_function'].partition(':')
    with _greedy.Session(layout.permutation_bins[0],y,features,borders,candidate_types=types,
            sample_weight=w,objective=loss,objective_param=float(param.split('=')[1]) if param else None,**args) as oracle:
        if layout.permutation_count>1:oracle.configure_permutations(layout.permutation_bins)
        for t in range(config['iterations']):
            oracle.select_permutation(cuda_search_permutation(config['random_seed'],t,layout.permutation_count))
            actual=oracle.step();expected=model._result.trees[t]
            for key in ('nodes','leaf_values','leaf_weights'):
                np.testing.assert_array_equal(getattr(actual,key),getattr(expected,key))
        np.testing.assert_array_equal(model.training_predictions_,oracle.predictions())
        return oracle.permutation_state['predictions']


@pytest.mark.parametrize('policy',POLICIES)
@pytest.mark.parametrize('loss,method',OBJECTIVES)
@pytest.mark.parametrize('ctr',['Borders','FeatureFreq'])
def test_public_ctr_forests_use_independent_dataset_cursors(tmp_path,policy,loss,method,ctr):
    cls,x,y,w,config=problem(policy,loss,method,ctr=ctr)
    model=cls(**config).fit(x,y,sample_weight=w)
    assert model._layout.ctrs and model._layout.permutation_count==4
    resident(model,y,w,config)
    x[::13,1]='unseen';readers(model,x,tmp_path)


@pytest.mark.parametrize('policy',POLICIES)
@pytest.mark.parametrize('kind',['No','Bayesian','Bernoulli','Poisson'])
@pytest.mark.parametrize('ctr',['Borders','FeatureFreq'])
def test_public_ctr_all_samplers_restore_every_cursor_and_history(tmp_path,policy,kind,ctr):
    cls,x,y,w,config=problem(policy,kind=kind,ctr=ctr);config['random_strength']=.4
    evaluation=x.copy();evaluation[::17,1]='unseen'
    fit=dict(sample_weight=w,eval_set=(evaluation,y,w),use_best_model=False)
    full=cls(**config).fit(x,y,**fit);expected=resident(full,y,w,config)
    path=tmp_path/'snapshot.npz';saved=dict(save_snapshot=True,snapshot_file=path,snapshot_interval=0)
    assert cls(**config).fit(x,y,**fit,**saved,callback=lambda i:i.iteration<2).tree_count_==2
    resumed=cls(**config).fit(x,y,**fit,**saved)
    for key in ('training_predictions_','loss_history_'):
        np.testing.assert_array_equal(getattr(resumed,key),getattr(full,key))
    for a,b in zip(resumed._result.trees,full._result.trees):
        for key in ('nodes','leaf_values','leaf_weights'):np.testing.assert_array_equal(getattr(a,key),getattr(b,key))
    assert resumed.get_evals_result()==full.get_evals_result()
    with np.load(path,allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive['permutation_predictions'],expected)
        np.testing.assert_array_equal(archive['predictions'],expected[-1])
    readers(resumed,evaluation,tmp_path)


@pytest.mark.parametrize('policy',POLICIES)
@pytest.mark.parametrize('mode',['No','AnyImprovement','Armijo'])
def test_public_binary_ctr_backtracking_and_trim_use_last_bank(tmp_path,policy,mode):
    cls,x,y,w,config=problem(policy,'Logloss',kind='Bernoulli')
    config.update(iterations=8,eval_metric='Accuracy',leaf_estimation_backtracking=mode)
    path=tmp_path/'best.npz'
    fit=dict(sample_weight=w,eval_set=(x,y,w),use_best_model=True,save_snapshot=True,snapshot_file=path)
    model=cls(**config).fit(x,y,**fit)
    history=model.get_evals_result()['validation']['Accuracy']
    assert model.tree_count_==int(np.argmax(history))+1
    np.testing.assert_array_equal(model.training_predictions_,predict_bins(model._layout.permutation_bins[-1],model._result.trees))
    with np.load(path,allow_pickle=False) as archive:
        assert len(archive['loss'])==9 and archive['permutation_predictions'].shape==(4,len(x))
    resumed=cls(**(config|dict(iterations=10))).fit(x,y,**fit)
    direct=cls(**(config|dict(iterations=10))).fit(x,y,sample_weight=w,eval_set=(x,y,w),use_best_model=True)
    np.testing.assert_array_equal(resumed.training_predictions_,direct.training_predictions_)
    assert resumed.get_evals_result()==direct.get_evals_result()
    readers(model,x,tmp_path)


@pytest.mark.parametrize('policy',POLICIES)
@pytest.mark.parametrize('loss',['Quantile:alpha=0.7','MAE','MAPE'])
def test_exact_ctr_leaves_match_resident_final_bank(tmp_path,policy,loss):
    cls,x,y,w,config=problem(policy,loss,'Exact',kind='Poisson')
    model=cls(**config).fit(x,y,sample_weight=w);resident(model,y,w,config);readers(model,x,tmp_path)


@pytest.mark.parametrize('policy',POLICIES)
@pytest.mark.parametrize('count',[1,2,7,64])
def test_public_dataset_counts_and_no_symmetric_model_size_penalty(policy,count):
    cls,x,y,w,config=problem(policy,count=count);config['iterations']=2
    a=cls(**(config|dict(model_size_reg=0))).fit(x,y,sample_weight=w)
    b=cls(**(config|dict(model_size_reg=3))).fit(x,y,sample_weight=w)
    assert a._layout.permutation_count==count
    np.testing.assert_array_equal(a.training_predictions_,b.training_predictions_)
    resident(a,y,w,config)


@pytest.mark.parametrize('change',['shape','nan','last_cursor','missing','learn_hash','eval_hash','count'])
def test_snapshots_bind_all_dataset_arrays_and_original_category_identity(tmp_path,change):
    cls,x,y,w,config=problem();evaluation=x.copy();evaluation[::7,1]='unknown-original'
    path=tmp_path/'snapshot.npz';fit=dict(sample_weight=w,eval_set=(evaluation,y,w),use_best_model=False,
        save_snapshot=True,snapshot_file=path,snapshot_interval=0)
    cls(**config).fit(x,y,**fit,callback=lambda i:i.iteration<2)
    if change in ('shape','nan','last_cursor','missing'):
        with np.load(path,allow_pickle=False) as archive:arrays={k:archive[k] for k in archive.files}
        header=json.loads(str(arrays.pop('metadata')));header.pop('checksum')
        if change=='shape':arrays['permutation_predictions']=arrays['permutation_predictions'][:,:-1]
        if change=='nan':arrays['permutation_predictions'][0,0]=np.nan
        if change=='last_cursor':arrays['permutation_predictions'][-1,0]+=.25
        if change=='missing':arrays.pop('permutation_predictions')
        header['checksum']=_fingerprint(arrays,header);arrays['metadata']=np.asarray(_json(header));np.savez(path,**arrays)
    elif change=='learn_hash':x=x.copy();x[0,1]='changed-original-category'
    elif change=='eval_hash':evaluation[::7,1]='unknown-replacement'
    else:config['permutation_count']=2
    with pytest.raises(ValueError,match='[Ss]napshot'):
        cls(**config).fit(x,y,**fit)


@pytest.mark.parametrize('policy',POLICIES)
def test_eval_only_category_moves_feature_from_onehot_to_ctr_with_dataframe_names(tmp_path,policy):
    import pandas as pd
    cls,x,y,w,config=onehot_problem(policy)
    config.update(one_hot_max_size=4,cat_features=['category','other'])
    frame=pd.DataFrame(x,columns=['left','category','right','other'])
    assert not cls(**config).fit(frame,y)._layout.ctrs
    evaluation=frame.copy();evaluation.loc[0,'category']='eval-only'
    model=cls(**config).fit(frame,y,eval_set=(evaluation,y),use_best_model=False)
    assert len(model._layout.ctrs)==1 and model._layout.permutation_count==4
    readers(model,evaluation,tmp_path)
