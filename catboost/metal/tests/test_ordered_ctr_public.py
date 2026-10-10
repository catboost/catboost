"""Standalone Ordered CTR preparation, shared lifecycle and final model tables."""
import json
import numpy as np
import pytest
from catboost_metal import _ordered
from test_ordered_groups_public import problem as grouped_problem,SIZES
from test_greedy_onehot_public import no_cpu_fit
from catboost import CatBoost
from test_native_ordered_ctrs import LOSSES


def assert_readers(model,x,path):
    raw=model.predict(x,prediction_type='RawFormulaVal')
    np.testing.assert_allclose(model.predict(x,prediction_type='RawFormulaVal',task_type='GPU'),raw,atol=7e-7,rtol=5e-6)
    for fmt in ('cbm','json'):
        p=path/('ordered-ctr.'+fmt);model.save_model(p,format=fmt)
        loaded=CatBoost().load_model(p,format=fmt)
        np.testing.assert_allclose(loaded.predict(x,prediction_type='RawFormulaVal'),raw,atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose(loaded.predict(x,prediction_type='RawFormulaVal',task_type='GPU'),raw,atol=7e-7,rtol=5e-6)
        if fmt=='json':
            doc=json.loads(p.read_text())
            if any(s['split_type']=='OnlineCtr' for t in doc['oblivious_trees'] for s in t['splits'] or []):
                assert doc['ctr_data'] and doc['features_info']['ctrs']


def problem(loss='RMSE',method='Newton',kind='No',count=4,ctr='Borders',history='Sample'):
    cls,x,y,w,config,g,gw=grouped_problem(loss,method,kind,count)
    config.update(one_hot_max_size=1,ctr_type=ctr,ctr_border_count=5,ctr_history_unit=history,model_size_reg=.8)
    return cls,x,y,w,config,g,gw


def resident(model,y,w,config,grouped=True):
    layout=model._layout;features=[];borders=[];types=[]
    for f,grid in enumerate(layout.borders):
        choices=layout.categorical[f].candidate_bins if f in layout.categorical else range(len(grid))
        features.extend([f]*len(choices));borders.extend(choices);types.extend([int(f in layout.categorical)]*len(choices))
    args={k:getattr(model,k) for k in ('iterations','depth','learning_rate','l2_leaf_reg','leaf_estimation_method',
        'leaf_estimation_iterations','leaf_estimation_backtracking','score_function','random_seed','bootstrap_type',
        'random_strength','bagging_temperature','subsample','mvs_reg','permutation_count','fold_permutation_block',
        'fold_len_multiplier','min_fold_size','model_size_reg')}
    objective,_,parameter=config['loss_function'].partition(':')
    expected=_ordered.train(layout.permutation_bins[0],y,features,borders,candidate_types=types,sample_weight=w,
        permutation_bins=layout.permutation_bins,ctr_unique_values=layout.ctr_unique_values(),group_sizes=SIZES if grouped else None,
        objective=objective,objective_param=float(parameter.split('=')[1]) if parameter else None,**args)
    for key in ('depths','split_features','split_bins','split_types','leaf_values','leaf_weights','predictions'):
        np.testing.assert_array_equal(getattr(model._result,key),getattr(expected,key))


@pytest.mark.parametrize('loss,method',LOSSES)
@pytest.mark.parametrize('ctr',['Borders','FeatureFreq'])
@pytest.mark.parametrize('history',['Sample','Group'])
def test_public_ctr_forests_match_every_resident_prefix(tmp_path,loss,method,ctr,history):
    cls,x,y,w,config,g,gw=problem(loss,method,ctr=ctr,history=history)
    model=cls(**config).fit(x,y,sample_weight=w,group_id=g,group_weight=gw)
    assert model._layout.ctrs and model._layout.permutation_count==4
    resident(model,y,w*gw,config)
    x[::13,1]='unseen';assert_readers(model,x,tmp_path)


@pytest.mark.parametrize('kind',['No','Bayesian','Bernoulli','Poisson','MVS'])
@pytest.mark.parametrize('count',[1,4,7])
@pytest.mark.parametrize('history',['Sample','Group'])
@pytest.mark.parametrize('ctr',['Borders','FeatureFreq'])
def test_public_banked_snapshots_recover_all_folds(tmp_path,kind,count,history,ctr):
    cls,x,y,w,config,g,gw=problem(kind=kind,count=count,history=history,ctr=ctr);config['random_strength']=.7
    heldout=x.copy();heldout[::17,1]='unseen'
    fit=dict(sample_weight=w,group_id=g,group_weight=gw,eval_set=(heldout,y,w),
        eval_group_id=g,eval_group_weight=gw,use_best_model=False)
    direct=cls(**config).fit(x,y,**fit)
    saved=dict(save_snapshot=True,snapshot_file=tmp_path/'ctr.npz',snapshot_interval=0)
    assert cls(**config).fit(x,y,**fit,**saved,callback=lambda info:info.iteration<2).tree_count_==2
    resumed=cls(**config).fit(x,y,**fit,**saved)
    for key in ('training_predictions_','loss_history_'):np.testing.assert_array_equal(getattr(resumed,key),getattr(direct,key))
    assert resumed.get_evals_result()==direct.get_evals_result()
    assert_readers(resumed,heldout,tmp_path)
    changed=x.copy();changed[0,1]='changed-original-category'
    with pytest.raises(ValueError,match='Snapshot does not match'):
        cls(**config).fit(changed,y,**fit,**saved)


@pytest.mark.parametrize('mode',['No','AnyImprovement','Armijo'])
@pytest.mark.parametrize('count',[1,4,7])
def test_ctr_best_model_labels_weights_and_extension(tmp_path,mode,count):
    cls,x,y,w,config,g,gw=problem('Logloss',kind='Bernoulli',count=count,history='Group')
    config.update(iterations=8,leaf_estimation_backtracking=mode,eval_metric='Accuracy',class_weights={'negative':.7,'positive':1.8})
    labels=np.where(y,'positive','negative')
    fit=dict(sample_weight=w,group_id=g,group_weight=gw,eval_set=(x,labels,w),
        eval_group_id=g,eval_group_weight=gw,use_best_model=True)
    saved=dict(save_snapshot=True,snapshot_file=tmp_path/'best.npz',snapshot_interval=0)
    model=cls(**config).fit(x,labels,**fit,**saved)
    assert model.tree_count_==int(np.argmax(model.get_evals_result()['validation']['Accuracy']))+1
    with np.load(saved['snapshot_file'],allow_pickle=False) as archive:assert len(archive['depths'])==8
    restored=cls(**config).fit(x,labels,**fit,**saved)
    np.testing.assert_array_equal(restored.training_predictions_,model.training_predictions_)
    extended=cls(**(config|dict(iterations=11))).fit(x,labels,**fit,**saved)
    direct=cls(**(config|dict(iterations=11))).fit(x,labels,**fit)
    np.testing.assert_array_equal(extended.training_predictions_,direct.training_predictions_)
    np.testing.assert_allclose(extended.predict_proba(x,task_type='GPU'),extended.predict_proba(x),atol=8e-7,rtol=6e-6)


@pytest.mark.parametrize('count',[1,4,64])
def test_ungrouped_ctr_forests_keep_numeric_fold_layout(count):
    cls,x,y,w,config,g,gw=problem(count=count)
    model=cls(**config).fit(x,y,sample_weight=w)
    resident(model,y,w,config,grouped=False)


@pytest.mark.parametrize('history',['Sample','Group'])
def test_large_block_histories_match_independent_cuda_orders(history):
    from catboost_metal._data import prepare_features
    from catboost_metal._ordered_rng import cuda_ordered_group_history_order
    from catboost_metal._categorical import cat_feature_hashes
    from cuda_ordered_group_reference import reference_group_order
    from test_ctrs_group import _reference
    sizes=[101,17,29,53]*251;rows=sum(sizes);groups=np.repeat(np.arange(len(sizes),dtype=np.uint32),sizes)
    rng=np.random.default_rng(71);raw=np.array([f'cat{i}' for i in rng.integers(0,97,rows)],object)[:,None]
    y=rng.integers(0,2,rows).astype(np.float32)
    orders=np.stack([cuda_ordered_group_history_order(sizes,p,129) for p in range(4)])
    layout,bins,_,_,_=prepare_features(raw,4,'Forbidden',[0],['category'],1,targets=y,objective='Logloss',
        permutation_count=4,history_orders=orders,ctr_group_ids=groups if history=='Group' else None)
    for p in range(4):
        order=reference_group_order(sizes,p,129)[1];np.testing.assert_array_equal(orders[p],order)
        expected,_=_reference(cat_feature_hashes(raw[:,0]),y,order,groups if history=='Group' else np.arange(rows),
            'Borders',0,.5,1.)
        np.testing.assert_array_equal(layout.permutation_bins[p][-1],np.searchsorted(layout.borders[-1],expected.astype(np.float32),side='left'))
