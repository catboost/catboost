"""Native Ordered equality splits, category identity and prefix recovery."""
import os
import numpy as np
import pytest
from catboost import CatBoost, CatBoostClassifier, CatBoostRegressor, CatBoostError, Pool
from catboost_metal import _ordered
from catboost_metal._categorical import cat_feature_hashes
from test_native_greedy_onehot import problem as categorical_problem, assert_readers, only_gpu
from test_native_greedy_api import OBJECTIVES, sampler, StopAfter

pytestmark = pytest.mark.skipif(os.environ.get('CATBOOST_NATIVE_METAL_TESTS') != '1',
                               reason='requires rebuilt native Ordered one-hot adapter')


def problem(loss='RMSE', method='Newton', kind='No', count=4):
    cls,x,y,w,config,numeric = categorical_problem('SymmetricTree',loss,method,kind)
    config.update(boosting_type='Ordered',has_time=False,permutation_count=count,
                  min_fold_size=16,fold_len_multiplier=1.7,fold_permutation_block=3)
    return cls,x,y,w,config,numeric


def resident(config,x,y,w,numeric,baseline=None,group_sizes=None):
    banks=list(numeric);features=[];borders=[];types=[]
    for f in range(2):
        features.extend([f]*3);borders.extend(range(3));types.extend([0]*3)
    for f,column in enumerate((1,3),2):
        hashes=cat_feature_hashes(x[:,column]);unique=np.unique(hashes)
        banks.append(np.searchsorted(unique,hashes).astype(np.uint8))
        features.extend([f]*len(unique));borders.extend(range(len(unique)));types.extend([1]*len(unique))
    loss,_,parameter=config['loss_function'].partition(':')
    keys=('iterations','depth','learning_rate','score_function','leaf_estimation_method',
          'leaf_estimation_iterations','bootstrap_type','random_seed','random_strength',
          'leaf_estimation_backtracking','min_fold_size','fold_len_multiplier','fold_permutation_block')
    args={name:config[name] for name in keys}
    args.update({name:config[name] for name in ('subsample','bagging_temperature','l2_leaf_reg') if name in config})
    return _ordered.train(np.array(banks),y,np.uint32(features),np.uint32(borders),
        candidate_types=np.uint8(types),sample_weight=w,initial_predictions=baseline,
        objective=loss,objective_param=float(parameter.split('=')[1]) if parameter else None,
        permutation_count=1 if config['has_time'] else config['permutation_count'],group_sizes=group_sizes,**args)


@pytest.mark.parametrize('loss,method',OBJECTIVES)
@pytest.mark.parametrize('score',['Cosine','NewtonCosine'])
def test_scalar_objectives_match_typed_runtime_and_readers(tmp_path,loss,method,score):
    cls,x,y,w,config,numeric=problem(loss,method,count=1)
    config.update(has_time=True,score_function=score)
    baseline=np.linspace(-.1,.1,len(y),dtype=np.float32)
    pool=Pool(x,y,cat_features=[1,3],weight=w,baseline=baseline)
    borders=tmp_path/'borders.tsv'
    borders.write_text(''.join(f'{f}\t{b+.5}\n' for f in (0,2) for b in range(3)))
    pool.quantize(input_borders=str(borders))
    model=cls().set_params(**config).fit(pool,eval_set=pool,use_best_model=False)
    expected=resident(config,x,y,w,numeric,baseline)
    assert model.get_metadata()['metal_backend']=='METAL'
    assert model.get_all_params()['boosting_type']=='Ordered'
    np.testing.assert_array_equal(model.get_tree_leaf_counts(),2**expected.depths)
    np.testing.assert_allclose(model.get_test_eval(),expected.predictions,atol=7e-7,rtol=5e-6)
    values=np.concatenate([expected.leaf_values[i,:1<<int(d)] for i,d in enumerate(expected.depths)])
    np.testing.assert_allclose(model.get_leaf_values(),values,atol=7e-7,rtol=5e-6)
    x[::13,1]='unseen'
    if expected.split_types.any():assert_readers(model,cls,x,tmp_path)
    else:assert np.isfinite(model.predict(x,task_type='GPU')).all()


@pytest.mark.parametrize('kind',['No','Bayesian','Bernoulli','Poisson','MVS'])
@pytest.mark.parametrize('count',[1,4])
@pytest.mark.parametrize('quantized',[False,True])
def test_all_samplers_recover_prefixes_and_original_category_identity(tmp_path,kind,count,quantized):
    cls,x,y,w,config,_=problem(kind=kind,count=count)
    config.update(random_strength=.7,one_hot_max_size=6)
    pool=Pool(x,y,cat_features=[1,3],weight=w)
    if quantized:pool.quantize()
    heldout=x.copy();heldout[::17,1]='unseen'
    evaluation=Pool(heldout,y,cat_features=[1,3],weight=w)
    direct=cls().set_params(**config).fit(pool,eval_set=evaluation,use_best_model=False)
    saved=config|dict(save_snapshot=True,snapshot_interval=0,snapshot_file='ordered.snapshot',
                      allow_writing_files=True,train_dir=str(tmp_path))
    partial=cls().set_params(**saved).fit(pool,eval_set=evaluation,use_best_model=False,callbacks=[StopAfter(2)])
    assert partial.tree_count_==2
    resumed=cls().set_params(**saved).fit(pool,eval_set=evaluation,use_best_model=False)
    for name in ('get_leaf_values','get_leaf_weights','get_test_eval'):
        np.testing.assert_array_equal(getattr(resumed,name)(),getattr(direct,name)())
    assert resumed.evals_result_==direct.evals_result_
    assert_readers(resumed,cls,heldout,tmp_path)
    changed=x.copy();changed[0,1]='changed-category'
    with pytest.raises(CatBoostError,match='(?i)snapshot.*differ|differ.*snapshot'):
        cls().set_params(**saved).fit(Pool(changed,y,cat_features=[1,3],weight=w),eval_set=evaluation,use_best_model=False)


@pytest.mark.parametrize('mode',['No','AnyImprovement','Armijo'])
@pytest.mark.parametrize('count',[1,4])
def test_binary_backtracking_initial_model_baseline_and_resume(tmp_path,mode,count):
    cls,x,y,w,config,_=problem('Logloss',kind='Bernoulli',count=count)
    config['leaf_estimation_backtracking']=mode
    pool=Pool(x,y,cat_features=[1,3],weight=w)
    initial=cls().set_params(**(config|dict(iterations=2))).fit(pool)
    pool.set_baseline(np.linspace(-.1,.2,len(y),dtype=np.float32))
    direct=cls().set_params(**config).fit(pool,init_model=initial,eval_set=pool,use_best_model=False)
    saved=config|dict(save_snapshot=True,snapshot_interval=0,snapshot_file='initial.snapshot',
                      allow_writing_files=True,train_dir=str(tmp_path))
    partial=cls().set_params(**saved).fit(pool,init_model=initial,eval_set=pool,use_best_model=False,callbacks=[StopAfter(2)])
    assert partial.tree_count_==4
    resumed=cls().set_params(**saved).fit(pool,init_model=initial,eval_set=pool,use_best_model=False)
    for name in ('get_leaf_values','get_leaf_weights','get_test_eval'):
        np.testing.assert_array_equal(getattr(resumed,name)(),getattr(direct,name)())
    assert resumed.evals_result_==direct.evals_result_
    np.testing.assert_allclose(direct.get_test_eval(),direct.predict(x,prediction_type='RawFormulaVal',task_type='GPU')+
                               pool.get_baseline().ravel(),atol=7e-7,rtol=5e-6)


@pytest.mark.parametrize('count',[1,4])
def test_onall_threshold_constant_columns_and_ignored_ctrs(tmp_path,count):
    cls,x,y,w,config,_=problem(count=count)
    heldout=x.copy();heldout[0,1]='unseen'
    pool=Pool(x,y,cat_features=[1,3],weight=w)
    evaluation=Pool(heldout,y,cat_features=[1,3],weight=w)
    model=cls().set_params(**config).fit(pool,eval_set=evaluation,use_best_model=False)
    assert model.get_metadata()['metal_permutations']==str(count)
    for limit in (1,4):
        ctr_model=cls().set_params(**(config|dict(one_hot_max_size=limit))).fit(pool,eval_set=evaluation,use_best_model=False)
        np.testing.assert_allclose(ctr_model.get_test_eval(),ctr_model.predict(heldout,task_type='GPU'),atol=7e-7,rtol=5e-6)
        path=tmp_path/f'ctr-limit-{limit}.json';ctr_model.save_model(path,format='json')
        assert 'OnlineCtr' in path.read_text()
    ignored=cls().set_params(**(config|dict(one_hot_max_size=1,ignored_features=[1,3]))).fit(pool)
    assert np.isfinite(ignored.predict(x,task_type='GPU')).all()
    cats=np.column_stack((np.repeat('constant',len(x)),x[:,1],x[:,3]))
    pure=cls().set_params(**config).fit(Pool(cats,y,cat_features=[0,1,2],weight=w))
    assert_readers(pure,cls,cats,tmp_path)


@pytest.mark.parametrize('count',[1,4])
def test_class_labels_weights_and_best_model(count):
    cls,x,y,w,config,_=problem('Logloss',count=count)
    labels=np.where(y,'positive','negative')
    config.update(iterations=9,class_weights={'negative':.7,'positive':1.8},
                  eval_metric='Accuracy',use_best_model=True)
    model=cls().set_params(**config).fit(Pool(x,labels,cat_features=[1,3],weight=w),
                                       eval_set=Pool(x,labels,cat_features=[1,3],weight=w))
    assert model.classes_.tolist()==['negative','positive']
    history=model.evals_result_['validation']['Accuracy']
    assert model.tree_count_==int(np.argmax(history))+1
    np.testing.assert_allclose(model.predict_proba(x,task_type='GPU'),model.predict_proba(x),atol=6e-7,rtol=5e-6)
