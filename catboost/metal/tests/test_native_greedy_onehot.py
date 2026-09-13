"""Native one-hot greedy trees, independent runtime routing and recovery."""
import json
import os
import numpy as np
import pytest
from catboost import CatBoost, CatBoostClassifier, CatBoostRegressor, CatBoostError, Pool
from catboost_metal import _greedy
from catboost_metal._categorical import cat_feature_hashes
from test_native_greedy_api import POLICIES, OBJECTIVES, options, sampler, StopAfter

pytestmark=pytest.mark.skipif(os.environ.get('CATBOOST_NATIVE_METAL_TESTS')!='1',reason='requires rebuilt native greedy one-hot adapter')


@pytest.fixture(autouse=True)
def only_gpu(monkeypatch):
    original=CatBoost._fit
    def checked(self,*args,**kwargs):
        assert self.get_params().get('task_type')=='GPU'
        return original(self,*args,**kwargs)
    monkeypatch.setattr(CatBoost,'_fit',checked)


def problem(policy='Depthwise',loss='RMSE',method='Newton',kind='No'):
    rng=np.random.default_rng(72183);rows=192
    numeric=rng.integers(0,4,(2,rows),dtype=np.uint8)
    code=rng.integers(0,4,rows);other=rng.integers(0,3,rows)
    x=np.empty((rows,4),object)
    x[:,0]=numeric[0].astype(float);x[:,2]=numeric[1].astype(float)
    x[:,1]=np.array(['amber','blue','cyan','黑'])[code];x[:,3]=np.array(['a','b','c'])[other]
    y=np.float32(2.1*(code==1)-1.3*(code==2)+.8*(other==0)+.15*numeric[0]-.1*numeric[1]-.4)
    objective=loss.partition(':')[0];cls=CatBoostRegressor
    if objective in ('Logloss','CrossEntropy'):
        cls=CatBoostClassifier;y=(y>0).astype(np.float32) if objective=='Logloss' else np.float32(1/(1+np.exp(-y)))
    elif objective in ('Poisson','Tweedie','LogLinQuantile','MAPE'):y=np.float32(np.exp(y/3))
    weight=rng.uniform(.3,2,rows).astype(np.float32);weight[::19]=0
    config=options(policy,iterations=4,depth=3,loss_function=loss,leaf_estimation_method=method,
        leaf_estimation_iterations=3,one_hot_max_size=5,has_time=True,permutation_count=4,**sampler(kind))
    if policy=='Lossguide':config['max_leaves']=5
    return cls,x,y,weight,config,numeric


def resident(config,x,y,weights,numeric,baseline=None):
    banks=list(numeric);features=[];borders=[];types=[]
    for f in range(2):features.extend([f]*3);borders.extend(range(3));types.extend([0]*3)
    for f,column in enumerate((1,3),2):
        hashes=cat_feature_hashes(x[:,column]);unique=np.unique(hashes)
        banks.append(np.searchsorted(unique,hashes).astype(np.uint8))
        features.extend([f]*len(unique));borders.extend(range(len(unique)));types.extend([1]*len(unique))
    loss,_,parameters=config['loss_function'].partition(':')
    args={name:config[name] for name in ('iterations','depth','grow_policy','learning_rate','score_function',
        'leaf_estimation_method','leaf_estimation_iterations','bootstrap_type','random_seed','random_strength','leaf_estimation_backtracking')}
    for name in ('max_leaves','bagging_temperature','subsample','l2_leaf_reg'):
        if name in config:args[name]=config[name]
    return _greedy.train(np.array(banks),y,np.array(features,np.uint32),np.array(borders,np.uint32),
        candidate_types=np.array(types,np.uint8),objective=loss,
        objective_param=float(parameters.split('=')[1]) if parameters else None,
        sample_weight=weights,initial_predictions=baseline,**args)


def assert_readers(model,cls,x,tmp_path):
    expected=model.predict(x,prediction_type='RawFormulaVal')
    for fmt in ('cbm','json'):
        p=tmp_path/('onehot.'+fmt);model.save_model(p,format=fmt)
        loaded=cls().load_model(p,format=fmt)
        np.testing.assert_allclose(loaded.predict(x,prediction_type='RawFormulaVal',task_type='GPU'),expected,atol=4e-7,rtol=4e-6)
        if fmt=='json':assert 'OneHotFeature' in p.read_text()


@pytest.mark.parametrize('policy',POLICIES)
@pytest.mark.parametrize('loss,method',OBJECTIVES)
def test_registered_scalar_losses_match_fixed_onehot_runtime_forests(tmp_path,policy,loss,method):
    cls,x,y,w,config,numeric=problem(policy,loss,method)
    baseline=np.linspace(-.1,.1,len(y),dtype=np.float32)
    pool=Pool(x,y,cat_features=[1,3],weight=w,baseline=baseline)
    path=tmp_path/'borders.tsv';path.write_text(''.join(f'{f}\t{b+.5}\n' for f in (0,2) for b in range(3)))
    pool.quantize(input_borders=str(path))
    model=cls().set_params(**config).fit(pool,eval_set=pool,use_best_model=False)
    expected=resident(config,x,y,w,numeric,baseline)
    assert model.get_metadata()['metal_permutations']=='1'
    np.testing.assert_array_equal(model.get_tree_leaf_counts(),[len(t.leaf_values) for t in expected.trees])
    np.testing.assert_allclose(model.get_test_eval(),expected.predictions,atol=5e-7,rtol=4e-6)
    at=0
    for tree in expected.trees:
        count=len(tree.leaf_values)
        np.testing.assert_allclose(np.sort(model.get_leaf_values()[at:at+count]),np.sort(tree.leaf_values),atol=5e-7,rtol=4e-6)
        np.testing.assert_allclose(np.sort(model.get_leaf_weights()[at:at+count]),np.sort(tree.leaf_weights),atol=2e-5,rtol=4e-6)
        at+=count
    x[::13,1]='unseen';assert_readers(model,cls,x,tmp_path)


@pytest.mark.parametrize('policy',POLICIES)
@pytest.mark.parametrize('kind',['No','Bayesian','Bernoulli','Poisson'])
@pytest.mark.parametrize('quantized',[False,True])
def test_raw_and_quantized_onehot_snapshots_metrics_and_changed_hashes(tmp_path,policy,kind,quantized):
    cls,x,y,w,config,_=problem(policy,kind=kind);config.update(has_time=False,random_strength=.7,one_hot_max_size=6)
    pool=Pool(x,y,cat_features=[1,3],weight=w)
    if quantized:pool.quantize()
    heldout=x.copy();heldout[::17,1]='unseen'
    evaluation=Pool(heldout,y,cat_features=[1,3],weight=w)
    direct=cls().set_params(**config).fit(pool,eval_set=evaluation,use_best_model=False)
    saved=config|dict(save_snapshot=True,snapshot_interval=0,snapshot_file='onehot.snapshot',allow_writing_files=True,train_dir=str(tmp_path))
    assert cls().set_params(**saved).fit(pool,eval_set=evaluation,use_best_model=False,callbacks=[StopAfter(2)]).tree_count_==2
    resumed=cls().set_params(**saved).fit(pool,eval_set=evaluation,use_best_model=False)
    for name in ('get_leaf_values','get_leaf_weights','get_test_eval'):
        np.testing.assert_array_equal(getattr(resumed,name)(),getattr(direct,name)())
    assert resumed.evals_result_==direct.evals_result_
    assert_readers(direct,cls,heldout,tmp_path)
    changed=x.copy();changed[0,1]='changed-category'
    with pytest.raises(CatBoostError,match='(?i)snapshot.*differ|differ.*snapshot'):
        cls().set_params(**saved).fit(Pool(changed,y,cat_features=[1,3],weight=w),eval_set=evaluation,use_best_model=False)


@pytest.mark.parametrize('policy',POLICIES)
@pytest.mark.parametrize('loss',['Quantile:alpha=0.7','MAE','MAPE'])
def test_exact_leaves_match_resident_weighted_residuals(tmp_path,policy,loss):
    cls,x,y,w,config,numeric=problem(policy,loss,'Exact','Bernoulli')
    pool=Pool(x,y,cat_features=[1,3],weight=w)
    path=tmp_path/'borders.tsv';path.write_text(''.join(f'{f}\t{b+.5}\n' for f in (0,2) for b in range(3)))
    pool.quantize(input_borders=str(path));model=cls().set_params(**config).fit(pool,eval_set=pool,use_best_model=False)
    expected=resident(config,x,y,w,numeric)
    np.testing.assert_allclose(model.get_test_eval(),expected.predictions,atol=7e-7,rtol=5e-6)
    assert_readers(model,cls,x,tmp_path)


@pytest.mark.parametrize('policy',POLICIES)
@pytest.mark.parametrize('mode',['No','AnyImprovement','Armijo'])
def test_binary_backtracking_with_categories_initial_model_baseline_and_resume(tmp_path,policy,mode):
    cls,x,y,w,config,_=problem(policy,'Logloss',kind='Bernoulli');config['leaf_estimation_backtracking']=mode
    pool=Pool(x,y,cat_features=[1,3],weight=w)
    initial=cls().set_params(**(config|dict(iterations=2))).fit(pool)
    pool.set_baseline(np.linspace(-.1,.2,len(y),dtype=np.float32))
    direct=cls().set_params(**config).fit(pool,init_model=initial,eval_set=pool,use_best_model=False)
    saved=config|dict(save_snapshot=True,snapshot_interval=0,snapshot_file='initial.snapshot',allow_writing_files=True,train_dir=str(tmp_path))
    partial=cls().set_params(**saved).fit(pool,init_model=initial,eval_set=pool,use_best_model=False,callbacks=[StopAfter(2)])
    assert partial.tree_count_==4
    resumed=cls().set_params(**saved).fit(pool,init_model=initial,eval_set=pool,use_best_model=False)
    for name in ('get_leaf_values','get_leaf_weights','get_test_eval'):
        np.testing.assert_array_equal(getattr(resumed,name)(),getattr(direct,name)())
    assert resumed.evals_result_==direct.evals_result_
    np.testing.assert_allclose(direct.get_test_eval(),direct.predict(x,prediction_type='RawFormulaVal',task_type='GPU')+pool.get_baseline().ravel(),atol=5e-7,rtol=4e-6)


@pytest.mark.parametrize('policy',POLICIES)
def test_eval_hash_union_threshold_and_ignored_categorical_columns(policy):
    cls,x,y,w,config,_=problem(policy);config['has_time']=False
    pool=Pool(x,y,cat_features=[1,3],weight=w);heldout=x.copy();heldout[0,1]='unseen'
    evaluation=Pool(heldout,y,cat_features=[1,3],weight=w)
    model=cls().set_params(**config).fit(pool,eval_set=evaluation)
    assert model.get_metadata()['metal_permutations']=='1'
    for limit in (1,4):
        ctr=cls().set_params(**(config|dict(one_hot_max_size=limit))).fit(pool,eval_set=evaluation)
        assert ctr.get_metadata()['metal_permutations']=='4'
        assert np.isfinite(ctr.predict(heldout,task_type='GPU')).all()
    ignored=cls().set_params(**(config|dict(one_hot_max_size=1,ignored_features=[1,3]))).fit(pool)
    assert np.isfinite(ignored.predict(x,task_type='GPU')).all()


@pytest.mark.parametrize('policy',POLICIES)
def test_pure_categorical_and_constant_columns_keep_original_feature_indices(tmp_path,policy):
    cls,x,y,w,config,_=problem(policy)
    cats=np.column_stack((np.repeat('constant',len(x)),x[:,1],x[:,3]))
    model=cls().set_params(**config).fit(Pool(cats,y,cat_features=[0,1,2],weight=w))
    assert_readers(model,cls,cats,tmp_path)
    with pytest.raises(CatBoostError,match='one_hot_max_size up to 255'):
        cls().set_params(**(config|dict(one_hot_max_size=256))).fit(Pool(cats,y,cat_features=[0,1,2],weight=w))
