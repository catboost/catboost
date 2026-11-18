"""Native GPU vector greedy models, serialization, metrics and recovery."""
import os
import numpy as np
import pytest
from catboost import CatBoost,CatBoostClassifier,CatBoostRegressor,CatBoostError,Pool
from catboost_metal import _multiclass
from test_greedy_vector_training import problem as private_problem,OBJECTIVES,SCORES
from test_native_multiclass import StopAfter
from test_greedy_training import POLICIES

pytestmark=pytest.mark.skipif(os.environ.get('CATBOOST_NATIVE_METAL_TESTS')!='1',reason='requires rebuilt vector greedy adapter')

@pytest.fixture(autouse=True)
def only_gpu(monkeypatch):
    original=CatBoost._fit
    def checked(self,*args,**kwargs):
        assert self.get_params().get('task_type')=='GPU'
        return original(self,*args,**kwargs)
    monkeypatch.setattr(CatBoost,'_fit',checked)


def inputs(objective='MultiClass',policy='Depthwise',category='numeric',count=4,**extra):
    args,banks,initial=private_problem(objective,policy)
    x=banks[0].T.astype(float);y=args['targets'];w=args['sample_weight'];cats=[]
    if category!='numeric':
        x=x.astype(object);x[:,2]=np.array(['c'+str(int(v)) for v in banks[0,2]]);cats=[2]
    config=dict(task_type='GPU',grow_policy=policy,loss_function=objective,iterations=4,depth=3,
        learning_rate=.19,l2_leaf_reg=2.3,leaf_estimation_method='Newton',leaf_estimation_iterations=3,
        leaf_estimation_backtracking='No',score_function='L2',random_strength=0,random_seed=617735,
        bootstrap_type='No',verbose=False,allow_writing_files=False,boost_from_average=False,
        has_time=count==1,permutation_count=count,border_count=7,one_hot_max_size=8 if category=='onehot' else 1,
        simple_ctr=['Borders:CtrBorderCount=7:Prior=0.5'] if objective!='RMSEWithUncertainty'
                   else ['FloatTargetMeanValue:CtrBorderCount=7:Prior=0.5'])
    if policy=='Lossguide':config['max_leaves']=7
    config.update(extra)
    return CatBoostRegressor if objective=='RMSEWithUncertainty' else CatBoostClassifier,x,y,w,cats,config,args,initial[0]


def raw(model,x,**kwargs):return model.predict(x,prediction_type='RawFormulaVal',**kwargs)


def readers(model,cls,x,path):
    expected=raw(model,x)
    np.testing.assert_allclose(raw(model,x,task_type='GPU'),expected,rtol=4e-6,atol=5e-7)
    for fmt in ('cbm','json'):
        output=path/('vector.'+fmt);model.save_model(output,format=fmt)
        restored=cls().load_model(output,format=fmt)
        np.testing.assert_array_equal(restored.get_tree_leaf_counts(),model.get_tree_leaf_counts())
        # JSON's builder can reorder the physical leaf blocks. Compare each
        # tree's complete vectors and then its predictions, preserving topology.
        dimensions=expected.shape[1]
        a=restored.get_leaf_values().reshape(-1,dimensions);b=model.get_leaf_values().reshape(-1,dimensions);offset=0
        for count in model.get_tree_leaf_counts():
            np.testing.assert_allclose(np.array(sorted(map(tuple,a[offset:offset+count]))),
                np.array(sorted(map(tuple,b[offset:offset+count]))),rtol=3e-16,atol=1e-17)
            offset+=count
        np.testing.assert_allclose(raw(restored,x,task_type='GPU'),expected,rtol=4e-6,atol=5e-7)


@pytest.mark.parametrize('objective',OBJECTIVES)
@pytest.mark.parametrize('policy',POLICIES)
@pytest.mark.parametrize('score',SCORES)
@pytest.mark.parametrize('method',['Newton','Gradient'])
def test_native_numeric_forests_match_resident_vector_training(tmp_path,objective,policy,score,method):
    cls,x,y,w,cats,config,args,initial=inputs(objective,policy,count=1,score_function=score,leaf_estimation_method=method)
    borders=tmp_path/'borders.tsv';borders.write_text(''.join(f'{f}\t{b+.5}\n' for f in range(4) for b in range(7)))
    pool=Pool(x,y,weight=w,baseline=initial);pool.quantize(input_borders=str(borders))
    model=cls().set_params(**config).fit(pool,eval_set=pool,use_best_model=False)
    private=args|dict(iterations=4,score_function=score,leaf_estimation_method=method,candidate_types=np.zeros(28,np.uint8))
    expected=_multiclass.train(**private)
    np.testing.assert_array_equal(model.get_tree_leaf_counts(),[len(t.leaf_weights) for t in expected.trees])
    np.testing.assert_allclose(np.asarray(model.get_test_evals()[0]).T,expected.predictions,rtol=5e-6,atol=1e-6)
    np.testing.assert_allclose(raw(model,x,task_type='GPU')+initial,expected.predictions,rtol=5e-6,atol=1e-6)
    values=model.get_leaf_values().reshape(-1,args['classes']);offset=0
    for tree in expected.trees:
        n=len(tree.leaf_weights)
        # CatBoost packs terminal-child nodes; compare full vector rows by sorted tuples.
        a=np.array(sorted(map(tuple,values[offset:offset+n])));b=np.array(sorted(map(tuple,tree.leaf_values)))
        np.testing.assert_allclose(a,b,rtol=5e-6,atol=1e-6)
        offset+=n
    readers(model,cls,x,tmp_path)


@pytest.mark.parametrize('objective',OBJECTIVES)
@pytest.mark.parametrize('policy',POLICIES)
@pytest.mark.parametrize('category',['numeric','onehot','ctr'])
@pytest.mark.parametrize('sampler',['No','Bayesian','Bernoulli','Poisson'])
def test_original_categories_weights_and_both_cursors_resume_exactly(tmp_path,objective,policy,category,sampler):
    extra=dict(bootstrap_type=sampler,score_function='Cosine',random_strength=.3)
    if sampler in ('Bernoulli','Poisson'):extra['subsample']=.71
    cls,x,y,w,cats,config,args,baseline=inputs(objective,policy,category,**extra)
    names=np.array(['amber','blue','cyan'])[y] if objective!='RMSEWithUncertainty' else y
    pool=Pool(x,names,weight=w,cat_features=cats,baseline=baseline)
    if sampler in ('No','Bernoulli'):pool.quantize()
    evaluation=Pool(x,names,weight=w,cat_features=cats,baseline=baseline)
    direct=cls().set_params(**config).fit(pool,eval_set=evaluation,use_best_model=False)
    assert direct.get_metadata()['metal_permutations']==('4' if category=='ctr' else '1')
    saved=config|dict(save_snapshot=True,snapshot_interval=0,snapshot_file='vector.snapshot',allow_writing_files=True,train_dir=str(tmp_path))
    partial=cls().set_params(**saved).fit(pool,eval_set=evaluation,use_best_model=False,callbacks=[StopAfter(2)])
    assert partial.tree_count_==2
    resumed=cls().set_params(**saved).fit(pool,eval_set=evaluation,use_best_model=False)
    for field in ('get_leaf_values','get_leaf_weights','get_tree_leaf_counts','get_test_evals'):
        np.testing.assert_array_equal(getattr(resumed,field)(),getattr(direct,field)())
    assert resumed.evals_result_==direct.evals_result_
    readers(direct,cls,x,tmp_path)
    np.testing.assert_allclose(np.asarray(direct.get_test_evals()[0]).T,raw(direct,x,task_type='GPU')+baseline,rtol=5e-6,atol=1e-6)
    changed=baseline.copy();changed[0,0]+=1
    with pytest.raises(CatBoostError,match='(?i)snapshot.*differ|differ.*snapshot'):
        cls().set_params(**saved).fit(Pool(x,names,weight=w,cat_features=cats,baseline=changed),eval_set=evaluation,use_best_model=False)


@pytest.mark.parametrize('objective',OBJECTIVES)
@pytest.mark.parametrize('policy',POLICIES)
@pytest.mark.parametrize('mode',['No','AnyImprovement','Armijo'])
def test_initial_model_best_model_and_backtracking_snapshot(tmp_path,objective,policy,mode):
    cls,x,y,w,cats,config,args,baseline=inputs(objective,policy,'ctr',leaf_estimation_backtracking=mode,iterations=6)
    pool=Pool(x,y,weight=w,cat_features=cats)
    initial=cls().set_params(**(config|dict(iterations=2))).fit(pool)
    pool.set_baseline(baseline)
    other=np.roll(y,len(y)//2)
    evaluation=Pool(x,other,weight=w,cat_features=cats,baseline=baseline)
    direct=cls().set_params(**config).fit(pool,init_model=initial,eval_set=evaluation,use_best_model=True)
    saved=config|dict(save_snapshot=True,snapshot_interval=0,snapshot_file='vector-best.snapshot',allow_writing_files=True,train_dir=str(tmp_path))
    cls().set_params(**saved).fit(pool,init_model=initial,eval_set=evaluation,use_best_model=True,callbacks=[StopAfter(2)])
    resumed=cls().set_params(**saved).fit(pool,init_model=initial,eval_set=evaluation,use_best_model=True)
    np.testing.assert_array_equal(resumed.get_leaf_values(),direct.get_leaf_values())
    np.testing.assert_array_equal(resumed.get_test_evals(),direct.get_test_evals())
    assert resumed.evals_result_==direct.evals_result_
    assert resumed.best_iteration_==direct.best_iteration_
    readers(resumed,cls,x,tmp_path)
