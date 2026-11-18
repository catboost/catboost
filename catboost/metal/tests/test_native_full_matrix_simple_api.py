"""Native Simple GPU fitting against independent sampled-matrix forests."""
import os
import numpy as np
import pytest
from catboost import CatBoostRanker,CatBoostError,Pool
from test_full_matrix_simple import problem,oracle
from test_native_pair_api import StopAfter

pytestmark=pytest.mark.skipif(os.environ.get('CATBOOST_NATIVE_METAL_QUERY_TESTS')!='1',reason='rebuilt native Metal adapter required')


def inputs(target,score='NewtonL2',kind='No'):
    args=problem(target,score,kind)
    if target=='pair':
        # Native Pool preparation flattens competitors in winner order. Edge
        # sampling is indexed in that prepared order, retaining duplicate order.
        order=np.argsort(args['pair_winners'],kind='stable')
        for key in ('pair_winners','pair_losers','pair_weights'):args[key]=args[key][order]
    x=args['bins'].T.astype(np.float32)
    options=dict(task_type='GPU',loss_function='PairLogitPairwise',iterations=3,depth=3,learning_rate=args['learning_rate'],
        l2_leaf_reg=args['l2_leaf_reg'],bayesian_matrix_reg=args['non_diagonal_regularization'],
        score_function=score,bootstrap_type=kind,leaf_estimation_method='Simple',leaf_estimation_iterations=1,
        random_seed=args['random_seed'],random_strength=0.,has_time=True,verbose=False,allow_writing_files=False,metric_period=1)
    if kind!='No':options['subsample']=.7
    kw=dict(weight=args['sample_weight'],baseline=args['initial_predictions'])
    if target=='pair':
        y=np.linspace(0,1,len(x));groups=np.zeros(len(x),int)
        kw.update(pairs=np.column_stack((args['pair_winners'],args['pair_losers'])),pairs_weight=args['pair_weights'])
    else:
        y=args['targets'];offsets=args['group_offsets'];groups=np.repeat(np.arange(len(offsets)-1),np.diff(offsets))
        scale=' '.join(f'{b-a},{int((y[a:b]>.5).sum())}:{float(value):.9g}' for a,b,value in zip(offsets[:-1],offsets[1:],args['query_scales']))
        options['loss_function']=f"QueryCrossEntropy:alpha={args['alpha']};raw_values_scale={scale}"
    return args,x,Pool(x,y,group_id=groups,**kw),options


@pytest.mark.parametrize('target,score',[
    *[('pair',s) for s in ('L2','Cosine','NewtonL2','NewtonCosine','SolarL2','LOOL2','SatL2')],
    *[('qce',s) for s in ('Cosine','NewtonL2','NewtonCosine','SolarL2','LOOL2','SatL2')]])
@pytest.mark.parametrize('kind',['No','Bernoulli'])
def test_native_simple_matches_independent_matrix_forest_and_diagonal_weights(tmp_path,target,score,kind):
    args,x,pool,options=inputs(target,score,kind)
    borders=tmp_path/'borders.tsv';borders.write_text(''.join(f'{f}\t{b+.5}\n' for f in range(len(args['bins'])) for b in range(3)))
    pool.quantize(input_borders=str(borders))
    model=CatBoostRanker().set_params(**options).fit(pool)
    cursor=args['initial_predictions'].copy();values=[];weights=[]
    for iteration in range(3):
        _,ids,v,w=oracle(args,cursor,iteration,target);values.append(v);weights.append(w);cursor=np.float32(cursor+v[ids])
    np.testing.assert_allclose(model.get_leaf_values(),np.concatenate(values),rtol=5e-4,atol=8e-6)
    np.testing.assert_allclose(model.get_leaf_weights(),np.concatenate(weights),rtol=4e-5,atol=4e-6)
    np.testing.assert_allclose(model.predict(x,task_type='GPU')+args['initial_predictions'],cursor,rtol=5e-4,atol=8e-6)
    assert model.get_all_params()['leaf_estimation_method']=='Simple'
    assert model.get_metadata()['metal_backend']=='METAL'


@pytest.mark.parametrize('target',['pair','qce'])
@pytest.mark.parametrize('mode',['No','AnyImprovement','Armijo'])
def test_native_simple_snapshot_and_normal_model_roundtrip(tmp_path,target,mode):
    args,x,pool,options=inputs(target,kind='Bernoulli');options.update(iterations=6,leaf_estimation_backtracking=mode)
    direct=CatBoostRanker().set_params(**options).fit(pool,eval_set=pool,use_best_model=False)
    saved=options|dict(save_snapshot=True,snapshot_interval=0,snapshot_file='simple.snapshot',allow_writing_files=True,train_dir=str(tmp_path))
    assert CatBoostRanker().set_params(**saved).fit(pool,eval_set=pool,use_best_model=False,callbacks=[StopAfter()]).tree_count_==3
    resumed=CatBoostRanker().set_params(**saved).fit(pool,eval_set=pool,use_best_model=False)
    for attr in ('get_leaf_values','get_leaf_weights','get_test_eval'):
        np.testing.assert_array_equal(getattr(resumed,attr)(),getattr(direct,attr)())
    assert resumed.evals_result_==direct.evals_result_
    for format in ('cbm','json'):
        path=tmp_path/('model.'+format);resumed.save_model(str(path),format=format)
        loaded=CatBoostRanker().load_model(str(path),format=format)
        np.testing.assert_allclose(loaded.predict(x,task_type='GPU'),direct.predict(x),atol=1e-10,rtol=1e-10)
        np.testing.assert_allclose(loaded.get_leaf_weights(),direct.get_leaf_weights(),atol=1e-10,rtol=1e-10)


@pytest.mark.parametrize('target',['pair','qce'])
def test_native_simple_empty_weak_target_and_validation(target):
    args,x,pool,options=inputs(target,kind='Bernoulli');options.update(subsample=1e-12)
    model=CatBoostRanker().set_params(**options).fit(pool,eval_set=pool,use_best_model=False)
    np.testing.assert_array_equal(model.get_leaf_values(),0.)
    np.testing.assert_array_equal(model.get_leaf_weights(),0.)
    np.testing.assert_array_equal(model.predict(x,task_type='GPU'),0.)
    for extra in ({'depth':0},{'leaf_estimation_iterations':2}):
        with pytest.raises(CatBoostError,match='(?i)(Simple|estimation iterations)'):
            CatBoostRanker().set_params(**(options|extra)).fit(pool)
