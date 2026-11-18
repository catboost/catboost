"""Public task_type=GPU YetiRankPairwise fitting and normal model lifecycle."""
import os
import numpy as np
import pytest
from catboost import CatBoostRanker,CatBoostError,Pool
from test_yeti_pair_training import problem,oracle
from test_native_pair_api import StopAfter

pytestmark=pytest.mark.skipif(os.environ.get('CATBOOST_NATIVE_METAL_QUERY_TESTS')!='1',reason='rebuilt native Metal adapter required')


def inputs(method='Simple',kind='No',unit='Object',score='NewtonL2'):
    args=problem(leaf_estimation_method=method,bootstrap_type=kind,sampling_unit=unit,score_function=score)
    args['candidate_types'].fill(0);x=args['bins'].T.astype(np.float32)
    groups=np.repeat(np.arange(len(args['group_offsets'])-1),np.diff(args['group_offsets']))
    pool=Pool(x,args['targets'],group_id=groups,weight=args['sample_weight'],baseline=args['initial_predictions'])
    options=dict(task_type='GPU',loss_function='YetiRankPairwise:permutations=7;decay=0.85',iterations=3,depth=3,
        learning_rate=args['learning_rate'],l2_leaf_reg=args['l2_leaf_reg'],bayesian_matrix_reg=args['non_diagonal_regularization'],
        leaf_estimation_method=method,leaf_estimation_iterations=1,score_function=score,bootstrap_type=kind,sampling_unit=unit,
        random_seed=args['random_seed'],random_strength=0.,has_time=True,verbose=False,allow_writing_files=False,metric_period=1)
    if kind=='Bernoulli':options['subsample']=args['subsample']
    if kind=='Bayesian':options['bagging_temperature']=args['bagging_temperature']
    return args,x,pool,options


@pytest.mark.parametrize('method',['Simple','Newton','Gradient'])
@pytest.mark.parametrize('kind,unit',[('No','Object'),('Bayesian','Object'),('Bernoulli','Object'),('Bernoulli','Group')])
@pytest.mark.parametrize('score',['L2','Cosine','NewtonL2','NewtonCosine','SolarL2','LOOL2','SatL2'])
def test_native_yeti_pair_matches_independent_generated_pair_forests(tmp_path,method,kind,unit,score):
    args,x,pool,options=inputs(method,kind,unit,score)
    borders=tmp_path/'borders.tsv';borders.write_text(''.join(f'{f}\t{b+.5}\n' for f in range(len(args['bins'])) for b in range(3)))
    pool.quantize(input_borders=str(borders));model=CatBoostRanker().set_params(**options).fit(pool,eval_set=pool,use_best_model=False)
    cursor=args['initial_predictions'].copy();values=[];weights=[]
    for iteration in range(3):
        _,ids,v,w=oracle(args,cursor,iteration);values.append(v);weights.append(w);cursor=np.float32(cursor+v[ids])
    np.testing.assert_allclose(model.get_leaf_values(),np.concatenate(values),rtol=5e-4,atol=8e-6)
    np.testing.assert_allclose(model.get_leaf_weights(),np.concatenate(weights),rtol=5e-5,atol=4e-6)
    np.testing.assert_allclose(model.predict(x,task_type='GPU')+args['initial_predictions'],cursor,rtol=5e-4,atol=8e-6)
    assert model.get_metadata()['metal_yeti_pair_rng']=='item_iteration_domains_v1'
    assert len(model.evals_result_['validation']['PFound'])==3


@pytest.mark.parametrize('method',['Simple','Newton','Gradient'])
@pytest.mark.parametrize('unit',['Object','Group'])
def test_native_yeti_pair_snapshot_metrics_and_model_roundtrip(tmp_path,method,unit):
    _,x,pool,options=inputs(method,'Bernoulli',unit);options.update(iterations=6,leaf_estimation_iterations=1 if method=='Simple' else 3)
    direct=CatBoostRanker().set_params(**options).fit(pool,eval_set=pool,use_best_model=False)
    saved=options|dict(save_snapshot=True,snapshot_interval=0,snapshot_file='pfound.snapshot',allow_writing_files=True,train_dir=str(tmp_path))
    partial=CatBoostRanker().set_params(**saved).fit(pool,eval_set=pool,use_best_model=False,callbacks=[StopAfter()]);assert partial.tree_count_==3
    resumed=CatBoostRanker().set_params(**saved).fit(pool,eval_set=pool,use_best_model=False)
    for attr in ('get_leaf_values','get_leaf_weights','get_test_eval'):np.testing.assert_array_equal(getattr(resumed,attr)(),getattr(direct,attr)())
    assert resumed.evals_result_==direct.evals_result_
    for fmt in ('cbm','json'):
        path=tmp_path/('pfound.'+fmt);resumed.save_model(str(path),format=fmt);loaded=CatBoostRanker().load_model(str(path),format=fmt)
        if fmt=='cbm':np.testing.assert_array_equal(loaded.predict(x,task_type='GPU'),direct.predict(x,task_type='GPU'))
        else:
            # JSON's decimal-to-double conversion can move a promoted leaf by
            # one double ULP; verify its original float32 value remains exact.
            np.testing.assert_array_equal(loaded.get_leaf_values().astype(np.float32),direct.get_leaf_values().astype(np.float32))
            np.testing.assert_allclose(loaded.predict(x,task_type='GPU'),direct.predict(x,task_type='GPU'),rtol=1e-13,atol=1e-16)
        np.testing.assert_allclose(loaded.get_leaf_weights(),direct.get_leaf_weights(),atol=1e-10,rtol=1e-10)


def test_native_empty_weak_target_simple_and_invalid_options():
    _,x,pool,options=inputs('Simple','Bernoulli','Group');options['subsample']=1e-12
    model=CatBoostRanker().set_params(**options).fit(pool,eval_set=pool,use_best_model=False)
    np.testing.assert_array_equal(model.get_leaf_values(),0);np.testing.assert_array_equal(model.get_leaf_weights(),0)
    np.testing.assert_array_equal(model.predict(x,task_type='GPU'),0)
    for extra in ({'depth':0},{'depth':9},{'leaf_estimation_iterations':2},{'bootstrap_type':'Poisson'},{'bootstrap_type':'MVS'},{'leaf_estimation_backtracking':'Armijo'}):
        with pytest.raises(CatBoostError):CatBoostRanker().set_params(**(options|extra)).fit(pool)
