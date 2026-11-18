"""Native generated-pair training above the previous 24-bit dense-ID limit."""
import os
import numpy as np
import pytest
from catboost import CatBoostRanker,Pool
from catboost.utils import eval_metric
from catboost_metal import _yeti_pair
from test_native_pair_api import StopAfter

pytestmark=pytest.mark.skipif(os.environ.get('CATBOOST_NATIVE_METAL_QUERY_TESTS')!='1',reason='requires rebuilt native Metal adapter')

@pytest.mark.parametrize('method',['Simple','Newton','Gradient'])
@pytest.mark.parametrize('kind,unit',[('No','Object'),('Bayesian','Object'),('Bernoulli','Object'),('Bernoulli','Group')])
def test_large_logical_pair_ids_train_resume_and_export_on_gpu(tmp_path,method,kind,unit):
    groups=33;n=groups*1023;rng=np.random.default_rng(64329)
    bins=rng.integers(0,4,(3,n),dtype=np.uint8);x=bins.T.astype(np.float32)
    y=np.float32(.1+.2*x[:,0]+.07*x[:,1]);w=rng.uniform(.2,2,n).astype(np.float32)
    group=np.repeat(np.arange(groups),1023);offsets=np.arange(groups+1,dtype=np.uint32)*1023
    pool=Pool(x,y,weight=w,group_id=group)
    borders=tmp_path/'borders.tsv';borders.write_text(''.join(f'{f}\t{b+.5}\n' for f in range(3) for b in range(3)))
    pool.quantize(input_borders=str(borders))
    options=dict(task_type='GPU',loss_function='YetiRankPairwise:permutations=1',iterations=4,depth=2,
        learning_rate=.13,l2_leaf_reg=3.,bayesian_matrix_reg=.2,leaf_estimation_method=method,
        leaf_estimation_iterations=1 if method=='Simple' else 3,bootstrap_type=kind,sampling_unit=unit,random_seed=74831,
        score_function='NewtonL2',random_strength=0,has_time=True,verbose=False,allow_writing_files=False)
    if kind=='Bernoulli':options['subsample']=.7
    if kind=='Bayesian':options['bagging_temperature']=.7
    direct=CatBoostRanker().set_params(**options).fit(pool,eval_set=pool,use_best_model=False)
    saved=options|dict(save_snapshot=True,snapshot_interval=0,snapshot_file='wide.snapshot',allow_writing_files=True,train_dir=str(tmp_path))
    partial=CatBoostRanker().set_params(**saved).fit(pool,eval_set=pool,use_best_model=False,callbacks=[StopAfter()]);assert partial.tree_count_==3
    resumed=CatBoostRanker().set_params(**saved).fit(pool,eval_set=pool,use_best_model=False)
    for attr in ('get_leaf_values','get_leaf_weights','get_test_eval'):np.testing.assert_array_equal(getattr(resumed,attr)(),getattr(direct,attr)())
    assert resumed.evals_result_==direct.evals_result_
    # Fix quantization and compare native registration/orchestration with the
    # standalone resident path, whose high-ID raw targets have separate probes.
    private_options=dict(iterations=4,depth=2,learning_rate=.13,l2_leaf_reg=3.,non_diagonal_regularization=.2,
        leaf_estimation_method=method,leaf_estimation_iterations=options['leaf_estimation_iterations'],
        bootstrap_type=kind,sampling_unit=unit,random_seed=74831,random_strength=0,score_function='NewtonL2',permutations=1,
        sample_weight=w,group_offsets=offsets,subsample=.7,bagging_temperature=.7)
    features=np.repeat(np.arange(3,dtype=np.uint32),3);cuts=np.tile(np.arange(3,dtype=np.uint32),3)
    with _yeti_pair.TrainingSession(bins,y,features,cuts,**private_options) as session:
        for _ in range(4):session.step()
        reference=session.result()
    np.testing.assert_allclose(direct.get_leaf_values(),reference.leaf_values.ravel(),rtol=2e-6,atol=1e-7)
    np.testing.assert_allclose(direct.predict(x,task_type='GPU'),reference.predictions,rtol=2e-6,atol=1e-7)
    expected=eval_metric(y,direct.predict(x,task_type='GPU'),'PFound',group_id=group,
        group_weight=np.repeat(w[::1023],1023),thread_count=1)[0]
    assert direct.evals_result_['validation']['PFound'][-1]==pytest.approx(expected,abs=2e-7)
    path=tmp_path/'wide.cbm';direct.save_model(path);loaded=CatBoostRanker().load_model(path)
    np.testing.assert_array_equal(loaded.predict(x,task_type='GPU'),direct.predict(x,task_type='GPU'))
