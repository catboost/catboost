"""Group-preserving Ordered folds, independent forest equations and recovery."""
import numpy as np
import pytest
from catboost_metal import _ordered
from catboost_metal._ordered_rng import cuda_ordered_group_history_order
from cuda_ordered_group_reference import reference_group_order,reference_grouped_descriptors
from cuda_ordered_reference import train_reference
from test_ordered_training import apple_silicon,prohibit_cpu_training,dataset,options,compare
from test_ordered_onehot_runtime import CASES

SIZES=[1,90,2,4,150,3,7,11,250,5,17,1]


def problem(objective='RMSE',parameter=None,method='Newton',categorical=False,**extra):
    bins,y,cf,cb,w=dataset(objective if objective in ('Logloss','CrossEntropy') else 'RMSE',rows=sum(SIZES))
    if objective in ('Poisson','Tweedie','LogLinQuantile','MAPE'):y=np.exp(y*.25).astype(np.float32)
    config=options(objective,iterations=3,depth=3,bias=0,learning_rate=.02 if objective=='Lq' and method=='Gradient' else .07,
        objective_param=parameter,leaf_estimation_method=method,leaf_estimation_iterations=2,
        min_fold_size=16,fold_len_multiplier=1.7,group_sizes=SIZES,sample_weight=w,
        candidate_types=np.uint8(cf==1) if categorical else None,**extra)
    return bins,y,cf,cb,config


@pytest.mark.parametrize('sizes',[SIZES,[3,5,7,11,13,17,19,23]])
@pytest.mark.parametrize('growth',[1.3,1.7,2.0])
@pytest.mark.parametrize('count',[1,4,7])
def test_grouped_descriptors_match_cuda_permutation_specific_boundaries(sizes,growth,count):
    rows=sum(sizes);bins=np.zeros((1,rows),np.uint8);y=np.ones(rows,np.float32)
    expected=reference_grouped_descriptors(sizes,count,growth=growth,min_fold_size=16)
    with _ordered.Session(bins,y,[],[],iterations=1,depth=0,group_sizes=sizes,permutation_count=count,
                          min_fold_size=16,fold_len_multiplier=growth) as session:
        state=session.state()
        np.testing.assert_array_equal(state['descriptors'],expected['descriptors'])
        assert len(state['cursors'])==expected['cursor_count']
        session.step();assert np.isfinite(session.predictions()).all()
    if count==7:assert len(set(map(len,expected['folds'])))>1


@pytest.mark.parametrize('block',[0,3,65,129])
@pytest.mark.parametrize('permutation',[0,1,3])
@pytest.mark.parametrize('sizes',[SIZES,[1,27,3,77]*600])
def test_group_shuffle_uses_document_block_policy(sizes,permutation,block):
    expected=reference_group_order(sizes,permutation,block)[1]
    np.testing.assert_array_equal(cuda_ordered_group_history_order(sizes,permutation,block),expected)


@pytest.mark.parametrize('objective,parameter,method',CASES)
@pytest.mark.parametrize('score',['Cosine','NewtonCosine'])
@pytest.mark.parametrize('count',[1,7])
@pytest.mark.parametrize('categorical',[False,True])
def test_grouped_scalar_forests_match_cuda_equations(objective,parameter,method,score,count,categorical):
    bins,y,cf,cb,config=problem(objective,parameter,method,categorical,score_function=score,permutation_count=count)
    expected=train_reference(bins,y,cf,cb,**config)
    with _ordered.Session(bins,y,cf,cb,**config) as session:
        for _ in range(config['iterations']):session.step()
        actual=session.result();state=session.state()
    compare(actual,expected)
    np.testing.assert_array_equal(actual.split_types,expected['split_types'])
    np.testing.assert_array_equal(state['descriptors'],expected['folds'])
    np.testing.assert_allclose(state['cursors'],expected['cursors'],atol=5e-6,rtol=2e-5)


@pytest.mark.parametrize('objective',['RMSE','Logloss'])
@pytest.mark.parametrize('sampler',['No','Bayesian','Bernoulli','Poisson','MVS'])
@pytest.mark.parametrize('backtracking',['No','AnyImprovement','Armijo'])
@pytest.mark.parametrize('count',[4,7])
@pytest.mark.parametrize('categorical',[False,True])
def test_variable_folds_all_samplers_restore_every_cursor(objective,sampler,backtracking,count,categorical):
    bins,y,cf,cb,config=problem(objective,categorical=categorical,permutation_count=count,
        bootstrap_type=sampler,subsample=.7,random_strength=.8,leaf_estimation_backtracking=backtracking)
    with _ordered.Session(bins,y,cf,cb,**config) as full:
        full.step();saved=full.state();trees=[full.step() for _ in range(2)]
        final=full.state();predictions=full.predictions()
    with _ordered.Session(bins,y,cf,cb,**(config|dict(iterations=2,initial_state=saved))) as restored:
        for expected in trees:
            actual=restored.step()
            for key in ('depth','split_features','split_bins','split_types','leaf_values','leaf_weights','loss'):
                np.testing.assert_array_equal(getattr(actual,key),getattr(expected,key))
        np.testing.assert_array_equal(restored.predictions(),predictions)
        np.testing.assert_array_equal(restored.state()['cursors'],final['cursors'])
        assert restored.state()['mvs_lambda']==final['mvs_lambda']


@pytest.mark.parametrize('sampler',['No','MVS'])
@pytest.mark.parametrize('noise',[0.,1.])
def test_large_final_group_keeps_full_prefix_with_empty_quality(sampler,noise):
    sizes=[1,1,1,997];bins,y,cf,cb,w=dataset(rows=sum(sizes))
    config=options(iterations=3,bias=0,group_sizes=sizes,permutation_count=1,
                   bootstrap_type=sampler,random_strength=noise,subsample=.7,sample_weight=w)
    with _ordered.Session(bins,y,cf,cb,**config) as session:
        np.testing.assert_array_equal(session.state()['descriptors'],[[1000,1000,0,0],[1000,1000,1000,0]])
        for _ in range(3):assert session.step().depth==0
        assert np.isfinite(session.predictions()).all()


@pytest.mark.parametrize('change',['too_few','zero','float','sum','split','row_order','repeat'])
def test_bad_group_contracts_fail_before_opening_gpu(monkeypatch,change):
    bins,y,cf,cb,config=problem()
    if change=='too_few':config['group_sizes']=[1,1,len(y)-2]
    elif change=='zero':config['group_sizes']=[0,1,1,len(y)-2]
    elif change=='float':config['group_sizes']=np.asarray(SIZES,dtype=float)
    elif change=='sum':config['group_sizes']=SIZES[:-1]
    else:
        order=np.arange(len(y),dtype=np.uint32)
        if change=='split':order[1],order[91]=order[91],order[1]
        elif change=='row_order':order[1:91]=order[1:91][::-1]
        else:order[1]=order[2]
        config['permutations']=order[None]
    monkeypatch.setattr(_ordered,'_load',lambda *a:pytest.fail('Invalid groups opened GPU'))
    with pytest.raises(ValueError):_ordered.Session(bins,y,cf,cb,**config)


def test_group_layout_snapshot_identity_precedes_gpu(monkeypatch):
    bins,y,cf,cb,config=problem()
    with _ordered.Session(bins,y,cf,cb,**config) as session:session.step();saved=session.state()
    sizes=SIZES.copy();sizes[1]-=1;sizes[2]+=1;config['group_sizes']=sizes
    monkeypatch.setattr(_ordered,'_load',lambda *a:pytest.fail('Mismatched groups opened GPU'))
    with pytest.raises(ValueError,match='match'):_ordered.Session(bins,y,cf,cb,**config,initial_state=saved)


def test_singleton_groups_retain_numeric_snapshot_identity():
    bins,y,cf,cb,w=dataset();config=options(sample_weight=w,permutation_count=4)
    with _ordered.Session(bins,y,cf,cb,**config) as numeric:
        first=numeric.step();saved=numeric.state();expected=numeric.step()
    with _ordered.Session(bins,y,cf,cb,**config,group_sizes=np.ones(len(y),np.uint32),initial_state=saved) as grouped:
        actual=grouped.step()
        np.testing.assert_array_equal(actual.leaf_values,expected.leaf_values)


@pytest.mark.parametrize('growth',[1.3,1.000000001])
def test_group_boundaries_use_double_growth_without_rounding_to_float(growth):
    sizes=[1,9,3,4,13];rows=sum(sizes)
    expected=reference_grouped_descriptors(sizes,1,growth=growth)
    with _ordered.Session(np.zeros((1,rows),np.uint8),np.ones(rows,np.float32),[],[],
                          iterations=1,depth=0,group_sizes=sizes,fold_len_multiplier=growth) as session:
        np.testing.assert_array_equal(session.state()['descriptors'],expected['descriptors'])
        if growth==1.3:
            # float32(1.3)*10 lies below 13 and would round to the wrong group.
            assert session.state()['descriptors'][0,1]==17
