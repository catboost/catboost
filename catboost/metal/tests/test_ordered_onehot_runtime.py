"""Typed Ordered candidates: independent fold sums, equations and recovery."""
import numpy as np
import pytest
from catboost_metal import _ordered
from cuda_ordered_reference import train_reference
from test_ordered_training import apple_silicon, prohibit_cpu_training, dataset, options, compare, SCALAR_CASES
from test_ordered_histograms import ordered_histogram_probe, histogram_problem


def direct(data):
    bins=data['bins'];sampled=data['sampled'].astype(float);leaves=data['leaves'];folds=data['folds']
    out=np.zeros((len(data['candidates']),leaves,len(folds),2,4))
    for f,(prefix,end,offset,_) in enumerate(folds):
        rows=data['permutation'][:end];values=sampled[offset:offset+end]
        for c,(feature,packed) in enumerate(data['candidates']):
            border=int(packed)&255
            right=bins[rows,feature]==border if int(packed)>>31 else bins[rows,feature]>border
            ids=2*data['leaf_ids'][rows]+right
            for lo,hi,base in ((0,int(prefix),0),(int(prefix),int(end),2)):
                for component,source in ((base,1),(base+1,0)):
                    out[c,:,f,:,component]=np.bincount(ids[lo:hi],weights=values[lo:hi,source],minlength=2*leaves).reshape(leaves,2)
    return out


@pytest.mark.parametrize('rows',[127,257,1027])
@pytest.mark.parametrize('cache',[0,1,2])
@pytest.mark.parametrize('reuse',[False,True])
@pytest.mark.parametrize('budget',[1<<14,1<<28])
def test_mixed_histograms_and_direct_fallback_equal_independent_sums(ordered_histogram_probe,rows,cache,reuse,budget):
    data=histogram_problem(rows,depth=3)
    candidates=data['candidates'].copy()
    candidates[np.isin(candidates[:,0],[0,2,4]),1]|=np.uint32(1<<31)
    candidates=np.concatenate((candidates,np.array([[0,(1<<31)|255],[4,(1<<31)|1]],np.uint32)))
    data['candidates']=candidates
    expected=direct(data)
    stats=ordered_histogram_probe.run(**data,reuse=reuse,tile_budget=budget,histogram_max_leaves=cache,candidate_batch_size=3)
    actual=stats['statistics']
    np.testing.assert_allclose(actual,expected,rtol=2e-6,atol=3e-5)
    if cache:assert stats['fallback_levels']>0
    assert stats['kernel_dispatches']>0


def problem(objective='RMSE',parameter=None,method='Newton',**extra):
    bins,_,cf,cb,w=dataset(rows=257)
    rng=np.random.default_rng(881)
    signal=2*(bins[1]==1)-1.5*(bins[1]==6)+.07*bins[0]+rng.normal(0,.1,len(w))
    y=(1/(1+np.exp(-signal)) if objective=='CrossEntropy' else (signal>.4) if objective=='Logloss' else
       np.exp(signal*.25) if objective in ('Poisson','Tweedie','LogLinQuantile') else signal)
    config=options(objective,iterations=3,learning_rate=.02 if objective=='Lq' and parameter>2 and method=='Gradient' else .07,
        bias=0.,objective_param=parameter,leaf_estimation_method=method,sample_weight=w,**extra)
    return bins,np.float32(y),cf,cb,dict(config,candidate_types=np.uint8(cf==1))


CASES=[(o,None,m) for o in ('RMSE','Logloss','CrossEntropy') for m in ('Newton','Gradient')]+SCALAR_CASES
@pytest.mark.parametrize('objective,parameter,method',CASES)
@pytest.mark.parametrize('score',['Cosine','NewtonCosine'])
@pytest.mark.parametrize('count',[1,4])
def test_typed_forests_and_all_prefix_cursors_match_cuda_equations(objective,parameter,method,score,count):
    bins,y,cf,cb,config=problem(objective,parameter,method,score_function=score,permutation_count=count)
    expected=train_reference(bins,y,cf,cb,**config)
    with _ordered.Session(bins,y,cf,cb,**config) as session:
        for _ in range(config['iterations']):session.step()
        actual=session.result();state=session.state()
    compare(actual,expected)
    np.testing.assert_array_equal(actual.split_types,expected['split_types'])
    np.testing.assert_allclose(state['cursors'],expected['cursors'],rtol=1e-5,atol=3e-6)
    if score=='NewtonCosine' and objective in ('LogLinQuantile','Quantile','MAE','MAPE'):
        # These objectives have zero structure curvature in CUDA. The safe
        # result is a constant tree, also produced by the independent oracle.
        assert not actual.depths.any()
    else:
        assert actual.split_types.any()


@pytest.mark.parametrize('objective',['RMSE','Logloss'])
@pytest.mark.parametrize('sampler',['No','Bayesian','Bernoulli','Poisson','MVS'])
@pytest.mark.parametrize('backtracking',['No','AnyImprovement','Armijo'])
@pytest.mark.parametrize('count',[1,4])
def test_every_sampler_backtracking_and_permutation_restores_exactly(objective,sampler,backtracking,count):
    bins,y,cf,cb,config=problem(objective,permutation_count=count,bootstrap_type=sampler,
        leaf_estimation_backtracking=backtracking,leaf_estimation_iterations=3,random_strength=.4,subsample=.7)
    with _ordered.Session(bins,y,cf,cb,**config) as full:
        first=full.step();saved=full.state()
        expected=[full.step() for _ in range(2)];final=full.state();predictions=full.predictions()
    with _ordered.Session(bins,y,cf,cb,**(config|dict(iterations=2,initial_state=saved))) as resumed:
        for ref in expected:
            tree=resumed.step()
            for key in ('depth','split_features','split_bins','split_types','leaf_values','leaf_weights','loss'):
                np.testing.assert_array_equal(getattr(tree,key),getattr(ref,key))
        np.testing.assert_array_equal(resumed.state()['cursors'],final['cursors'])
        np.testing.assert_array_equal(resumed.predictions(),predictions)


@pytest.mark.parametrize('change',['flags','numeric_255','mixed','shape','dtype'])
def test_invalid_typed_candidates_fail_before_gpu(monkeypatch,change):
    bins,y,cf,cb,config=problem()
    cf=cf.copy();cb=cb.copy();types=config['candidate_types'].copy()
    if change=='flags':types[0]=2
    if change=='numeric_255':cb[0]=255
    if change=='mixed':types[0]=1
    if change=='shape':types=types[:-1]
    if change=='dtype':types=types.astype(float)
    config['candidate_types']=types
    monkeypatch.setattr(_ordered,'_load',lambda *a:(_ for _ in ()).throw(AssertionError('Invalid input reached GPU')))
    with pytest.raises(ValueError):_ordered.Session(bins,y,cf,cb,**config)


def test_snapshot_fingerprint_preserves_type_identity_before_gpu(monkeypatch):
    bins,y,cf,cb,config=problem()
    with _ordered.Session(bins,y,cf,cb,**config) as session:session.step();saved=session.state()
    config['candidate_types']=np.zeros_like(cf,np.uint8)
    monkeypatch.setattr(_ordered,'_load',lambda *a:(_ for _ in ()).throw(AssertionError('Mismatched snapshot reached GPU')))
    with pytest.raises(ValueError,match='match'):_ordered.Session(bins,y,cf,cb,**config,initial_state=saved)
