"""Resident QCE batching, memory accounting and invalid-trial isolation."""
import ctypes as ct
import hashlib
from pathlib import Path
import platform,subprocess
import numpy as np
import pytest
from test_qce_candidate_kernels import reference as candidate_reference
from test_query_cross_entropy_training import problem,leaf_walk,target
from test_bootstrap import uniforms


class Params(ct.Structure):
    _fields_=[(n,ct.c_uint32) for n in ('rows','groups','features','candidates','parents','leaves','iterations','tile','query_tile',
        'bootstrap','seed_low','seed_high','absolute','invalid_trial','r0','r1')]+[(n,ct.c_float) for n in ('alpha','l2','non_diag','subsample')]


@pytest.fixture(scope='module')
def runtime():
    if platform.system()!='Darwin' or platform.machine()!='arm64':pytest.skip('Apple GPU required')
    root=Path(__file__).resolve().parents[1];source=Path(__file__).with_name('query_cross_entropy_runtime_probe.mm')
    deps=[source,*(root/'native').glob('*.h'),root/'native/metal_sort.mm']
    digest=hashlib.sha256(b''.join(p.read_bytes() for p in sorted(deps))).hexdigest()[:20]
    output=root/'.build'/('qce_runtime_'+digest+'.dylib')
    if not output.exists():
        subprocess.run(['xcrun','clang++','-std=c++17','-O2','-fobjc-arc','-dynamiclib','-framework','Foundation','-framework','Metal',
            str(source),str(root/'native/metal_sort.mm'),'-o',str(output)],capture_output=True,text=True,check=True)
    lib=ct.CDLL(str(output));lib.cbm_qce_runtime_probe.argtypes=[ct.POINTER(Params),ct.c_uint64]+[ct.c_void_p]*27+[ct.c_uint32]
    def run(args,*,tile=8,query_tile=64,budget=2**30,invalid_trial=False,minimum_threads=32):
        rows=len(args['targets']);groups=len(args['group_offsets'])-1;count=len(args['candidate_features']);leaves=1<<args['depth']
        ids=np.arange(rows,dtype=np.uint32)%leaves;parents=ids%(leaves//2)
        order=np.argsort(ids,kind='stable').astype(np.uint32);offsets=np.searchsorted(ids[order],np.arange(leaves+1)).astype(np.uint32)
        arrays=[np.ascontiguousarray(a,d) for a,d in zip((args['targets'],args['sample_weight'],args['initial_predictions'],args['group_offsets'],args['query_scales'],
            args['bins'],parents,ids,order,offsets,args['candidate_features'],args['candidate_bins'],args['candidate_types']),
            (np.float32,np.float32,np.float32,np.uint32,np.float32,np.uint8,np.uint32,np.uint32,np.uint32,np.uint32,np.uint32,np.uint32,np.uint8))]
        stats=np.zeros((rows,4),np.float32);qs=np.zeros((groups,4),np.float32);mask=np.zeros(groups,np.uint8);scores=np.zeros((count,2),np.float32)
        g=np.zeros(leaves,np.float32);h=np.zeros((leaves,leaves),np.float32);values=np.zeros(leaves,np.float32);weights=np.zeros(leaves,np.float32)
        loss=np.zeros(2,np.float64);trial=ct.c_double();size=ct.c_uint64();dispatch=ct.c_uint64();tiles=np.zeros(2,np.uint32);error=ct.create_string_buffer(4096)
        p=Params(rows,groups,len(args['bins']),count,leaves//2,leaves,args['leaf_estimation_iterations'],tile,query_tile,args['bootstrap_type']=='Bernoulli',718,0,3,invalid_trial,minimum_threads,0,
            args['alpha'],args['l2_leaf_reg'],args['non_diagonal_regularization'],args['subsample']);p.bootstrap*=2
        code=lib.cbm_qce_runtime_probe(ct.byref(p),budget,*(a.ctypes.data for a in (*arrays,stats,qs,mask,scores,g,h,values,weights,loss)),
            ct.byref(trial),ct.byref(size),ct.byref(dispatch),tiles.ctypes.data,error,len(error))
        if code:raise ValueError(error.value.decode())
        return dict(stats=stats,groups=qs,mask=mask,scores=scores,gradient=g,hessian=h,values=values,weights=weights,loss=loss,trial=trial.value,bytes=size.value,dispatches=dispatch.value,tiles=tiles)
    return run


@pytest.mark.parametrize('tile',[1,2,4,8,9])
@pytest.mark.parametrize('query_tile',[1,3,8])
def test_candidate_and_query_tile_changes_preserve_every_result(runtime,tile,query_tile):
    args=problem(bootstrap_type='Bernoulli');expected=runtime(args);actual=runtime(args,tile=tile,query_tile=query_tile)
    for key in ('stats','groups','mask','scores','gradient','hessian','values','weights','loss'):
        np.testing.assert_array_equal(actual[key],expected[key],err_msg=key)


@pytest.mark.parametrize('kind',['No','Bernoulli'])
def test_actual_qce_statistics_drive_sampled_candidates_and_original_query_leaves(runtime,kind):
    args=problem(bootstrap_type=kind,learning_rate=1.,leaf_estimation_iterations=1);actual=runtime(args)
    ids=np.arange(len(args['targets']),dtype=np.uint32)%8
    expected_mask=np.ones(8,np.uint8) if kind=='No' else (uniforms(8,seed=718,iteration=3)<np.float32(.7)).astype(np.uint8)
    np.testing.assert_array_equal(actual['mask'],expected_mask)
    candidates=candidate_reference((actual['stats'],actual['groups'],args['group_offsets'],args['bins'],ids%4,
        args['candidate_features'],args['candidate_bins'],args['candidate_types'],actual['mask']),4,args['l2_leaf_reg'],args['non_diagonal_regularization'])
    np.testing.assert_allclose(actual['scores'].sum(axis=1,dtype=float),[c[3] for c in candidates],rtol=3e-5,atol=3e-6)
    expected_values,expected_weights,_=leaf_walk(args,args['initial_predictions'],ids,8)
    np.testing.assert_allclose(actual['values'],expected_values,rtol=3e-5,atol=3e-6)
    np.testing.assert_allclose(actual['weights'],expected_weights,rtol=3e-6,atol=1e-6)
    expected=target(args,args['initial_predictions'],ids,8)
    np.testing.assert_allclose(actual['gradient'],expected[3],rtol=4e-5,atol=4e-6)
    np.testing.assert_allclose(actual['hessian'],expected[4],rtol=4e-5,atol=4e-6)


def test_workspace_shrinks_before_allocation_and_preserves_arithmetic(runtime):
    args=problem(depth=6);small=runtime(args,tile=1,query_tile=1);bounded=runtime(args,budget=small['bytes'])
    assert bounded['bytes']==small['bytes'] and bounded['tiles'].tolist()==[1,1]
    np.testing.assert_array_equal(small['values'],bounded['values']);np.testing.assert_array_equal(small['scores'],bounded['scores'])
    with pytest.raises(ValueError,match='budget'):runtime(args,budget=small['bytes']-1)


def test_invalid_trial_has_separate_status_and_recovery_preserves_accepted_loss(runtime):
    result=runtime(problem(),invalid_trial=True)
    assert np.isposinf(result['trial']) and np.isfinite(result['loss']).all()


@pytest.mark.parametrize('size',[1,17,32,33,64,65,128,129,256])
@pytest.mark.parametrize('sampling',['No','Bernoulli'])
def test_query_threadgroup_specialization_is_bit_identical_to_full_256_threads(runtime,size,sampling):
    rng=np.random.default_rng(3262);sizes=np.array([1,3,7,size]);offsets=np.r_[0,sizes.cumsum()].astype(np.uint32);rows=int(offsets[-1])
    args=problem(bins=rng.integers(0,4,(3,rows),dtype=np.uint8),targets=rng.integers(0,2,rows).astype(np.float32),
        group_offsets=offsets,query_scales=np.array([.7,1.3,1,.8],np.float32),sample_weight=rng.uniform(.2,2,rows).astype(np.float32),
        initial_predictions=rng.normal(0,.4,rows).astype(np.float32),bootstrap_type=sampling)
    expected=runtime(args,minimum_threads=256);actual=runtime(args)
    for key in ('stats','groups','mask','scores','gradient','hessian','values','weights','loss'):
        np.testing.assert_array_equal(actual[key],expected[key],err_msg=key)
