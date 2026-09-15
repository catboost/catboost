"""Resident candidate batches and original-edge coupled leaf updates."""
import ctypes as ct
import hashlib
from pathlib import Path
import platform
import subprocess
import numpy as np
import pytest
from test_pairwise_candidate_kernels import problem,reference as candidate_reference
from test_pairwise_matrix_kernels import reference as leaf_reference,edge_reference
from test_bootstrap import uniforms,seed_for_item,next_word


class Params(ct.Structure):
    _fields_=[(n,ct.c_uint32) for n in ('rows','pairs','features','candidates','parents','leaves','iterations','tile',
        'leaf_method','score_method','bootstrap','seed_low','seed_high','absolute_iteration','r0','r1')]
    _fields_ += [(n,ct.c_float) for n in ('l2','non_diag','temperature','subsample')]


@pytest.fixture(scope='module')
def runtime():
    if platform.system()!='Darwin' or platform.machine()!='arm64':pytest.skip('Apple GPU required')
    source=Path(__file__).with_name('pairwise_matrix_runtime_probe.mm');root=source.parent.parent
    names=('metal_full_matrix_runtime.h','metal_pairwise_matrix_runtime.h','metal_pairwise_matrix_kernels.h','metal_pairwise_candidate_kernels.h',
        'metal_leaf_matrix_kernels.h','metal_pairwise_score_kernels.h','metal_bootstrap_kernels.h','metal_sort.h','metal_sort.mm','metal_sort_kernels.h','metal_trainer.h')
    deps=[source,*(root/'native'/n for n in names)]
    digest=hashlib.sha256(b''.join(p.read_bytes() for p in deps)).hexdigest()[:20]
    path=root/'.build'/('pairwise_matrix_runtime_'+digest+'.dylib')
    if not path.exists():
        subprocess.run(['xcrun','clang++','-std=c++17','-O2','-fobjc-arc','-dynamiclib','-framework','Foundation',
            '-framework','Metal',str(source),str(root/'native/metal_sort.mm'),'-o',str(path)],capture_output=True,text=True,check=True)
    lib=ct.CDLL(str(path))
    lib.cbm_pair_matrix_runtime_probe.argtypes=[ct.POINTER(Params),ct.c_uint64]+[ct.c_void_p]*21+[ct.c_uint32]
    lib.cbm_pair_matrix_runtime_probe.restype=ct.c_int
    def run(args,*,leaves=8,iterations=3,tile=4,method='Newton',score='Newton',bootstrap=0,seed=718,
            iteration=0,temperature=.7,subsample=.7,object_weights=None,budget=2**30):
        bins,parent,point,win,lose,pairs,features,borders,types=args
        ids=np.arange(len(parent),dtype=np.uint32)%leaves
        weights=np.ascontiguousarray(object_weights if object_weights is not None else np.linspace(.2,2,len(parent)),np.float32)
        rows=np.argsort(ids,kind='stable').astype(np.uint32)
        offsets=np.searchsorted(ids[rows],np.arange(leaves+1)).astype(np.uint32)
        arrays=[np.ascontiguousarray(a,d) for a,d in zip((bins,parent,ids,point,win,lose,pairs,weights,rows,offsets,features,borders,types),
            (np.uint8,np.uint32,np.uint32,np.float32,np.uint32,np.uint32,np.float32,np.float32,np.uint32,np.uint32,np.uint32,np.uint32,np.uint8))]
        count=len(features);values=np.zeros(leaves,np.float32);leaf_weights=np.zeros(leaves,np.float32)
        scores=np.zeros((count,2),np.float32);loss=np.zeros(2,np.float64)
        allocated,dispatches,selected_tile=ct.c_uint64(),ct.c_uint64(),ct.c_uint32();error=ct.create_string_buffer(4096)
        p=Params(len(parent),len(win),len(bins),count,leaves//2,leaves,iterations,tile,method=='Gradient',score=='Gradient',
            bootstrap,seed&0xffffffff,seed>>32,iteration,0,0,3.,.2,temperature,subsample)
        code=lib.cbm_pair_matrix_runtime_probe(ct.byref(p),budget,*(a.ctypes.data for a in (*arrays,scores,values,leaf_weights,loss)),
            ct.byref(allocated),ct.byref(dispatches),ct.byref(selected_tile),error,len(error))
        if code:raise ValueError(error.value.decode())
        return dict(scores=scores.sum(axis=1,dtype=float),values=values,weights=leaf_weights,loss=loss,
                    allocated=allocated.value,dispatches=dispatches.value,tile=selected_tile.value)
    return run


def multipliers(count,kind,seed,iteration):
    uniform=uniforms(count,seed=seed,iteration=iteration)
    if kind==0:return np.ones(count,np.float32)
    if kind==1:return np.power(-np.log(uniform+np.float32(1e-20)),np.float32(.7)).astype(np.float32)
    if kind==2:return (uniform<np.float32(.7)).astype(np.float32)
    result=[];lam=np.float32(-np.log(np.float32(1)-np.float32(.7)))
    for index in range(count):
        state=seed_for_item(index,seed=seed,iteration=iteration);total=np.float32(0);draws=0
        while True:
            state,word=next_word(state)
            value=np.maximum(np.minimum(np.float32(word)*np.float32(2**-32),np.nextafter(np.float32(1),np.float32(0))),np.float32(2**-32))
            total=np.float32(total+np.float32(np.log(value)));draws+=1
            if total<=-lam:break
        result.append(draws-1)
    return np.asarray(result,np.float32)


@pytest.mark.parametrize('kind',[0,1,2,3])
@pytest.mark.parametrize('method',['Newton','Gradient'])
@pytest.mark.parametrize('score',['Newton','Gradient'])
def test_edge_bootstrap_and_original_weight_leaf_walk(runtime,kind,method,score):
    args=problem(parents=4,pairs=2053,count=9)
    actual=runtime(args,bootstrap=kind,method=method,score=score,iteration=3)
    sampled=list(args);sampled[5]=np.float32(args[5]*multipliers(len(args[3]),kind,718,3))
    expected=candidate_reference(sampled,4,score,0,9)
    np.testing.assert_allclose(actual['scores'],[v[3] for v in expected],rtol=4e-5,atol=2e-5)
    ids=np.arange(len(args[1]),dtype=np.uint32)%8
    values=np.zeros(8,np.float32)
    for _ in range(3):
        ref=leaf_reference(np.float32(args[2]+values[ids]),*args[3:6],ids,8,method)
        values=np.float32(values+ref['direction']);values[-1]=0
    values=np.float32(values-values.mean(dtype=np.float64))
    np.testing.assert_allclose(actual['values'],values,rtol=2e-5,atol=2e-6)
    expected_weights=np.bincount(ids,weights=np.linspace(.2,2,len(ids)).astype(np.float32),minlength=8)
    np.testing.assert_allclose(actual['weights'],expected_weights,rtol=2e-6,atol=2e-6)
    edge=edge_reference(np.float32(args[2]+actual['values'][ids]),*args[3:6])
    np.testing.assert_allclose(actual['loss'],[edge[:,3].sum(),edge[:,2].sum()],rtol=3e-6,atol=2e-5)


@pytest.mark.parametrize('tile',[1,2,7,16,32])
def test_persistent_candidate_tiles_and_leaf_reuse_do_not_change_results(runtime,tile):
    args=problem(parents=4,count=32)
    expected=runtime(args,tile=32);actual=runtime(args,tile=tile)
    for key in ('scores','values','weights','loss'):np.testing.assert_array_equal(actual[key],expected[key])
    assert actual['allocated']<=expected['allocated']


def test_budget_selects_a_smaller_tile_before_allocating(runtime):
    args=problem(parents=4,count=32)
    small=runtime(args,tile=1)
    selected=runtime(args,tile=32,budget=small['allocated'])
    assert selected['tile']==1 and selected['allocated']==small['allocated']
    with pytest.raises(ValueError,match='budget'):runtime(args,tile=32,budget=small['allocated']-1)


@pytest.mark.parametrize('bad',['pairs','mvs','point','weights'])
def test_invalid_metadata_or_gpu_point_is_rejected(runtime,bad):
    args=list(problem())
    options={}
    if bad=='pairs':args[3][0]=len(args[1])
    elif bad=='mvs':options['bootstrap']=4
    elif bad=='point':args[2][1]=np.nan
    else:options['object_weights']=np.full(len(args[1]),-1,np.float32)
    with pytest.raises(ValueError):runtime(args,**options)
