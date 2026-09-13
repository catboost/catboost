"""Adaptive sparse/dense PFound storage, bounds and exact resident leaf results."""
import ctypes as ct
import hashlib,platform,subprocess
from pathlib import Path
import numpy as np
import pytest
from test_pfound_pair_runtime import problem

class Params(ct.Structure):
    _fields_=[(n,ct.c_uint32) for n in ('rows','groups','features','candidates','parents','leaves','permutations','iterations',
        'leaf_method','score_method','bootstrap','seed_low','seed_high','absolute','group_sampling','passes')]
    _fields_ += [(n,ct.c_float) for n in ('l2','non_diag','temperature','subsample')]
    _fields_ += [('compact',ct.c_uint32),('sparse',ct.c_uint32)]

@pytest.fixture(scope='module')
def runtime():
    if platform.system()!='Darwin' or platform.machine()!='arm64':pytest.skip('Apple GPU required')
    root=Path(__file__).resolve().parents[1];source=Path(__file__).with_name('pfound_sparse_runtime_probe.mm')
    deps=[source,*sorted((root/'native').glob('*.h')),root/'native/metal_sort.mm']
    digest=hashlib.sha256(b''.join(p.read_bytes() for p in deps)).hexdigest()[:20]
    output=root/'.build'/('pfound_sparse_runtime_'+digest+'.dylib')
    if not output.exists():
        result=subprocess.run(['xcrun','clang++','-std=c++17','-O2','-fobjc-arc','-dynamiclib','-framework','Foundation',
            '-framework','Metal',str(source),str(root/'native/metal_sort.mm'),'-o',str(output)],capture_output=True,text=True)
        assert result.returncode==0,result.stdout+result.stderr
    lib=ct.CDLL(str(output));lib.cbm_pfound_sparse_runtime_probe.argtypes=[ct.POINTER(Params),ct.c_uint64]+[ct.c_void_p]*23+[ct.c_uint32]
    lib.cbm_pfound_sparse_runtime_probe.restype=ct.c_int
    def run(problem,*,bootstrap=0,seed=738,absolute=0,subsample=.5,temperature=.7,group_sampling=False,
            method='Newton',score='Newton',passes=2,iterations=3,permutations=7,budget=2**30,compact=True,sparse=True):
        bins,y,w,x,off=problem;n=len(y);groups=len(off)-1;leaves=8;count=9
        parent=np.arange(n,dtype=np.uint32)%4;ids=np.arange(n,dtype=np.uint32)%leaves
        rows=np.argsort(ids,kind='stable').astype(np.uint32);loff=np.searchsorted(ids[rows],np.arange(leaves+1)).astype(np.uint32)
        features=np.arange(count,dtype=np.uint32)%3;borders=np.arange(count,dtype=np.uint32)//3;types=(np.arange(count)%2).astype(np.uint8)
        arrays=[np.ascontiguousarray(a,d) for a,d in zip((bins,parent,ids,x,y,w,off,rows,loff,features,borders,types),
            (np.uint8,np.uint32,np.uint32,np.float32,np.float32,np.float32,np.uint32,np.uint32,np.uint32,np.uint32,np.uint32,np.uint8))]
        cap=min(2*n//groups+8,1023);pc=max(1,sum(min(int(s),cap)*(min(int(s),cap)-1)//2 for s in np.diff(off)))
        if compact and sparse:pc=min(pc,sum(min(int(s),cap) for s in np.diff(off))*permutations)
        shapes=np.zeros((passes,2,5),np.uint32);docs=np.zeros((passes,2,n),np.uint32);pairs=np.zeros((passes,2,pc,2),np.uint32)
        edges=np.zeros((passes,2,pc,4),np.float32);matrix=np.zeros((passes,2,pc),np.float32)
        scores=np.zeros((passes,count,2),np.float32);values=np.zeros((passes,leaves),np.float32);weights=np.zeros_like(values)
        allocated=ct.c_uint64();seconds=ct.c_double();error=ct.create_string_buffer(4096)
        p=Params(n,groups,len(bins),count,4,leaves,permutations,iterations,method=='Gradient',score=='Gradient',bootstrap,seed&0xffffffff,seed>>32,
            absolute,group_sampling,passes,3.,.2,temperature,subsample,compact,sparse)
        code=lib.cbm_pfound_sparse_runtime_probe(ct.byref(p),budget,*(a.ctypes.data for a in (*arrays,shapes,docs,pairs,edges,matrix,scores,values,weights)),
            ct.byref(allocated),ct.byref(seconds),error,len(error))
        if code:raise ValueError(error.value.decode())
        return dict(shapes=shapes,docs=docs,pairs=pairs,edges=edges,matrix=matrix,scores=scores.sum(axis=-1,dtype=float),values=values,weights=weights,
            allocated=allocated.value,seconds=seconds.value,parent=parent,ids=ids,features=features,borders=borders,types=types)
    return run


@pytest.mark.parametrize('sizes',[(3,)*17,(17,31),(64,)*7,(1023,),(4097,),(16,512,16)])
@pytest.mark.parametrize('bootstrap,group',[(0,False),(1,False),(2,False),(2,True)])
@pytest.mark.parametrize('method',['Newton','Gradient'])
def test_adaptive_storage_matches_compacted_dense_results(runtime,sizes,bootstrap,group,method):
    data=problem(sizes);options=dict(bootstrap=bootstrap,group_sampling=group,method=method,permutations=3,passes=2)
    dense=runtime(data,**options,sparse=False);actual=runtime(data,**options)
    np.testing.assert_array_equal(actual['shapes'][...,:4],dense['shapes'][...,:4])
    for key in ('scores','values','weights','docs'):
        np.testing.assert_array_equal(actual[key].view(np.uint8),dense[key].view(np.uint8),err_msg=key)
    for step in range(2):
        for fixed in (False,True):
            rows,groups,pairs,active,slots=actual['shapes'][step,int(fixed)]
            assert slots==min(int(pairs),int(rows)*3)
            # Sum raw weights by original endpoints, after storage changes.
            n=int(slots);valid=actual['matrix'][step,int(fixed),:n]!=0
            got={(int(a),int(b)):v for (a,b),v in zip(actual['pairs'][step,int(fixed),:n][valid],actual['matrix'][step,int(fixed),:n][valid])}
            count=int(pairs);mask=dense['matrix'][step,int(fixed),:count]!=0
            expected={(int(a),int(b)):v for (a,b),v in zip(dense['pairs'][step,int(fixed),:count][mask],dense['matrix'][step,int(fixed),:count][mask])}
            assert got==expected
    cap=min(2*sum(sizes)//len(sizes)+8,1023);sampled=sum(min(n,cap) for n in sizes)
    dense_pairs=sum(min(n,cap)*(min(n,cap)-1)//2 for n in sizes)
    if dense_pairs>sampled*3:assert actual['allocated']<dense['allocated']
    else:assert actual['allocated']==dense['allocated']


def test_large_query_workspace_scales_with_capped_rows_and_permutations(runtime):
    data=problem((65537,));dense=runtime(data,sparse=False,passes=1,iterations=1,permutations=3)
    actual=runtime(data,passes=1,iterations=1,permutations=3)
    assert actual['shapes'][0,0,0]==1023 and actual['shapes'][0,0,4]==3069
    assert actual['allocated']<dense['allocated']/10
    for key in ('scores','values','weights'):np.testing.assert_array_equal(actual[key],dense[key])


def test_sparse_allocation_budget_is_exact_and_rejects_smaller_than_minimum(runtime):
    data=problem((1023,));actual=runtime(data,passes=1,iterations=1,permutations=3)
    exact=runtime(data,passes=1,iterations=1,permutations=3,budget=actual['allocated'])
    assert exact['allocated']==actual['allocated']
    with pytest.raises(ValueError,match='budget'):runtime(data,budget=1024)


@pytest.mark.parametrize('sparse',[False,True])
def test_bayesian_overflow_on_unvisited_dense_pair_is_rejected(runtime,sparse):
    # Seed 738's dense pair 1371 overflows at this temperature, although the
    # single permutation never visits it. The old sparse path accepted it.
    with pytest.raises(ValueError,match='Nonfinite generated PFound target'):
        runtime(problem((64,),constant=True),bootstrap=1,seed=738,temperature=43.82158660888672,
            permutations=1,passes=1,iterations=1,sparse=sparse)


@pytest.mark.parametrize('temperature',[20,30])
def test_finite_high_temperature_preserves_dense_results(runtime,temperature):
    data=problem((64,),constant=True);options=dict(bootstrap=1,temperature=temperature,permutations=1,passes=1,iterations=1)
    dense=runtime(data,**options,sparse=False);actual=runtime(data,**options)
    for key in ('scores','values','weights'):
        np.testing.assert_array_equal(actual[key].view(np.uint8),dense[key].view(np.uint8))


@pytest.mark.parametrize('groups',[33,257])
@pytest.mark.parametrize('bootstrap',[0,1])
def test_sparse_generated_pairs_above_24_bit_ids_preserve_prefix_targets(runtime,groups,bootstrap):
    data=problem((1023,)*groups);n=32*1023
    prefix_data=(data[0][:,:n],data[1][:n],data[2][:n],data[3][:n],data[4][:33])
    options=dict(bootstrap=bootstrap,permutations=1,passes=1,iterations=1)
    prefix=runtime(prefix_data,**options);actual=runtime(data,**options)
    assert actual['shapes'][0,0,2]>2**24 and actual['shapes'][0,0,4]==groups*1023
    assert actual['allocated']<2**30
    for key in ('scores','values','weights'):assert np.isfinite(actual[key]).all()
    for fixed in (0,1):
        count=int(actual['shapes'][0,fixed,4]);p=actual['pairs'][0,fixed,:count];raw=actual['matrix'][0,fixed,:count]
        mask=(raw!=0)&(p[:,1]<n)
        got={(int(a),int(b)):(r.tobytes(),e.tobytes()) for (a,b),r,e in zip(p[mask],raw[mask],actual['edges'][0,fixed,:count][mask])}
        count=int(prefix['shapes'][0,fixed,4]);p0=prefix['pairs'][0,fixed,:count];raw0=prefix['matrix'][0,fixed,:count];mask0=raw0!=0
        expected={(int(a),int(b)):(r.tobytes(),e.tobytes()) for (a,b),r,e in zip(p0[mask0],raw0[mask0],prefix['edges'][0,fixed,:count][mask0])}
        assert got==expected
        # Real generated edges address queries beyond the former global limit.
        live=p[raw!=0].astype(np.uint64);local_left=live[:,0]%1023;local_right=live[:,1]%1023
        pair_ids=(live[:,0]//1023)*522753+local_right*(local_right-1)//2+local_left
        assert int(pair_ids.max())>2**24
