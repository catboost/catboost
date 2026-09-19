"""Sparse adjacent contributions against the preserved dense GPU algorithm."""
import ctypes as ct
import hashlib,platform,subprocess
from pathlib import Path
import numpy as np
import pytest
from test_pfound_pair_kernels import Params,probe

@pytest.fixture(scope='module')
def sparse():
    if platform.system()!='Darwin' or platform.machine()!='arm64':pytest.skip('Apple GPU required')
    root=Path(__file__).resolve().parents[1];source=Path(__file__).with_name('pfound_sparse_probe.mm')
    names=('metal_pfound_pair_kernels.h','metal_pfound_sparse_kernels.h','metal_bootstrap_kernels.h','metal_sort.h','metal_sort.mm','metal_sort_kernels.h')
    digest=hashlib.sha256(b''.join(p.read_bytes() for p in (source,*(root/'native'/n for n in names)))).hexdigest()[:20]
    output=root/'.build'/('pfound_sparse_'+digest+'.dylib')
    if not output.exists():
        result=subprocess.run(['xcrun','clang++','-std=c++17','-O2','-fobjc-arc','-dynamiclib','-framework','Foundation','-framework','Metal',str(source),str(root/'native/metal_sort.mm'),'-o',str(output)],capture_output=True,text=True)
        assert result.returncode==0,result.stdout+result.stderr
    lib=ct.CDLL(str(output));lib.cbm_pfound_sparse_probe.argtypes=[ct.POINTER(Params)]+[ct.c_void_p]*12+[ct.c_uint32]
    lib.cbm_pfound_sparse_probe.restype=ct.c_int
    def run(y,w,x,offsets,*,permutations=10,decay=.85,seed=147193,bootstrap=0,temperature=.7,absolute=0,document_ids=None):
        y,w,x=[np.ascontiguousarray(v,np.float32) for v in (y,w,x)];off=np.ascontiguousarray(offsets,np.uint32)
        docs=np.ascontiguousarray(np.arange(len(y)) if document_ids is None else document_ids,np.uint32)
        n=len(y);count=int(sum(int(size)*(int(size)-1)//2 for size in np.diff(off)));slots=n*permutations
        assert y.shape==w.shape==x.shape==docs.shape and y.ndim==1
        assert slots<=2**24,'Probe test input would allocate more than the GPU slot limit'
        exp=np.empty(n,np.float32);keys=np.empty(slots,np.uint32);matrix=np.empty(slots,np.float32);pairs=np.empty((slots,2),np.uint32);edges=np.empty((slots,4),np.float32)
        p=Params(n,len(off)-1,0,count,permutations,seed&0xffffffff,seed>>32,0,decay,temperature,bootstrap,absolute)
        error=ct.create_string_buffer(4096);allocated=ct.c_uint64()
        code=lib.cbm_pfound_sparse_probe(ct.byref(p),*(v.ctypes.data for v in (y,w,x,off,docs,exp,keys,matrix,pairs,edges)),ct.byref(allocated),error,len(error))
        if code:raise ValueError(error.value.decode())
        return dict(exponent=exp,keys=keys,matrix=matrix,pairs=pairs,edges=edges,allocated=allocated.value,dense_pairs=count)
    return run


def check_equal(dense,actual):
    keys=actual['keys'];valid=keys!=np.uint32(0xffffffff)
    assert np.all(keys[:-1]<=keys[1:])
    head=valid.copy();head[1:] &= keys[1:]!=keys[:-1]
    np.testing.assert_array_equal(actual['exponent'].view(np.uint32),dense['exponent'].view(np.uint32))
    np.testing.assert_array_equal(actual['matrix'][~head],0)
    matrix=np.zeros(actual['dense_pairs'],np.float32);matrix[keys[head]]=actual['matrix'][head]
    np.testing.assert_array_equal(matrix.view(np.uint32),dense['matrix'].view(np.uint32))
    np.testing.assert_array_equal(actual['pairs'][head],dense['pairs'][keys[head]])
    np.testing.assert_array_equal(actual['edges'][head].view(np.uint32),dense['edges'][keys[head]].view(np.uint32))
    np.testing.assert_array_equal(actual['edges'][~head],0)


@pytest.mark.parametrize('sizes',[(1,),(2,),(17,31,5),(1023,),(300,700,25,1023,1),(2,)*512,(1,3)*256])
@pytest.mark.parametrize('permutations,decay',[(1,.85),(3,0),(7,.45),(10,1)])
def test_sparse_storage_preserves_dense_permutation_sums_and_edges_exactly(sparse,probe,sizes,permutations,decay):
    n=sum(sizes);rng=np.random.default_rng(n+permutations);off=np.r_[0,np.cumsum(sizes)]
    y=rng.uniform(0,1,n).astype(np.float32);w=rng.uniform(.2,3,n).astype(np.float32);w[::19]=0;x=rng.normal(size=n).astype(np.float32)
    options=dict(permutations=permutations,decay=decay,seed=0xf1234567abcd9765,document_ids=np.arange(n,dtype=np.uint32)*3+7)
    actual=sparse(y,w,x,off,**options);dense=probe(y,w,x,off,**options);check_equal(dense,actual)
    assert len(actual['keys'])==n*permutations


@pytest.mark.parametrize('seed',[0,1,2**32-1,2**32,2**64-1])
@pytest.mark.parametrize('temperature',[0,.7,1])
def test_sparse_bayesian_draws_remain_indexed_by_original_dense_pair_id(sparse,probe,seed,temperature):
    rng=np.random.default_rng(739);n=255;y=rng.uniform(0,1,n).astype(np.float32);w=rng.uniform(.2,3,n).astype(np.float32);x=rng.normal(size=n).astype(np.float32)
    args=(y,w,x,[0,31,64,255]);options=dict(bootstrap=1,temperature=temperature,seed=seed,absolute=17)
    check_equal(probe(*args,**options),sparse(*args,**options))


def test_sparse_capacity_scales_with_permutations_instead_of_dense_triangle(sparse,probe):
    n=1023;args=(np.linspace(0,1,n),np.ones(n),np.linspace(-3,3,n),[0,n])
    actual=sparse(*args);check_equal(probe(*args),actual)
    dense_bytes=32*n+32*(n*(n-1)//2)+8*2+8+4
    assert actual['allocated']<dense_bytes/20
    assert len(actual['matrix'])==10230 and actual['dense_pairs']==522753


@pytest.mark.parametrize('constant',[False,True])
def test_sparse_underflow_tiny_pairs_and_constant_labels(sparse,probe,constant):
    y=np.full(5,.5) if constant else np.array([0,1e-21,.7,.2,.9])
    args=(y,[1e20,1,1,1,1],[0,0,-1000,-999,0],[0,2,5])
    options=dict(permutations=7)
    check_equal(probe(*args,**options),sparse(*args,**options))


@pytest.mark.parametrize('targets,weights,point',[
    ([0,np.nan],[1,1],[0,0]),([0,1],[-1,1],[0,0]),([0,1],[1,np.inf],[0,0]),([0,1],[1,1],[0,np.inf]),
    ([-3e38,3e38],[1,1],[0,0]),([0,10],[3e38,1],[0,0]),
])
def test_sparse_rejects_invalid_and_overflowed_observations(sparse,targets,weights,point):
    with pytest.raises(ValueError):sparse(targets,weights,point,[0,2],permutations=1)


@pytest.mark.parametrize('temperature',[20,30,60,3e38])
def test_sparse_bayesian_overflow_behavior_matches_dense(sparse,probe,temperature):
    args=(np.full(64,.5),np.ones(64),np.linspace(-1,1,64),[0,64])
    options=dict(bootstrap=1,temperature=temperature,permutations=1)
    if temperature<=30:
        check_equal(probe(*args,**options),sparse(*args,**options))
    else:
        for generate in (probe,sparse):
            with pytest.raises(ValueError):generate(*args,**options)
