"""Actual Metal YetiRank target versus an independent seeded pair reference."""
import ctypes as ct
import hashlib
from pathlib import Path
import platform
import subprocess

import numpy as np
import pytest


class Params(ct.Structure):
    _fields_ = [(name, ct.c_uint32) for name in
                ('rows', 'groups', 'tasks', 'permutations', 'seed_low', 'seed_high')]
    _fields_ += [('decay', ct.c_float), ('center_rows', ct.c_uint32)]


@pytest.fixture(scope='module')
def probe():
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        pytest.skip('requires Apple Silicon Metal')
    root = Path(__file__).resolve().parents[1]
    source, header = root/'tests/yeti_rank_probe.mm', root/'native/metal_yeti_rank_kernels.h'
    digest = hashlib.sha256(source.read_bytes()+header.read_bytes()).hexdigest()[:20]
    path = root/'.build'/('yeti_rank_'+digest+'.dylib')
    if not path.exists():
        subprocess.run(['xcrun', 'clang++', '-std=c++17', '-O2', '-fobjc-arc', '-dynamiclib',
                        '-framework', 'Foundation', '-framework', 'Metal', str(source), '-o', str(path)],
                       capture_output=True, text=True, check=True)
    lib = ct.CDLL(str(path)); f32, u32 = ct.POINTER(ct.c_float), ct.POINTER(ct.c_uint32)
    lib.cbm_yeti_rank_probe.argtypes = [ct.POINTER(Params), f32, f32, f32, u32, f32, f32, ct.c_char_p, ct.c_uint32]
    lib.cbm_yeti_rank_probe.restype = ct.c_int

    def run(targets, weights, point, offsets, *, permutations=10, decay=.85, seed=147193, center_rows=None):
        y, w, x = [np.ascontiguousarray(v, np.float32) for v in (targets, weights, point)]
        off = np.ascontiguousarray(offsets, np.uint32)
        assert y.ndim == 1 and y.shape == w.shape == x.shape
        result = np.empty((len(y), 2), np.float32); prep = np.empty_like(result)
        p = Params(len(y), len(off)-1, 0, permutations, seed & 0xffffffff, seed >> 32, decay,
                   len(y) if center_rows is None else center_rows)
        message = ct.create_string_buffer(4096)
        code = lib.cbm_yeti_rank_probe(ct.byref(p), *(v.ctypes.data_as(f32) for v in (y,w,x)),
            off.ctypes.data_as(u32), prep.ctypes.data_as(f32), result.ctypes.data_as(f32), message, len(message))
        if code: raise ValueError(message.value.decode())
        return prep, result
    return run


def reference(targets, weights, point, offsets, *, permutations=10, decay=.85, seed=147193, center_rows=None):
    """Generate CUDA's lane random stream, sort queries, then add explicit edges."""
    f = np.float32
    y,w,x = [np.asarray(v,np.float32) for v in (targets,weights,point)]
    qids = np.repeat(np.arange(len(offsets)-1),np.diff(offsets))
    centered = x.copy()
    stop = len(y) if center_rows is None else center_rows
    for start,end in zip(offsets[:-1],offsets[1:]):
        centered[start:min(end,stop)] -= f(np.mean(x[start:end],dtype=np.float64))
    centered = np.minimum(centered,f(70))
    exponents = np.exp(centered,dtype=np.float32)
    result = np.zeros((len(y),2),np.float32)
    q = 0
    while q < len(offsets)-1:
        start = offsets[q]; limit=min(start+1024,len(y))
        next_q = len(offsets)-1 if limit==len(y) else qids[limit]
        end=offsets[next_q]; count=end-start
        states=(np.uint64(127*q)+np.uint64(16807)*np.arange(256,dtype=np.uint64)+1)&0xffffffff
        def advance():
            nonlocal states
            states=(states*np.uint64(1664525)+np.uint64(1013904223))&0xffffffff
            return states.astype(np.float32)*f(2.328306435996595e-10)
        for _ in range(3): advance()
        states=(states+np.uint64((seed&0xffffffff)+(seed>>32)))&0xffffffff
        for _ in range(3): advance()
        a=exponents[start:end]; r=f(y[start:end]*w[start:end]); logs=centered[start:end]
        use_log=np.any(a<f(1e-15))
        for _ in range(permutations):
            uniform=np.concatenate([advance() for _ in range(4)])[:count]
            keys=f(logs+np.log(uniform)-np.log(f(1.000001)-uniform)) if use_log else f(a*f(uniform/(f(1.000001)-uniform)))
            order=np.lexsort((np.arange(count),-keys,qids[start:end]))
            for local_q in range(q,next_q):
                begin=offsets[local_q]-start; last=offsets[local_q+1]-start
                for rank in range(1,last-begin):
                    left,right=order[begin+rank-1:begin+rank+1]
                    mass=f(f(f(.15)*f(f(decay)**f(rank-1)))*f(abs(r[left]-r[right]))/f(permutations))
                    if use_log:
                        diff=float(logs[left])-float(logs[right]); tiny=np.exp(-abs(diff))
                        pr=tiny/(1+tiny) if diff>=0 else 1/(1+tiny)
                        pl=1/(1+tiny) if diff>=0 else tiny/(1+tiny)
                        gradient=f(mass*(pr if r[left]>r[right] else -pl))
                    else:
                        gradient=f(f(mass*(a[right] if r[left]>r[right] else -a[left]))/f(a[left]+a[right]))
                    result[start+left]+=np.array([gradient,mass],np.float32)
                    result[start+right]+=np.array([-gradient,mass],np.float32)
        q=next_q
    return np.column_stack((centered,exponents)),result


@pytest.mark.parametrize('sizes',[[1],[2],[17,31,5],[255,256,257],[1023],[300,700,25,1023,1],[1]*1025])
@pytest.mark.parametrize('permutations,decay',[(1,.85),(7,.45),(10,1),(3,0)])
def test_seeded_weighted_cuda_pair_equations(probe,sizes,permutations,decay):
    rng=np.random.default_rng(sum(sizes)+permutations)
    offsets=np.r_[0,np.cumsum(sizes)]; n=offsets[-1]
    y=rng.uniform(0,1,n).astype(np.float32); w=rng.uniform(.2,3,n).astype(np.float32);w[::19]=0
    x=rng.normal(0,1,n).astype(np.float32)
    options=dict(permutations=permutations,decay=decay,seed=0xfedcba9812345678)
    expected=reference(y,w,x,offsets,**options); actual=probe(y,w,x,offsets,**options)
    np.testing.assert_allclose(actual[0],expected[0],rtol=3e-6,atol=1e-6)
    np.testing.assert_allclose(actual[1],expected[1],rtol=4e-5,atol=3e-7)
    for begin,end in zip(offsets[:-1],offsets[1:]):
        assert abs(actual[1][begin:end,0].sum(dtype=float))<3e-6
    assert np.all(actual[1][:,1]>=0)


def test_legacy_prefix_centering_is_explicit_and_differs_from_full_query_centering(probe):
    y=np.array([.1,.9,.3,.8,.2,.6],np.float32);x=np.array([10,11,9,-3,-2,-1],np.float32)
    offsets=[0,3,6]; w=np.ones(6,np.float32)
    legacy=probe(y,w,x,offsets,center_rows=2)
    expected=reference(y,w,x,offsets,center_rows=2)
    np.testing.assert_allclose(legacy[1],expected[1],rtol=4e-5,atol=3e-7)
    full=probe(y,w,x,offsets)
    assert not np.allclose(legacy[1],full[1])
    shifted=probe(y,w,x+100,offsets)
    np.testing.assert_allclose(full[1],shifted[1],rtol=2e-5,atol=3e-7)


def test_log_domain_preserves_very_negative_logits(probe):
    y=np.array([0,1,.3,.7],np.float32);w=np.ones(4,np.float32)
    x=np.array([-1002,-1000,-999,-1001],np.float32)
    expected=reference(y,w,x,[0,4],center_rows=0)
    actual=probe(y,w,x,[0,4],center_rows=0)
    np.testing.assert_allclose(actual[1],expected[1],rtol=2e-5,atol=3e-7)
    assert np.isfinite(actual[1]).all() and np.any(actual[1][:,1]>0)


def test_seed_replay_and_relevance_weights(probe):
    rng=np.random.default_rng(84); y=rng.uniform(0,1,79).astype(np.float32);w=np.ones(79,np.float32)
    x=np.zeros(79,np.float32)
    a=probe(y,w,x,[0,79],seed=1)[1]
    np.testing.assert_array_equal(a,probe(y,w,x,[0,79],seed=1)[1])
    assert not np.array_equal(a,probe(y,w,x,[0,79],seed=2)[1])
    np.testing.assert_allclose(probe(y,w*2,x,[0,79],seed=1)[1],2*a,rtol=2e-6,atol=2e-7)
    np.testing.assert_array_equal(probe(y,np.zeros_like(w),x,[0,79])[1],0)


@pytest.mark.parametrize('case',['large_query','negative_weight','nan_point','nan_target','decay','permutations'])
def test_native_boundaries(probe,case):
    n=1024 if case=='large_query' else 2
    y=np.zeros(n,np.float32);w=np.ones(n,np.float32);x=np.zeros(n,np.float32);kw={}
    if case=='negative_weight':w[0]=-1
    if case=='nan_point':x[0]=np.nan
    if case=='nan_target':y[0]=np.nan
    if case=='decay':kw['decay']=np.inf
    if case=='permutations':kw['permutations']=0
    with pytest.raises(ValueError):probe(y,w,x,[0,n],**kw)
