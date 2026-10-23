"""PFoundF/YetiRankPairwise GPU weights versus explicit seeded query pairs."""
import ctypes as ct
import hashlib
from pathlib import Path
import platform,subprocess
import numpy as np
import pytest
from test_bootstrap import uniforms

class Params(ct.Structure):
    _fields_=[(name,ct.c_uint32) for name in ('rows','groups','tasks','pairs','permutations','seed_low','seed_high','reserved')]
    _fields_ += [('decay',ct.c_float),('temperature',ct.c_float),('bootstrap',ct.c_uint32),('absolute',ct.c_uint32)]

@pytest.fixture(scope='module')
def probe():
    if platform.system()!='Darwin' or platform.machine()!='arm64':pytest.skip('Apple GPU required')
    root=Path(__file__).resolve().parents[1];source=Path(__file__).with_name('pfound_pair_probe.mm')
    deps=[source,root/'native/metal_pfound_pair_kernels.h',root/'native/metal_bootstrap_kernels.h']
    digest=hashlib.sha256(b''.join(p.read_bytes() for p in deps)).hexdigest()[:20]
    output=root/'.build'/('pfound_pair_'+digest+'.dylib')
    if not output.exists():
        subprocess.run(['xcrun','clang++','-std=c++17','-O2','-fobjc-arc','-dynamiclib','-framework','Foundation','-framework','Metal',str(source),'-o',str(output)],capture_output=True,text=True,check=True)
    lib=ct.CDLL(str(output));lib.cbm_pfound_pair_probe.argtypes=[ct.POINTER(Params)]+[ct.c_void_p]*11+[ct.c_uint32]
    lib.cbm_pfound_pair_probe.restype=ct.c_int
    def run(y,w,x,offsets,*,permutations=10,decay=.85,seed=147193,bootstrap=0,temperature=.7,absolute=0,document_ids=None):
        y,w,x=[np.ascontiguousarray(v,np.float32) for v in (y,w,x)];off=np.ascontiguousarray(offsets,np.uint32)
        docs=np.ascontiguousarray(np.arange(len(y)) if document_ids is None else document_ids,np.uint32)
        n=len(y);count=int(sum(int(size)*(int(size)-1)//2 for size in np.diff(off)))
        assert y.shape==w.shape==x.shape==docs.shape and y.ndim==1
        exponent=np.empty(n,np.float32);matrix=np.empty(max(count,1),np.float32);pairs=np.empty((max(count,1),2),np.uint32)
        edges=np.empty((max(count,1),4),np.float32);gradient=np.empty((n,2),np.float32)
        p=Params(n,len(off)-1,0,count,permutations,seed&0xffffffff,seed>>32,0,decay,temperature,bootstrap,absolute)
        error=ct.create_string_buffer(4096)
        code=lib.cbm_pfound_pair_probe(ct.byref(p),*(v.ctypes.data for v in (y,w,x,off,docs,exponent,matrix,pairs,edges,gradient)),error,len(error))
        if code:raise ValueError(error.value.decode())
        return dict(exponent=exponent,matrix=matrix[:count],pairs=pairs[:count],edges=edges[:count],gradient=gradient.sum(axis=1,dtype=float))
    return run


def reference(y,w,x,offsets,*,permutations=10,decay=.85,seed=147193,bootstrap=0,temperature=.7,absolute=0,document_ids=None):
    f=np.float32;y,w,x=[np.asarray(v,np.float32) for v in (y,w,x)];off=np.asarray(offsets,int);n=len(y);qids=np.repeat(np.arange(len(off)-1),np.diff(off))
    exp=np.empty(n,np.float32);pair_offsets=[0];a_parts=[];b_parts=[]
    for begin,end in zip(off[:-1],off[1:]):
        exp[begin:end]=np.exp(f(x[begin:end]-x[begin:end].max()),dtype=np.float32)
        b,a=np.tril_indices(end-begin,-1);a_parts.append(a+begin);b_parts.append(b+begin);pair_offsets.append(pair_offsets[-1]+len(a))
    a,b=np.concatenate(a_parts),np.concatenate(b_parts);matrix=np.zeros(len(a),np.float32)
    q=0
    while q<len(off)-1:
        begin=off[q];limit=min(begin+1024,n);next_q=len(off)-1 if limit==n else qids[limit];count=off[next_q]-begin
        cuda_seed=((seed&0xffffffff)+(seed>>32))&0xffffffff
        states=(np.uint64(127*q)+np.uint64(16807)*np.arange(256,dtype=np.uint64)+np.uint64(cuda_seed)*np.uint64(1+q))&0xffffffff
        def advance():
            nonlocal states
            states=(states*np.uint64(1664525)+np.uint64(1013904223))&0xffffffff
            return states.astype(np.float32)*f(2.328306435996595e-10)
        advance();states=(states+np.uint64(cuda_seed))&0xffffffff
        for _ in range(3):advance()
        for _ in range(permutations):
            u=np.concatenate([advance() for _ in range(4)])[:count]
            keys=f(exp[begin:begin+count]*f(u/(f(1.000001)-u)))
            order=np.lexsort((np.arange(count),-keys,qids[begin:begin+count]))+begin
            for query in range(q,next_q):
                local=order[off[query]-begin:off[query+1]-begin]
                for rank,(left,right) in enumerate(zip(local[:-1],local[1:])):
                    lo,hi=sorted((int(left-off[query]),int(right-off[query])));index=pair_offsets[query]+hi*(hi-1)//2+lo
                    value=f(f(f(.15)*f(f(decay)**f(rank)))*f(abs(y[left]-y[right]))/f(permutations))
                    matrix[index]=f(matrix[index]+value)
        q=next_q
    raw=matrix.copy()
    if bootstrap:
        u=uniforms(len(a),seed=seed,iteration=absolute)
        raw=f(raw*np.power(-np.log(f(u+f(1e-20))),f(temperature),dtype=np.float32))
    mass=np.where(abs(raw)>f(1e-20),f(raw*w[a]),f(0)).astype(np.float32)
    ax,ay=f(exp[a]+f(1e-20)),f(exp[b]+f(1e-20));g=f(f(mass*np.where(y[a]>y[b],ay,-ax))/f(ax+ay))
    gradient=np.zeros(n);np.add.at(gradient,a,g);np.add.at(gradient,b,-g)
    docs=np.arange(n,dtype=np.uint32) if document_ids is None else np.asarray(document_ids,np.uint32)
    return dict(exponent=exp,matrix=matrix,pairs=np.column_stack((docs[a],docs[b])),edges=np.column_stack((g,mass,mass,np.zeros(len(a),np.float32))),gradient=gradient)


@pytest.mark.parametrize('sizes',[[1],[2],[17,31,5],[255,256,257],[1023],[300,700,25,1023,1],[1]*1025,[2]*512,[1,3]*256])
@pytest.mark.parametrize('permutations,decay',[(1,.85),(7,.45),(10,1),(3,0)])
def test_seeded_pfound_matrices_and_gradients_match_explicit_pairs(probe,sizes,permutations,decay):
    rng=np.random.default_rng(sum(sizes)+permutations);off=np.r_[0,np.cumsum(sizes)];n=int(off[-1])
    y=rng.uniform(0,1,n).astype(np.float32);w=rng.uniform(.2,3,n).astype(np.float32);w[::19]=0;x=rng.normal(0,1,n).astype(np.float32)
    kwargs=dict(permutations=permutations,decay=decay,seed=0xf1234567abcd9765,document_ids=np.arange(n,dtype=np.uint32)*3+7)
    actual=probe(y,w,x,off,**kwargs);expected=reference(y,w,x,off,**kwargs)
    np.testing.assert_array_equal(actual['pairs'],expected['pairs'])
    for key in ('exponent','matrix','edges','gradient'):
        np.testing.assert_allclose(actual[key],expected[key],rtol=5e-5,atol=2e-7,err_msg=key)
    assert abs(actual['gradient'].sum())<2e-7


@pytest.mark.parametrize('seed',[0,1,2**32-1,2**32,2**64-1])
@pytest.mark.parametrize('temperature',[0,.7,1])
def test_bayesian_pair_weight_sampling_is_applied_before_query_weights(probe,seed,temperature):
    rng=np.random.default_rng(319);y=rng.uniform(0,1,129);w=rng.uniform(.2,3,129);x=rng.normal(size=129);off=[0,11,64,129]
    kwargs=dict(bootstrap=1,temperature=temperature,seed=seed,absolute=17)
    actual=probe(y,w,x,off,**kwargs);expected=reference(y,w,x,off,**kwargs)
    for key in actual:np.testing.assert_allclose(actual[key],expected[key],rtol=5e-5,atol=3e-7,err_msg=key)


@pytest.mark.parametrize('labels',[[.2,.8],[.8,.2]])
def test_pair_mass_uses_lower_document_weight_not_relevance_weight(probe,labels):
    actual=probe(labels,[2,9],[0,0],[0,2],permutations=1)
    np.testing.assert_allclose(actual['matrix'],[.09],rtol=1e-6)
    np.testing.assert_allclose(actual['edges'][0,1:3],[.18,.18],rtol=1e-6)
    assert np.sign(actual['gradient'][0])==np.sign(labels[0]-labels[1])


def test_constant_relevance_ignores_varying_document_weights(probe):
    actual=probe(np.full(79,.5),np.linspace(.1,3,79),np.linspace(-1000,1000,79),[0,11,79])
    for key in ('matrix','edges','gradient'):np.testing.assert_array_equal(actual[key],0)


def test_exp_underflow_and_raw_zero_filter_follow_cuda_target(probe):
    y=np.array([0.,1e-21,.7,.2,.9],np.float32);w=np.array([1e20,1,1,1,1],np.float32);x=np.array([0,0,-1000,-999,0],np.float32)
    actual=probe(y,w,x,[0,2,5],permutations=7);expected=reference(y,w,x,[0,2,5],permutations=7)
    assert actual['matrix'][0]>0 and actual['edges'][0,1]==0
    for key in actual:np.testing.assert_allclose(actual[key],expected[key],rtol=5e-5,atol=2e-7,err_msg=key)


@pytest.mark.parametrize('kwargs',[{'permutations':0},{'decay':-1},{'decay':1.1},{'bootstrap':2},{'temperature':-1},{'document_ids':[4,4]}])
def test_pfound_target_rejects_invalid_metadata(probe,kwargs):
    with pytest.raises(ValueError):probe([0,1],[1,1],[0,0],[0,2],**kwargs)


@pytest.mark.parametrize('targets,weights,point',[
    ([0,np.nan],[1,1],[0,0]),([0,1],[-1,1],[0,0]),([0,1],[1,np.inf],[0,0]),([0,1],[1,1],[0,np.inf]),
    ([-3e38,3e38],[1,1],[0,0]),([0,10],[3e38,1],[0,0]),
])
def test_invalid_observations_and_gpu_mass_overflow_are_reported(probe,targets,weights,point):
    with pytest.raises(ValueError):probe(targets,weights,point,[0,2])


def test_sampled_query_limit_and_increasing_document_ids(probe):
    with pytest.raises(ValueError,match='1023'):
        probe(np.arange(1024)%2,np.ones(1024),np.zeros(1024),[0,1024])
    with pytest.raises(ValueError,match='increase'):
        probe([0,1],[1,1],[0,0],[0,2],document_ids=[8,4])
