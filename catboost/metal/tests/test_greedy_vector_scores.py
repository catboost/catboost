"""CUDA greedy vector scores with float histograms and expanded parent stats."""
import ctypes as ct
import hashlib
from pathlib import Path
import subprocess
import numpy as np
import pytest
from test_greedy_kernels import GreedyParams,SPLIT_DTYPE


def expanded(value):
    high=np.asarray(value,np.float32)
    return np.stack((high,np.float32(np.asarray(value,float)-high.astype(float))),axis=-1)


@pytest.fixture(scope='module')
def probe():
    tests=Path(__file__).parent;native=tests.parent/'native'
    sources=[tests/'greedy_vector_probe.mm',tests/'greedy_probe.mm',
        native/'metal_greedy_kernels.h',native/'metal_multiclass_scores.h',native/'metal_greedy_vector_scores.h']
    digest=hashlib.sha256(b''.join(p.read_bytes() for p in sources)).hexdigest()[:20]
    library=tests.parent/'.build'/('greedy_vector_probe_'+digest+'.dylib')
    if not library.exists():
        result=subprocess.run(['xcrun','clang++','-std=c++17','-O2','-fobjc-arc','-dynamiclib',
            '-framework','Foundation','-framework','Metal',str(sources[0]),'-o',str(library)],capture_output=True,text=True)
        assert result.returncode==0,result.stderr
    lib=ct.CDLL(str(library));lib.cbm_greedy_score_probe.argtypes=[ct.POINTER(GreedyParams)]+[ct.c_void_p]*12+[ct.c_uint32]
    lib.cbm_greedy_score_probe.restype=ct.c_int
    def run(sums,weights,parent_sums,parent_weights,score,missing,noise,padding=0,groups=2):
        d,l,b=sums.shape
        # Seven feature tiles of four bins. Mix equality and prefix candidates.
        features=np.repeat(np.arange(b//4,dtype=np.uint32),4);bins=np.tile(np.arange(4,dtype=np.uint32),b//4)
        offsets=np.arange(0,b+1,4,dtype=np.uint32);types=np.uint8(features%2)
        fw=np.linspace(.6,1.4,len(offsets)-1,dtype=np.float32);n=np.full(len(fw),noise,np.float32)
        hs=l*b+padding;ls=l+padding
        h=np.full((d,hs),np.nan,np.float32);h[:,:l*b]=sums.reshape(d,-1)
        parent=np.full((d,ls,2),np.nan,np.float32);parent[:,:l]=expanded(parent_sums)
        p=GreedyParams(1,len(fw),l,b,b,groups,score,1,16,l,0,d,hs,ls,missing,0,.73,0,0,0)
        output=np.zeros(l,SPLIT_DTYPE);error=ct.create_string_buffer(4096)
        arrays=[h,np.ascontiguousarray(weights,np.float32),parent,expanded(parent_weights),features,bins,types,offsets,fw,n,output]
        result=lib.cbm_greedy_score_probe(ct.byref(p),*[a.ctypes.data for a in arrays],error,len(error))
        assert not result,error.value.decode()
        return output,fw
    return run


def score_terms(terms,kind,noise,l2=.73):
    total=np.float32(0);numerator=0.;denominator=float(np.float32(1e-10));l2=float(np.float32(l2))
    for g,w in terms:
        if kind==1:
            mu=g/(w+l2) if w>0 else 0
            numerator+=g*mu;denominator+=w*mu*mu
            continue
        term=0.
        if kind==0 and w>1e-20:term=-g*g/(w+l2)
        elif kind==4 and w>1e-20:term=-g*g*(1+2*np.log1p(w))/w
        elif kind==5 and w>0:
            adjustment=np.float32(w/(w-1)) if w>1 else np.float32(0)
            adjustment=np.float32(adjustment*adjustment);term=float(adjustment)*(-g*g)/w
        elif kind==6 and w>0:
            adjustment=np.float32(w*(w-2)/(w*w-3*w+1)) if w>2 else np.float32(0)
            term=float(adjustment)*(-g*g/w)
        total=np.float32(float(total)+term)
    return np.float32(np.float32(-numerator/np.sqrt(denominator))+np.float32(noise)) if kind==1 else total


def reference(sums,weights,parents,parent_weights,kind,missing,noise,feature_weights):
    dimensions,leaves,bins=sums.shape;gains=np.empty((leaves,bins),np.float32)
    for leaf in range(leaves):
        for candidate in range(bins):
            left_w=max(float(weights[leaf,candidate]),0.)
            right_w=max(float(np.float32(parent_weights[leaf]-left_w)),0.)
            if min(left_w,right_w)<1e-20:gains[leaf,candidate]=0;continue
            before=[];after=[];left_total=0.;parent_total=0.
            for k in range(dimensions):
                g=float(sums[k,leaf,candidate]);parent=parents[k,leaf]
                other=float(np.float32(parent-g))
                before.append((parent,parent_weights[leaf]));after.extend(((g,left_w),(other,right_w)))
                left_total+=g;parent_total+=parent
            if missing:
                before.append((-parent_total,parent_weights[leaf]))
                after.extend(((-left_total,left_w),(-(parent_total-left_total),right_w)))
            gains[leaf,candidate]=np.float32(np.float32(score_terms(after,kind,noise)-score_terms(before,kind,noise))*feature_weights[candidate//4])
    return gains


@pytest.mark.parametrize('dimensions',[1,2,3,7,31,63])
@pytest.mark.parametrize('kind',[0,1,4,5,6])
@pytest.mark.parametrize('missing',[False,True])
@pytest.mark.parametrize('noise',[0.,.31])
def test_vector_greedy_scores_match_cuda_mixed_precision_equations(probe,dimensions,kind,missing,noise):
    rng=np.random.default_rng(461853+dimensions);leaves,bins=3,28
    parents=rng.normal(0,7,(dimensions,leaves));parent_weights=rng.uniform(4,30,leaves)
    sums=np.float32(rng.normal(0,4,(dimensions,leaves,bins)))
    weights=np.float32(rng.uniform(.01,.99,(leaves,bins))*parent_weights[:,None])
    weights[0,0]=0;weights[1,4]=np.float32(parent_weights[1]+1)
    actual,fw=probe(sums,weights,parents,parent_weights,kind,missing,noise,padding=13)
    expected=reference(sums,weights,parents,parent_weights,kind,missing,noise,fw)
    winners=expected.argmin(axis=1)
    np.testing.assert_array_equal(actual['index'],winners)
    np.testing.assert_allclose(actual['gain'],expected[np.arange(leaves),winners],rtol=5e-6,atol=3e-6)
    np.testing.assert_array_equal(actual['error'],0)


@pytest.mark.parametrize('kind',[0,1,4,5,6])
@pytest.mark.parametrize('parent_weight',[1+2**-26,2+2**-24,2.6180339,1e20])
def test_parent_weight_residuals_survive_thresholds_and_sat_pole(probe,kind,parent_weight):
    parents=np.array([[.31],[.7]]);pw=np.array([parent_weight])
    sums=np.float32([[[.2,.3,.1,.4]],[[.05,.27,.1,.2]]])
    weights=np.float32([[.13,.25,.49,.67]])*np.float32(parent_weight)
    actual,fw=probe(sums,weights,parents,pw,kind,False,0,padding=7)
    expected=reference(sums,weights,parents,pw,kind,False,0,fw)
    winner=expected.argmin(axis=1)
    np.testing.assert_array_equal(actual['index'],winner)
    np.testing.assert_allclose(actual['gain'],expected[np.arange(1),winner],rtol=1e-5,atol=1e-7)
    assert not actual['error'].any()


@pytest.mark.parametrize('kind',[0,1,4,5,6])
@pytest.mark.parametrize('gradient,weight',[(1e25,1e20),(1e-25,1.0000001192092896),(1e-15,1e-18)])
def test_score_terms_keep_cuda_exponent_range(probe,kind,gradient,weight):
    parents=np.array([[gradient],[gradient*.5]]);pw=np.array([weight*4])
    sums=np.float32([[[gradient*.2,gradient*.7,gradient*.3,gradient*.4]],
                       [[gradient*.5,gradient*.1,gradient*.4,gradient*.3]]])
    weights=np.float32([[weight,weight*2,weight*3,weight*.5]])
    actual,fw=probe(sums,weights,parents,pw,kind,True,0)
    expected=reference(sums,weights,parents,pw,kind,True,0,fw);winner=expected.argmin(axis=1)
    np.testing.assert_array_equal(actual['index'],winner)
    np.testing.assert_allclose(actual['gain'],expected[np.arange(1),winner],rtol=1e-5,atol=np.finfo(np.float32).smallest_subnormal*2)
    assert not actual['error'].any()


@pytest.mark.parametrize('gradient',[1e20,1e25,1e30])
def test_cosine_double_intermediates_can_exceed_float_max(probe,gradient):
    parents=np.array([[gradient],[-gradient*.5]]);pw=np.array([5.])
    sums=np.float32([[[gradient*.8,gradient*.1,gradient*.4,gradient*.2]],
                       [[-gradient*.1,-gradient*.5,-gradient*.3,-gradient*.2]]])
    weights=np.float32([[1.,2.,3.,4.]])
    actual,fw=probe(sums,weights,parents,pw,1,True,0)
    expected=reference(sums,weights,parents,pw,1,True,0,fw);winner=expected.argmin(axis=1)
    np.testing.assert_array_equal(actual['index'],winner)
    np.testing.assert_allclose(actual['gain'],expected[np.arange(1),winner],rtol=5e-6)
    assert not actual['error'].any()
