"""Compact variable-tree Metal inference without fitting a CPU model."""
import ctypes as ct
import platform

import numpy as np
import pytest

from catboost_metal import _inference

TERMINAL = 2**32-1


@pytest.fixture(scope='module')
def predict():
    if platform.system()!='Darwin' or platform.machine()!='arm64':
        pytest.skip('requires Apple Silicon Metal')
    lib=ct.CDLL(str(_inference.build_library()))
    fn=lib.cbm_predict_non_symmetric_bins_multidim
    fn.argtypes=[ct.POINTER(_inference.InferenceParams),ct.c_uint32]+[ct.c_void_p,ct.c_uint64]*6+[
        ct.POINTER(_inference.InferenceStats),ct.c_char_p,ct.c_size_t]
    fn.restype=ct.c_int
    def run(bins,roots,nodes,leaves,*,bias=None,scale=1,begin=0,end=None,batch=97):
        bins=np.ascontiguousarray(bins,np.uint8);roots=np.ascontiguousarray(roots,np.uint32)
        nodes=np.ascontiguousarray(nodes,np.uint32).reshape(-1,6)
        leaves=np.ascontiguousarray(leaves,np.float64)
        if leaves.ndim==1:leaves=leaves[:,None]
        dimensions=leaves.shape[1]
        biases=np.ascontiguousarray(np.zeros(dimensions) if bias is None else bias,np.float64)
        result=np.empty((bins.shape[1],dimensions),np.float64)
        p=_inference.InferenceParams(bins.shape[1],bins.shape[0],len(roots),0,0,begin,
            len(roots) if end is None else end,batch,scale,0)
        stats=_inference.InferenceStats();error=ct.create_string_buffer(4096)
        arrays=[(biases,biases.size),(bins,bins.size),(roots,len(roots)),(nodes,len(nodes)),
                (leaves,leaves.size),(result,result.size)]
        args=[item for arr,count in arrays for item in (arr.ctypes.data,count)]
        if fn(ct.byref(p),dimensions,*args,ct.byref(stats),error,len(error)):
            raise ValueError(error.value.decode())
        return result,stats
    return run


def forest(trees,dimensions):
    rng=np.random.default_rng(765)
    roots=[];nodes=[];leaves=[]
    for tree in range(trees):
        root=len(nodes);base=len(leaves);roots.append(root)
        nodes.extend([[tree%3,tree%4,tree%2,root+1,root+2,TERMINAL],
                      [0,0,0,0,0,base],
                      [(tree+1)%3,1,0,root+3,root+4,TERMINAL],
                      [0,0,0,0,0,base+1],[0,0,0,0,0,base+2]])
        leaves.extend(rng.normal(size=(3,dimensions)))
    return np.array(roots,np.uint32),np.array(nodes,np.uint32),np.array(leaves)


def reference(bins,roots,nodes,leaves,begin,end):
    output=np.zeros((bins.shape[1],leaves.shape[1]))
    for row in range(bins.shape[1]):
        for root in roots[begin:end]:
            index=int(root)
            while int(nodes[index,5])==TERMINAL:
                feature,border,kind,left,right,_=map(int,nodes[index])
                value=int(bins[feature,row])
                index=right if (value==border if kind else value>border) else left
            output[row]+=leaves[nodes[index,5]]
    return output


@pytest.mark.parametrize('dimensions',[1,2,7,64])
@pytest.mark.parametrize('begin,end',[(0,73),(0,17),(3,69),(70,73)])
def test_mixed_variable_trees_dimensions_ranges_and_batching(predict,dimensions,begin,end):
    roots,nodes,leaves=forest(73,dimensions)
    bins=np.random.default_rng(723).integers(0,7,(3,1031),dtype=np.uint8)
    bias=np.linspace(-.4,.7,dimensions);scale=1.2718281828459
    actual,stats=predict(bins,roots,nodes,leaves,bias=bias,scale=scale,begin=begin,end=end)
    expected=reference(bins,roots,nodes,leaves,begin,end)*scale+(bias if begin==0 else 0)
    np.testing.assert_allclose(actual,expected,rtol=2e-12,atol=2e-12)
    assert stats.kernel_dispatches==22 and stats.device_name.startswith(b'Apple')


@pytest.mark.parametrize('depth',[30,100,1500,65535])
def test_deep_chain_storage_is_linear(predict,depth):
    nodes=[]
    for level in range(depth):
        nodes.extend([[0,0,0,2*level+2,2*level+1,TERMINAL],[0,0,0,0,0,level]])
    nodes.append([0,0,0,0,0,depth])
    leaves=np.arange(depth+1,dtype=float)/3
    bins=np.array([[0,1,0,1]],np.uint8)
    actual,_=predict(bins,[0],nodes,leaves)
    np.testing.assert_allclose(actual[:,0],[depth/3,0,depth/3,0],rtol=2e-13,atol=2e-13)
    assert len(nodes)==2*depth+1


def test_constant_forest_compensates_cancellation_without_features(predict):
    leaves=np.tile([1e10,1.234567890123,-1e10],1001)
    nodes=np.zeros((len(leaves),6),np.uint32);nodes[:,5]=np.arange(len(leaves),dtype=np.uint32)
    actual,_=predict(np.empty((0,7),np.uint8),np.arange(len(leaves)),nodes,leaves)
    np.testing.assert_allclose(actual[:,0],1001*1.234567890123,rtol=3e-6)


def test_empty_rows_and_ranges_do_not_dispatch(predict):
    roots,nodes,leaves=forest(3,2)
    empty,stats=predict(np.empty((3,0),np.uint8),roots,nodes,leaves)
    assert empty.shape==(0,2) and stats.kernel_dispatches==0
    zeros,stats=predict(np.zeros((3,7),np.uint8),roots,nodes,leaves,begin=2,end=2,bias=[1,2])
    np.testing.assert_array_equal(zeros,0)
    assert stats.kernel_dispatches==0


@pytest.mark.parametrize('case',['cycle','shared','unreachable','root','leaf','feature','border','type','nonfinite'])
def test_native_graph_validation_rejects_unsafe_walks(predict,case):
    roots,nodes,leaves=forest(1,1)
    if case=='cycle':nodes[0,3]=0
    if case=='shared':nodes[0,3]=nodes[0,4]
    if case=='unreachable':nodes=np.vstack([nodes,[0,0,0,0,0,0]])
    if case=='root':roots[0]=99
    if case=='leaf':nodes[1,5]=99
    if case=='feature':nodes[0,0]=3
    if case=='border':nodes[0,1]=255
    if case=='type':nodes[0,2]=2
    if case=='nonfinite':leaves[0]=np.nan
    with pytest.raises(ValueError):predict(np.zeros((3,2),np.uint8),roots,nodes,leaves)
