"""GPU query/document sampling, stable compaction and bounded pair prefixes."""
import ctypes as ct
import hashlib
from pathlib import Path
import platform,subprocess
import numpy as np
import pytest

class Params(ct.Structure):
    _fields_=[(name,ct.c_uint32) for name in ('rows','groups','passes','max_query','pair_limit','allow_failures','r0','r1')]

@pytest.fixture(scope='module')
def sampler():
    if platform.system()!='Darwin' or platform.machine()!='arm64':pytest.skip('Apple GPU required')
    root=Path(__file__).resolve().parents[1];source=Path(__file__).with_name('query_sampler_probe.mm')
    deps=[source,*[root/'native'/name for name in ('metal_query_sampler_runtime.h','metal_query_sampler_kernels.h','metal_sort.h','metal_sort.mm','metal_sort_kernels.h')]]
    digest=hashlib.sha256(b''.join(p.read_bytes() for p in deps)).hexdigest()[:20];output=root/'.build'/('query_sampler_'+digest+'.dylib')
    if not output.exists():
        subprocess.run(['xcrun','clang++','-std=c++17','-O2','-fobjc-arc','-dynamiclib','-framework','Foundation','-framework','Metal',str(source),str(root/'native/metal_sort.mm'),'-o',str(output)],capture_output=True,text=True,check=True)
    lib=ct.CDLL(str(output));lib.cbm_query_sampler_probe.argtypes=[ct.POINTER(Params),ct.c_uint64]+[ct.c_void_p]*12+[ct.c_uint32]
    lib.cbm_query_sampler_probe.restype=ct.c_int
    def run(offsets,keys,masks,fraction,*,max_query=0,pair_limit=2**24,budget=2**30,allow_failures=False):
        off=np.ascontiguousarray(offsets,np.uint32);rows=int(off[-1]);groups=len(off)-1
        fractions=np.atleast_1d(fraction).astype(np.float32);passes=len(fractions)
        keys=np.ascontiguousarray(np.broadcast_to(keys,(passes,rows)),np.uint32)
        masks=np.ascontiguousarray(np.broadcast_to(masks,(passes,groups)),np.float32)
        shapes=np.zeros((passes,4),np.uint32);docs=np.zeros((passes,rows),np.uint32);qids=np.zeros_like(docs)
        offsets_out=np.zeros((passes,groups+1),np.uint32);pair_out=np.zeros_like(offsets_out);row_masks=np.zeros_like(docs)
        params=Params(rows,groups,passes,max_query,pair_limit,allow_failures,0,0);memory=ct.c_uint64();error=ct.create_string_buffer(4096)
        code=lib.cbm_query_sampler_probe(ct.byref(params),budget,*(v.ctypes.data for v in (off,keys,masks,fractions,shapes,docs,qids,offsets_out,pair_out,row_masks)),ct.byref(memory),error,len(error))
        if code:raise ValueError(error.value.decode())
        result=[]
        for i,(n,g,p,failed) in enumerate(shapes):
            result.append(dict(documents=docs[i,:n],qids=qids[i,:n],offsets=offsets_out[i,:g+1],pair_offsets=pair_out[i,:g+1],mask=row_masks[i],pairs=int(p),failed=bool(failed)))
        return result,memory.value
    return run


def reference(offsets,keys,masks,fraction,max_query=0):
    off=np.asarray(offsets);rows=int(off[-1]);groups=len(off)-1;cap=max_query or min(2*rows//groups+8,1023)
    documents=[];new_offsets=[0];pairs=[0];qids=[];mask=np.zeros(rows,np.uint32)
    for q,(start,end) in enumerate(zip(off[:-1],off[1:])):
        size=int(end-start);taken=min(cap,max(min(2,size),int(np.ceil(np.float32(fraction)*np.float32(size))))) if masks[q]>0 else 0
        if not taken:continue
        order=np.argsort(keys[start:end],kind='stable')[:taken]+start;order.sort();documents.extend(order);mask[order]=1
        qids.extend([len(new_offsets)-1]*taken);new_offsets.append(new_offsets[-1]+taken);pairs.append(pairs[-1]+taken*(taken-1)//2)
    return dict(documents=np.array(documents,np.uint32),qids=np.array(qids,np.uint32),offsets=np.array(new_offsets,np.uint32),
        pair_offsets=np.array(pairs,np.uint32),mask=mask,pairs=int(pairs[-1]),failed=False)


@pytest.mark.parametrize('sizes',[[1],[2],[17,31,5],[255,256,257],[1023],[4097],[300,700,25,1023,1],[1]*1025,[2]*512,[1]*511+[4097],[65537]])
@pytest.mark.parametrize('fraction',[1e-9,.13,.5,1])
def test_gpu_counts_shuffle_selection_and_compaction_match_cuda_rules(sampler,sizes,fraction):
    rng=np.random.default_rng(sum(sizes));off=np.r_[0,np.cumsum(sizes)];keys=rng.integers(0,2**32,int(off[-1]),dtype=np.uint32)
    masks=(rng.random(len(sizes))>.3).astype(np.float32);masks[-1]=1
    actual,bytes=sampler(off,keys,masks,fraction);expected=reference(off,keys,masks,fraction)
    for key in expected:np.testing.assert_array_equal(actual[0][key],expected[key],err_msg=key)
    assert 0<bytes<2**30


@pytest.mark.parametrize('max_query',[2,17,1023])
def test_stable_ties_and_minimum_two_documents(sampler,max_query):
    off=np.array([0,1,2,19,84,85,1142]);keys=np.zeros(1142,np.uint32);masks=np.ones(6,np.float32)
    actual,_=sampler(off,keys,masks,[1e-9,.5,1],max_query=max_query)
    for result,fraction in zip(actual,[1e-9,.5,1]):
        for key,value in reference(off,keys,masks,fraction,max_query).items():np.testing.assert_array_equal(result[key],value,err_msg=key)


def test_repeated_workspace_recovers_from_empty_and_rejected_masks(sampler):
    off=[0,17,30,62];keys=np.arange(62,dtype=np.uint32)[::-1];masks=np.array([[1,1,1],[0,0,0],[1,np.nan,1],[1,0,1]],np.float32)
    actual,_=sampler(off,keys,masks,[.5,1,.5,.5],allow_failures=True)
    for i in (0,1,3):
        expected=reference(off,keys,masks[i],[.5,1,.5,.5][i])
        for key,value in expected.items():np.testing.assert_array_equal(actual[i][key],value,err_msg=key)
    assert actual[2]['failed']
    assert actual[1]['pairs']==0 and actual[1]['offsets'].tolist()==[0]


@pytest.mark.parametrize('groups',[1,2,257,65537])
def test_saturating_pair_prefix_reports_capacity_without_uint32_wrap(sampler,groups):
    off=np.arange(groups+1,dtype=np.uint32)*3;keys=np.arange(int(off[-1]),dtype=np.uint32);masks=np.ones(groups,np.float32)
    with pytest.raises(ValueError,match='capacity'):sampler(off,keys,masks,1,pair_limit=2)


def test_pair_capacity_rejection_does_not_poison_next_small_sample(sampler):
    off=[0,17,31];keys=np.arange(31,dtype=np.uint32);masks=np.ones(2,np.float32)
    actual,_=sampler(off,keys,masks,[1,1e-9],pair_limit=2,allow_failures=True)
    assert actual[0]['failed'] and not actual[1]['failed'] and actual[1]['pairs']==2


@pytest.mark.parametrize('option',[{'fraction':0},{'fraction':1.1},{'max_query':1},{'max_query':1024},{'budget':1}])
def test_invalid_sampler_options_are_rejected(sampler,option):
    with pytest.raises(ValueError):sampler([0,2],np.zeros(2,np.uint32),np.ones(1,np.float32),**({'fraction':1}|option))


@pytest.mark.parametrize('groups',[33,257,4110])
def test_original_pair_ids_can_exceed_24_bits_without_dense_storage(sampler,groups):
    rows=groups*1023;offsets=np.arange(groups+1,dtype=np.uint32)*1023
    actual,allocated=sampler(offsets,np.arange(rows,dtype=np.uint32),np.ones(groups,np.float32),1,pair_limit=2**32-2)
    result=actual[0];expected=np.arange(groups+1,dtype=np.uint64)*522753
    assert result['pairs']==int(expected[-1]) and result['pairs']>2**24
    np.testing.assert_array_equal(result['pair_offsets'],expected)
    np.testing.assert_array_equal(result['documents'],np.arange(rows,dtype=np.uint32))
    assert allocated<64*rows
    if groups==4110:assert result['pairs']>2**31


def test_wide_pair_limit_failure_and_recovery_on_the_same_sampler(sampler):
    groups=33;rows=groups*1023;offsets=np.arange(groups+1,dtype=np.uint32)*1023
    actual,_=sampler(offsets,np.tile(np.arange(rows,dtype=np.uint32),(2,1)),np.ones((2,groups),np.float32),[1,.1],
        pair_limit=2**24+1,allow_failures=True)
    assert actual[0]['failed'] and not actual[1]['failed']
    assert actual[1]['pairs']==groups*103*102//2


def test_sampler_rejects_the_reserved_pair_sentinel(sampler):
    with pytest.raises(ValueError,match='capacity'):
        sampler([0,2],np.zeros(2,np.uint32),np.ones(1,np.float32),1,pair_limit=2**32-1)
