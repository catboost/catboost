"""Bounded radix scratch reused within and between caller-owned commands."""
import ctypes as ct
import hashlib
from pathlib import Path
import platform
import subprocess
import numpy as np
import pytest


@pytest.fixture(scope='module')
def workspace():
    if platform.system()!='Darwin' or platform.machine()!='arm64':pytest.skip('Apple GPU required')
    source=Path(__file__).with_name('sort_workspace_probe.mm');root=source.parent.parent
    deps=[source,*(root/'native'/n for n in ('metal_sort.h','metal_sort.mm','metal_sort_kernels.h'))]
    digest=hashlib.sha256(b''.join(p.read_bytes() for p in deps)).hexdigest()[:20]
    path=root/'.build'/('sort_workspace_'+digest+'.dylib')
    if not path.exists():
        subprocess.run(['xcrun','clang++','-std=c++17','-O2','-fobjc-arc','-dynamiclib','-framework','Foundation',
            '-framework','Metal',str(source),str(root/'native/metal_sort.mm'),'-o',str(path)],capture_output=True,text=True,check=True)
    lib=ct.CDLL(str(path))
    lib.cbm_sort_workspace_probe.argtypes=[ct.c_uint32,ct.c_uint32]+[ct.c_void_p]*3+[ct.c_uint32]*3+[ct.c_void_p]*5+[ct.c_uint32]
    lib.cbm_sort_workspace_probe.restype=ct.c_int
    def run(keys,*,capacity=None,mode=0,in_place=False,low_bits=32):
        counts=np.array([len(a) for a in keys],np.uint32)
        flat=np.ascontiguousarray(np.concatenate(keys),np.uint32)
        payload=np.random.default_rng(138).integers(0,2**32,len(flat),dtype=np.uint32)
        output=np.zeros_like(flat);carried=np.zeros_like(payload)
        capacity=int(max(counts)) if capacity is None else capacity
        size,dispatches=ct.c_uint64(),ct.c_uint64();error=ct.create_string_buffer(4096)
        code=lib.cbm_sort_workspace_probe(capacity,len(counts),counts.ctypes.data,flat.ctypes.data,payload.ctypes.data,
            mode,in_place,low_bits,output.ctypes.data,carried.ctypes.data,ct.byref(size),ct.byref(dispatches),error,len(error))
        if code:raise ValueError(error.value.decode())
        start=0
        for values in keys:
            end=start+len(values);order=np.argsort(np.asarray(values,np.uint32)&((1<<low_bits)-1),kind='stable')
            np.testing.assert_array_equal(output[start:end],flat[start:end][order])
            np.testing.assert_array_equal(carried[start:end],payload[start:end][order]);start=end
        expected=12*capacity
        if capacity:
            elements=((capacity+255)//256)*16
            while True:
                expected+=4*elements
                if elements==1:break
                elements=(elements+255)//256
        assert size.value==expected
        return size.value,dispatches.value
    return run


@pytest.mark.parametrize('mode',[0,1])
@pytest.mark.parametrize('in_place',[False,True])
@pytest.mark.parametrize('pattern',['ties','random','descending'])
def test_variable_batch_sizes_reuse_exact_allocation_and_keep_stable_payload_order(workspace,mode,in_place,pattern):
    counts=[65539,0,1,255,256,257,4097,17,65539]
    rng=np.random.default_rng(139)
    keys=[(rng.integers(0,7 if pattern=='ties' else 2**32,n,dtype=np.uint32) if pattern!='descending'
           else np.arange(n,0,-1,dtype=np.uint32)) for n in counts]
    workspace(keys,mode=mode,in_place=in_place)


def test_capacity_and_pending_command_ownership_are_checked(workspace):
    with pytest.raises(ValueError,match='workspace capacity'):workspace([[3,2,1]],capacity=2)
    with pytest.raises(ValueError,match='another command'):workspace([[3,2,1],[2,1]],mode=2)
    with pytest.raises(ValueError,match='at most'):workspace([[]],capacity=2**24+1)
    assert workspace([[],[]],capacity=0)==(0,0)


@pytest.mark.parametrize('bits',[4,8,12,16,20,24,28,32])
@pytest.mark.parametrize('mode',[0,1])
@pytest.mark.parametrize('in_place',[False,True])
def test_low_key_bits_sort_stably_with_odd_even_passes_and_aliases(workspace,bits,mode,in_place):
    rng=np.random.default_rng(bits+193)
    keys=[rng.integers(0,2**32,n,dtype=np.uint32) for n in (65539,0,1,257,17)]
    workspace(keys,low_bits=bits,mode=mode,in_place=in_place)


@pytest.mark.parametrize('bits',[0,1,3,5,31,33,36])
def test_invalid_low_key_bit_ranges_are_rejected(workspace,bits):
    with pytest.raises(ValueError,match='key bits'):workspace([[2,1]],low_bits=bits)


def test_flag_compaction_needs_one_eighth_the_radix_dispatches(workspace):
    keys=[np.arange(65539,dtype=np.uint32)%2]
    short=workspace(keys,low_bits=4);full=workspace(keys)
    assert short[0]==full[0] and short[1]*8==full[1]
