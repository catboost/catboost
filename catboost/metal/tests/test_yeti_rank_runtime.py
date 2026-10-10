"""Resident YetiRank GPU target, caller-owned commands and explicit oracle seeds."""
from contextlib import contextmanager
import ctypes as ct
import hashlib
from pathlib import Path
import platform
import subprocess

import numpy as np
import pytest
from test_yeti_rank_kernels import reference


@pytest.fixture(scope='module')
def runtime():
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        pytest.skip('requires Apple Silicon Metal')
    root=Path(__file__).resolve().parents[1]
    files=[root/'tests/yeti_rank_runtime_probe.mm',root/'native/metal_yeti_rank_runtime.h',root/'native/metal_yeti_rank_kernels.h']
    digest=hashlib.sha256(b''.join(p.read_bytes() for p in files)).hexdigest()[:20]
    path=root/'.build'/('yeti_rank_runtime_'+digest+'.dylib')
    if not path.exists():
        subprocess.run(['xcrun','clang++','-std=c++17','-O2','-fobjc-arc','-dynamiclib',
                        '-framework','Foundation','-framework','Metal',str(files[0]),'-o',str(path)],
                        capture_output=True,text=True,check=True)
    lib=ct.CDLL(str(path));f32=ct.POINTER(ct.c_float);u32=ct.POINTER(ct.c_uint32)
    lib.cbm_yeti_runtime_create.argtypes=[ct.c_uint32,ct.c_uint32,u32,ct.c_uint32,ct.c_float,ct.c_uint32,
                                        ct.POINTER(ct.c_uint64),ct.c_char_p,ct.c_uint32]
    lib.cbm_yeti_runtime_create.restype=ct.c_void_p
    lib.cbm_yeti_runtime_destroy.argtypes=[ct.c_void_p]
    lib.cbm_yeti_runtime_center.argtypes=[ct.c_void_p,f32,ct.c_uint32,ct.c_char_p,ct.c_uint32]
    lib.cbm_yeti_runtime_center.restype=ct.c_int
    lib.cbm_yeti_runtime_step.argtypes=[ct.c_void_p,f32,f32,ct.c_uint32,u32,f32,f32,ct.c_uint32,ct.c_uint64,
                                      ct.c_uint32,f32,f32,ct.POINTER(ct.c_uint64),ct.c_char_p,ct.c_uint32]
    lib.cbm_yeti_runtime_step.restype=ct.c_int

    @contextmanager
    def create(offsets,permutations=10,decay=.85,legacy=False):
        off=np.ascontiguousarray(offsets,np.uint32);n=int(off[-1]);message=ct.create_string_buffer(4096)
        size=ct.c_uint64()
        handle=lib.cbm_yeti_runtime_create(n,len(off)-1,off.ctypes.data_as(u32),permutations,decay,legacy,
                                         ct.byref(size),message,len(message))
        if not handle:raise ValueError(message.value.decode())
        class Session:
            allocated_bytes=size.value
            last_dispatches=0
            def center(self,leaves):
                values=np.array(leaves,dtype=np.float32,copy=True)
                code=lib.cbm_yeti_runtime_center(handle,values.ctypes.data_as(f32),len(values),message,len(message))
                if code:raise ValueError(message.value.decode())
                return values
            def step(self,labels,weights,cursor,leaves=None,ids=None,seed=147193,apply=False,mode=0):
                y,w,x=[np.ascontiguousarray(v,np.float32) for v in (labels,weights,cursor)]
                leaf=np.ascontiguousarray([0] if leaves is None else leaves,np.float32)
                indices=np.ascontiguousarray(np.zeros(n) if ids is None else ids,np.uint32)
                assert y.shape==w.shape==x.shape==indices.shape==(n,)
                g=np.empty(n,np.float32);mass=np.empty(n,np.float32);count=ct.c_uint64()
                result=lib.cbm_yeti_runtime_step(handle,x.ctypes.data_as(f32),leaf.ctypes.data_as(f32),len(leaf),
                    indices.ctypes.data_as(u32),y.ctypes.data_as(f32),w.ctypes.data_as(f32),apply,seed,mode,
                    g.ctypes.data_as(f32),mass.ctypes.data_as(f32),ct.byref(count),message,len(message))
                self.last_dispatches=count.value
                if result:raise ValueError(message.value.decode())
                assert count.value==4
                return np.column_stack((g,mass))
        try:yield Session()
        finally:lib.cbm_yeti_runtime_destroy(handle)
    return create


@pytest.mark.parametrize('sizes',[[1],[19,31,7],[1023],[300,700,25,1023,1]])
@pytest.mark.parametrize('apply',[False,True])
@pytest.mark.parametrize('legacy',[False,True])
def test_resident_derivatives_apply_original_row_leaf_updates(runtime,sizes,apply,legacy):
    offsets=np.r_[0,np.cumsum(sizes)];n=int(offsets[-1]);rng=np.random.default_rng(n)
    x=rng.normal(size=n).astype(np.float32);y=rng.uniform(0,1,n).astype(np.float32)
    w=rng.uniform(0,3,n).astype(np.float32);w[::17]=0
    leaves=np.array([.1,-.2,.7],np.float32);ids=rng.integers(0,len(leaves),n,dtype=np.uint32)
    point=np.float32(x+leaves[ids]) if apply else x
    with runtime(offsets,permutations=7,decay=.45,legacy=legacy) as session:
        expected=reference(y,w,point,offsets,permutations=7,decay=.45,seed=0xfedcba9812345678,
                           center_rows=len(sizes) if legacy else None)[1]
        actual=session.step(y,w,x,leaves,ids,seed=0xfedcba9812345678,apply=apply)
        np.testing.assert_allclose(actual,expected,rtol=4e-5,atol=3e-7)
        assert session.allocated_bytes<=24*n+12*len(sizes)+8
        assert session.allocated_bytes>=24*n+4*(len(sizes)+1)+12
        np.testing.assert_array_equal(session.step(y,w,x,leaves,ids,seed=0xfedcba9812345678,apply=apply),actual)


def test_explicit_seeds_replay_without_hidden_random_consumption(runtime):
    rng=np.random.default_rng(179);y=rng.uniform(size=79);w=np.ones(79);x=rng.normal(size=79)
    with runtime([0,31,79]) as session:
        first=session.step(y,w,x,seed=47)
        second=session.step(y,w,x,seed=48)
        assert not np.array_equal(first,second)
        np.testing.assert_array_equal(session.step(y,w,x,seed=47),first)
    with runtime([0,31,79]) as resumed:
        np.testing.assert_array_equal(resumed.step(y,w,x,seed=48),second)


@pytest.mark.parametrize('case',['nan_cursor','overflow_point','nan_target','negative_weight','overflow_weighted_target','leaf_id'])
def test_gpu_validation_is_sticky_and_recoverable_at_explicit_boundary(runtime,case):
    y=np.array([.1,.9,.4,.7],np.float32);w=np.ones(4,np.float32);x=np.zeros(4,np.float32)
    leaf=np.array([0.,.5],np.float32);ids=np.array([0,1,0,1],np.uint32)
    invalid=[a.copy() for a in (y,w,x,leaf,ids)];a,b,c,d,e=invalid
    if case=='nan_cursor':c[0]=np.nan
    elif case=='overflow_point':c[1]=np.finfo(np.float32).max;d[1]=np.finfo(np.float32).max
    elif case=='nan_target':a[0]=np.nan
    elif case=='negative_weight':b[0]=-1
    elif case=='overflow_weighted_target':a[0]=np.finfo(np.float32).max;b[0]=2
    else:e[1]=2
    with runtime([0,4]) as session:
        with pytest.raises(ValueError,match='Invalid YetiRank GPU'):
            session.step(a,b,c,d,e,apply=True)
        assert session.last_dispatches==4
        with pytest.raises(ValueError,match='Invalid YetiRank GPU'):
            session.step(y,w,x,leaf,ids,apply=True)
        restored=session.step(y,w,x,leaf,ids,apply=True,mode=1)
        expected=reference(y,w,np.float32(x+leaf[ids]),[0,4])[1]
        np.testing.assert_allclose(restored,expected,rtol=4e-5,atol=3e-7)


def test_alias_rejected_before_encoding_without_damaging_target(runtime):
    with runtime([0,3]) as session:
        with pytest.raises(ValueError,match='cannot alias'):
            session.step([0,.5,1],[1]*3,[0]*3,mode=2)
        assert session.last_dispatches==0
        assert np.isfinite(session.step([0,.5,1],[1]*3,[0]*3)).all()


@pytest.mark.parametrize('leaves',[1,2,7,256,1025,65536])
def test_zero_average_includes_empty_leaf_coordinates(runtime,leaves):
    rng=np.random.default_rng(leaves);values=rng.normal(size=leaves).astype(np.float32)
    values[::3]=0
    expected=(values.astype(float)-values.mean(dtype=float)).astype(np.float32)
    with runtime([0,2]) as session:
        np.testing.assert_allclose(session.center(values),expected,rtol=2e-6,atol=3e-7)


def test_nonfinite_leaf_centering_is_an_explicit_gpu_error(runtime):
    with runtime([0,2]) as session:
        with pytest.raises(ValueError,match='Invalid YetiRank GPU'):
            session.center([np.nan,1.,0.])


@pytest.mark.parametrize('offsets,options', [([0,1024],{}),([0,2,1,3],{}),([0,0,3],{}),
    ([1,3],{}),([0,3],{'permutations':0}),([0,3],{'decay':np.inf})])
def test_invalid_layout_and_configuration_rejected(runtime,offsets,options):
    with pytest.raises(ValueError):
        with runtime(offsets,**options):pass
