"""Pairwise split solver, original-H quadratic score, and winner selection."""
import ctypes as ct
import hashlib
from pathlib import Path
import platform
import subprocess

import numpy as np
import pytest

from test_leaf_matrix_kernels import MatrixParams, laplacian


@pytest.fixture(scope='module')
def probes():
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        pytest.skip('requires Apple Silicon Metal')
    source=Path(__file__).with_name('pairwise_score_probe.mm'); root=source.parent.parent
    headers=[root/'native'/name for name in ('metal_leaf_matrix_kernels.h','metal_pairwise_score_kernels.h')]
    digest=hashlib.sha256(source.read_bytes()+b''.join(p.read_bytes() for p in headers)).hexdigest()[:20]
    path=root/'.build'/('pairwise_score_'+digest+'.dylib')
    if not path.exists():
        subprocess.run(['xcrun','clang++','-std=c++17','-O2','-fobjc-arc','-dynamiclib',
                        '-framework','Foundation','-framework','Metal',str(source),'-o',str(path)],
                       capture_output=True,text=True,check=True)
    lib=ct.CDLL(str(path))
    lib.cbm_pairwise_score_probe.argtypes=[ct.POINTER(MatrixParams)]+[ct.c_void_p]*7+[ct.c_uint32]
    lib.cbm_pairwise_score_probe.restype=ct.c_int
    lib.cbm_pairwise_selection_probe.argtypes=[ct.c_uint32,ct.c_uint32,ct.c_float]+[ct.c_void_p]*6+[ct.c_uint32]
    lib.cbm_pairwise_selection_probe.restype=ct.c_int

    def score(h,g,*,diagonal=False,l2=3.,non_diag=.2):
        h,g=np.ascontiguousarray(h,np.float32),np.ascontiguousarray(g,np.float32)
        n=len(g);assert h.shape==(n,n)
        work=np.zeros((n,n,2),np.float32);direction=np.zeros(n,np.float32);result=np.zeros(2,np.float32)
        status=ct.c_uint32();error=ct.create_string_buffer(4096)
        p=MatrixParams(n,diagonal,0,0,l2,non_diag,0,1)
        code=lib.cbm_pairwise_score_probe(ct.byref(p),*(a.ctypes.data for a in (h,g,work,direction,result)),
            ct.byref(status),error,len(error))
        if code:raise ValueError(error.value.decode())
        return work.sum(axis=-1,dtype=float),direction,result.sum(dtype=float),status.value

    def select(values,features,weights,previous=0):
        values=np.asarray(values,np.float32)
        if values.ndim==1:values=np.column_stack((values,np.zeros_like(values)))
        arrays=[np.ascontiguousarray(values,np.float32),np.ascontiguousarray(features,np.uint32),
                np.ascontiguousarray(weights,np.float32)]
        assert arrays[0].shape==(len(features),2)
        winner=ct.c_uint32();result=np.zeros(2,np.float32);error=ct.create_string_buffer(4096)
        code=lib.cbm_pairwise_selection_probe(len(values),len(weights),previous,*(a.ctypes.data for a in arrays),
            ct.byref(winner),result.ctypes.data,error,len(error))
        if code:raise ValueError(error.value.decode())
        return winner.value,result
    return score,select


def stabilized(h,diagonal,l2,non_diag):
    h=np.asarray(h,np.float32);n=len(h);trace=np.float32(0)
    for v in np.diag(h):trace=np.float32(trace+v)
    count=np.sum(np.diag(h)>np.float32(1e-9))
    average=np.float32(trace/np.float32(count)) if count else np.float32(0)
    matrix=h.astype(float)-float(np.float32(non_diag))/n
    for row in range(n):
        if h[row,row]<=np.float32(1e-7):matrix[row,row]+=float(np.float32(average+np.float32(.1)))
        matrix[row,row]+=float(np.float32(np.float32(.05)*average+np.float32(1e-20)))
        matrix[row,row]+=float(np.float32(l2))+float(np.float32(non_diag))
    if not diagonal:matrix[-1,:]=0;matrix[:,-1]=0
    return matrix


@pytest.mark.parametrize('n',[1,2,4,8,32,128,256])
@pytest.mark.parametrize('diagonal',[False,True])
def test_full_candidate_direction_and_original_hessian_score(probes,n,diagonal):
    score,_=probes;h=laplacian(n)
    if diagonal:h+=np.eye(n,dtype=np.float32)*.4
    g=np.random.default_rng(782+n).normal(size=n).astype(np.float32)
    if not diagonal:g-=g.mean(dtype=float)
    work,direction,value,status=score(h,g,diagonal=diagonal)
    expected=stabilized(h,diagonal,3,.2);size=n if diagonal else n-1
    beta=np.zeros(n)
    if size:beta[:size]=np.linalg.solve(expected[:size,:size],g[:size])
    if not diagonal:beta-=beta.mean()
    assert status==0
    np.testing.assert_allclose(work,expected,rtol=2e-13,atol=2e-12)
    np.testing.assert_allclose(direction,beta,rtol=4e-6,atol=3e-7)
    actual_beta=direction.astype(float)
    expected_score=actual_beta@g.astype(float)-.5*actual_beta@h.astype(float)@actual_beta
    assert value==pytest.approx(expected_score,rel=3e-12,abs=3e-11)


@pytest.mark.parametrize('diagonal',[False,True])
@pytest.mark.parametrize('mass',[0,1e-10,1e-8,1e-7,1e-6,1])
def test_empty_and_low_diagonal_stabilizer_is_split_specific(probes,diagonal,mass):
    score,_=probes;h=np.diag(np.array([mass,0,2,3],np.float32));g=np.array([1,-1,2,-2],np.float32)
    work,_,_,status=score(h,g,diagonal=diagonal,l2=.25,non_diag=.5)
    assert status==0
    np.testing.assert_allclose(work,stabilized(h,diagonal,.25,.5),rtol=2e-13,atol=2e-12)
    assert work[1,1]<10  # Leaf estimation has a different empty-diagonal rule.


def test_ridge_is_excluded_from_scored_quadratic(probes):
    score,_=probes;h=np.array([[2,-2],[-2,2]],np.float32);g=np.array([3,-3],np.float32)
    _,beta,value,status=score(h,g,l2=100,non_diag=20)
    assert status==0
    expected=beta.astype(float)@g-.5*beta.astype(float)@h@beta.astype(float)
    assert value==pytest.approx(expected,rel=2e-12)
    assert abs(value-.5*beta@g)>0.03


@pytest.mark.parametrize('n',[1,2,257,1025])
def test_first_candidate_ties_and_persistent_score_sign(probes,n):
    _,select=probes
    values=np.full(n,5,np.float32);features=np.zeros(n,np.uint32);weights=[2]
    winner,result=select(values,features,weights,previous=-3)
    assert winner==0
    np.testing.assert_array_equal(result,[-5,4])


def test_feature_penalty_applies_to_gain_not_raw_score(probes):
    _,select=probes
    winner,result=select([10,8,7],[0,1,2],[.1,1,3],previous=-6)
    assert winner==2
    np.testing.assert_array_equal(result,[-7,3])


def test_invalid_candidates_are_skipped_and_empty_selection_is_explicit(probes):
    _,select=probes
    assert select([np.nan,np.inf,2,100],[0,0,0,7],[1])[0]==2
    assert select([1],[0],[-1])[0]==0xffffffff
    assert select([],[],[1])[0]==0xffffffff
