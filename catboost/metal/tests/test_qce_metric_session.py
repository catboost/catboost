"""Persistent metric-only GPU buffers: numerical identity, ownership and recovery."""
import ctypes as ct
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
from catboost_metal import _query_cross_entropy as qce


def data(sizes=(1,7,11)):
    n=sum(sizes);rng=np.random.default_rng(n+3421);y=rng.uniform(.1,.9,n).astype(np.float32)
    w=rng.uniform(.2,2,n).astype(np.float32);w[::7]=0;point=rng.normal(size=n).astype(np.float32)
    scales=rng.uniform(-1,2,len(sizes)).astype(np.float32);offsets=np.r_[0,np.cumsum(sizes)].astype(np.uint32)
    return y,w,offsets,scales,point


@pytest.mark.parametrize('sizes',[(1,7,11),(16,)*4,(32,33),(64,65),(128,129),(256,)*4])
@pytest.mark.parametrize('alpha',[0,.37,.95,1])
def test_metric_session_reuses_buffers_and_matches_one_shot_bits(sizes,alpha):
    y,w,offsets,scales,point=data(sizes)
    with qce.MetricSession(y,w,offsets,scales) as session:
        allocated=session.allocated_bytes
        for iteration in range(4):
            x=np.float32(point+.25*iteration)
            expected=qce.metric(x,y,w,offsets,alpha=alpha,query_scales=scales)
            assert session.evaluate(x,alpha)==expected
            assert session.evaluations==iteration+1 and session.allocated_bytes==allocated


def test_metadata_is_copied_once_and_future_alpha_changes_remain_correct():
    y,w,offsets,scales,point=data();saved=[a.copy() for a in (y,w,offsets,scales)]
    with qce.MetricSession(y,w,offsets,scales) as session:
        y[:]=np.nan;w[:]=-1;offsets[:]=0;scales[:]=np.inf
        for alpha in (.1,.9,0,1):
            assert session.evaluate(point,alpha)==qce.metric(point,saved[0],saved[1],saved[2],alpha=alpha,query_scales=saved[3])


@pytest.mark.parametrize('change',['nan','infinity','shape','alpha_low','alpha_high','alpha_nan'])
def test_invalid_evaluation_preserves_workspace_for_next_call(change):
    y,w,offsets,scales,point=data()
    with qce.MetricSession(y,w,offsets,scales) as session:
        original=session.evaluate(point);x=point.copy();alpha=.95
        if change=='nan':x[0]=np.nan
        if change=='infinity':x[0]=np.inf
        if change=='shape':x=x[:-1]
        if change=='alpha_low':alpha=-.1
        if change=='alpha_high':alpha=1.1
        if change=='alpha_nan':alpha=np.nan
        with pytest.raises(ValueError):session.evaluate(x,alpha)
        assert session.evaluations==1 and session.evaluate(point)==original and session.evaluations==2


def test_gpu_overflow_status_is_cleared_before_next_evaluation():
    y=np.array([0,1],np.float32);w=np.ones(2,np.float32);scales=np.array([1e10],np.float32)
    with qce.MetricSession(y,w,[0,2],scales) as session:
        with pytest.raises(ValueError):session.evaluate([np.finfo(np.float32).max]*2)
        assert session.evaluations==0
        assert session.evaluate([0,0])==qce.metric([0,0],y,w,[0,2],query_scales=scales)
        assert session.evaluations==1


def test_exact_metric_budget_and_failed_creation_recover():
    y,w,offsets,scales,point=data()
    with qce.MetricSession(y,w,offsets,scales) as initial:exact=initial.allocated_bytes
    with pytest.raises(ValueError,match='budget'):qce.MetricSession(y,w,offsets,scales,budget=exact-1)
    with qce.MetricSession(y,w,offsets,scales,budget=exact) as session:
        assert session.allocated_bytes==exact and np.isfinite(session.evaluate(point))


@pytest.mark.parametrize('budget',[0,-1,True,1.5,2**30+1])
def test_invalid_metric_budget_is_rejected(budget):
    with pytest.raises(ValueError,match='budget'):qce.MetricSession([0,1],None,[0,2],budget=budget)


def test_close_is_idempotent_and_old_handles_never_reopen():
    y,w,offsets,scales,point=data();session=qce.MetricSession(y,w,offsets,scales);handle=ct.c_void_p(session._handle.value)
    session.close();session.close()
    with pytest.raises(ValueError,match='closed'):session.evaluate(point)
    with qce.MetricSession(y,w,offsets,scales) as fresh:
        assert handle.value!=fresh._handle.value
        error=ct.create_string_buffer(2048);result=ct.c_double();count=ct.c_uint64()
        code=fresh._lib.cbm_query_cross_entropy_metric_evaluate(handle,len(y),qce._native._f32(point),.95,
            ct.byref(result),ct.byref(count),error,len(error))
        assert code and b'closed or invalid' in error.value and fresh.evaluations==0
        assert np.isfinite(fresh.evaluate(point))


def test_serialized_shared_handle_and_independent_parallel_workspaces():
    y,w,offsets,scales,point=data((64,)*4);alphas=[0,.2,.7,1]*3
    expected=[qce.metric(point,y,w,offsets,alpha=a,query_scales=scales) for a in alphas]
    with qce.MetricSession(y,w,offsets,scales) as session:
        with ThreadPoolExecutor(4) as threads:actual=list(threads.map(lambda a:session.evaluate(point,a),alphas))
        assert actual==expected and session.evaluations==len(alphas)
    def independent(alpha):
        with qce.MetricSession(y,w,offsets,scales) as session:return session.evaluate(point,alpha)
    with ThreadPoolExecutor(4) as threads:assert list(threads.map(independent,alphas))==expected
