"""Bounded scale parsing and actual GPU target metrics without training."""
import numpy as np
import pytest
from catboost_metal import _query_cross_entropy as qce
from test_query_cross_entropy_kernels import reference


@pytest.mark.parametrize('description,expected',[
    ('',[1,1,1]),('0,0:.4',[.4,.4,.4]),('0,0:.4 0,0:2',[.4,.4,.4]),
    ('2,1:1.7 0,0:.4',[.4,1.7,.4]),('2,1:1.7 2,1:.9',[1,.9,1]),
    ('4294967295,0:4 0,0:.3',[.3,.3,.3]),('3,1:2 3,2:.8',[1,1,.8]),
    ('1,0:0 2,1:-.3',[0,-.3,1]),
])
def test_bounded_scale_lookup_keeps_cuda_defaults_and_duplicate_semantics(description,expected):
    result=qce.select_scales(description,[0,0,1,0,1,1],[0,1,3,6])
    np.testing.assert_array_equal(result,np.array(expected,np.float32))


@pytest.mark.parametrize('description',[' ',':','2,3:1','-1,0:1','2.0,0:1','4294967296,0:1','0,0:nan','0,0:inf','0,0:1e100','0,0:1  2,1:1'])
def test_scale_description_rejects_invalid_or_nonfinite_values(description):
    with pytest.raises(ValueError):qce.select_scales(description,[0],[0,1])


@pytest.mark.parametrize('alpha',[0,.37,.95,1])
@pytest.mark.parametrize('scales',[[1,1,1],[.8,1.5,.3],[-.5,0,1.3]])
def test_gpu_metric_matches_independent_constrained_query_optima(alpha,scales):
    offsets=np.array([0,1,8,19],np.uint32);rng=np.random.default_rng(381)
    y=rng.uniform(.1,.9,19);w=rng.uniform(.2,2,19);x=rng.normal(0,1,19);w[::7]=0
    expected=reference(y,w,x,offsets,np.zeros(19,np.uint32),alpha=alpha,scales=scales)[1][:,2].sum()/w.astype(np.float32).sum(dtype=float)
    actual=qce.metric(x,y,w,offsets,alpha=alpha,query_scales=scales)
    assert actual==pytest.approx(expected,rel=3e-6,abs=2e-7)


def test_metric_uses_cuda_single_class_threshold_and_scale_instead_of_shared_cpu_metric():
    y=np.array([.3,.300005,0,1]);w=np.array([1,2,3,4]);x=np.array([-1,2,.3,-.8]);offsets=[0,2,4]
    expected=reference(y,w,x,offsets,np.zeros(4,np.uint32),alpha=.8,scales=[2,1.4])[1][:,2].sum()/w.sum()
    actual=qce.metric(x,y,w,offsets,alpha=.8,query_scales=[2,1.4])
    assert actual==pytest.approx(expected,rel=3e-6)
    from catboost.utils import eval_metric
    host=eval_metric(y,x,'QueryCrossEntropy:alpha=.8',weight=w,group_id=[0,0,1,1])[0]
    assert abs(actual-host)>.1


def test_invalid_metric_state_does_not_poison_next_gpu_call():
    with pytest.raises(ValueError):qce.metric([2,2],[0,1],[1,1],[0,2],query_scales=[np.finfo(np.float32).max])
    assert np.isfinite(qce.metric([0,0],[0,1],[1,1],[0,2]))
