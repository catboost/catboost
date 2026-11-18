"""CUDA query equations projected through Metal's variable-node tree policies.

Whole-query normalization, every estimated leaf, final cursor and metrics are
checked independently; no CPU CatBoost fitting is permitted.
"""
import ctypes as ct
import platform
import numpy as np
import pytest
from catboost import CatBoost
from catboost_metal import _greedy
from catboost_metal._native import QueryOptions, ObjectiveOptions, _u32, _u8, _f32
from cuda_querywise_reference import query_terms, query_loss, leaf_reference
from test_querywise_training import problem
from test_greedy_training import route, POLICIES

pytestmark = pytest.mark.skipif(platform.system() != 'Darwin' or platform.machine() != 'arm64', reason='Apple GPU required')
OBJECTIVES=('QueryRMSE','QuerySoftMax')

@pytest.fixture(autouse=True)
def no_cpu(monkeypatch):
    def forbidden(*a,**kw): raise AssertionError('CPU CatBoost fitting is forbidden')
    monkeypatch.setattr(CatBoost,'_fit',forbidden)


def options(objective,policy,**extra):
    return dict(objective=objective,grow_policy=policy,iterations=3,depth=3,max_leaves=6,
        learning_rate=.18,l2_leaf_reg=2.3,leaf_estimation_iterations=3,
        score_function='Cosine',query_beta=.7,query_lambda=.03)|extra


def root(bins,y,w,raw,offsets,cf,cb,types,objective,score,l2,beta,lam):
    g,h,*_=query_terms(y,raw,w,offsets,objective,beta,lam)
    g=g.astype(np.float32).astype(float)
    weight=h.astype(np.float32).astype(float) if score.startswith('Newton') else w.astype(float)
    def value(sums,masses):
        if score=='SolarL2': return sum(-s*s*(1+2*np.log1p(m))/m for s,m in zip(sums,masses) if m>1e-20)
        if score=='LOOL2': return sum(-s*s/m*(m/(m-1))**2 for s,m in zip(sums,masses) if m>1)
        if score.endswith('L2'): return sum(-s*s/(m+l2) for s,m in zip(sums,masses) if m>1e-20)
        d=[s/(m+l2) if m>0 else 0 for s,m in zip(sums,masses)]
        return -np.dot(sums,d)/np.sqrt(1e-10+np.dot(masses,np.square(d)))
    parent=value([g.sum()],[weight.sum()]);gains=[]
    for f,b,t in zip(cf,cb,types):
        right=bins[f]==b if t else bins[f]>b
        masses=[weight[~right].sum(),weight[right].sum()]
        gains.append(0 if min(masses)<1e-20 else value([g[~right].sum(),g[right].sum()],masses)-parent)
    return int(np.argmin(gains)),min(gains)


@pytest.mark.parametrize('objective',OBJECTIVES)
@pytest.mark.parametrize('policy',POLICIES)
@pytest.mark.parametrize('method',['Newton','Gradient'])
@pytest.mark.parametrize('score',['L2','Cosine','NewtonL2','NewtonCosine','SolarL2','LOOL2'])
@pytest.mark.parametrize('onehot',[False,True])
def test_weighted_queries_all_policies_score_and_leaf_equations(objective,policy,method,score,onehot):
    bins,y,w,offsets,raw,cf,cb=problem(objective)
    types=np.zeros(len(cf),np.uint8)
    if onehot: types[cf==2]=1
    params=options(objective,policy,score_function=score,leaf_estimation_method=method)
    result=_greedy.train(bins,y,cf,cb,sample_weight=w,group_offsets=offsets,
        initial_predictions=raw,candidate_types=types,**params)
    winner,gain=root(bins,y,w,raw,offsets,cf,cb,types,objective,score,2.3,.7,.03)
    if policy!='Depthwise' or gain<0:
        assert result.trees[0].nodes[0,:3].tolist()==[int(cf[winner]),int(cb[winner]),int(types[winner])]
    assert result.loss[0]==pytest.approx(query_loss(y,raw,w,offsets,objective,.7,.03),rel=4e-6,abs=2e-7)
    for tree in result.trees:
        ids=route(tree,bins)
        point,masses,_=leaf_reference(y,raw,w,offsets,ids,len(tree.leaf_values),objective=objective,
            l2_leaf_reg=2.3,leaf_estimation_method=method,leaf_estimation_iterations=3,query_beta=.7,query_lambda=.03)
        np.testing.assert_allclose(tree.leaf_values,point*np.float32(.18),atol=2e-6,rtol=5e-5)
        np.testing.assert_allclose(tree.leaf_weights,masses,rtol=3e-6,atol=2e-6)
        raw=raw+tree.leaf_values[ids]
        assert tree.loss==pytest.approx(query_loss(y,raw,w,offsets,objective,.7,.03),rel=5e-6,abs=2e-7)
    np.testing.assert_array_equal(result.predictions,raw)


@pytest.mark.parametrize('objective',OBJECTIVES)
@pytest.mark.parametrize('policy',POLICIES)
@pytest.mark.parametrize('method',['Newton','Gradient'])
@pytest.mark.parametrize('mode',['AnyImprovement','Armijo'])
@pytest.mark.parametrize('steps',[3,7])
def test_query_backtracking_matches_whole_query_walker(objective,policy,method,mode,steps):
    bins,y,w,offsets,raw,cf,cb=problem(objective)
    result=_greedy.train(bins,y,cf,cb,sample_weight=w,group_offsets=offsets,initial_predictions=raw,
        **options(objective,policy,leaf_estimation_method=method,leaf_estimation_iterations=steps,
            leaf_estimation_backtracking=mode,query_beta=3.,query_lambda=.001,l2_leaf_reg=.4))
    for tree in result.trees:
        ids=route(tree,bins)
        point,masses,trace=leaf_reference(y,raw,w,offsets,ids,len(tree.leaf_values),objective=objective,
            l2_leaf_reg=.4,leaf_estimation_method=method,leaf_estimation_iterations=steps,
            leaf_estimation_backtracking=mode,query_beta=3.,query_lambda=.001)
        # At convergence the GPU float32 query loss can change Armijo's final
        # accept/reject decision against the independent double loss. Three
        # steps remain a strict equation check; seven bound the resulting
        # leaf and metric differences rather than assuming identical branches.
        expected=point*np.float32(.18)
        np.testing.assert_allclose(tree.leaf_values,expected,atol=4e-6 if steps==3 else 8e-5,rtol=1e-4)
        actual_loss=query_loss(y,raw+tree.leaf_values[ids],w,offsets,objective,3.,.001)
        reference_loss=query_loss(y,raw+expected[ids],w,offsets,objective,3.,.001)
        assert abs(actual_loss-reference_loss)<(1e-6 if steps==3 else 2e-5)
        np.testing.assert_allclose(tree.leaf_weights,masses,rtol=3e-6,atol=2e-6)
        raw+=tree.leaf_values[ids]
    np.testing.assert_array_equal(result.predictions,raw)


@pytest.mark.parametrize('objective',OBJECTIVES)
@pytest.mark.parametrize('policy',POLICIES)
@pytest.mark.parametrize('count',[1,4,7])
def test_every_ctr_bank_estimates_query_point_on_its_own_partition(objective,policy,count):
    bins,y,w,offsets,raw,cf,cb=problem(objective)
    banks=np.stack([bins.copy() for _ in range(count)])
    for p in range(1,count): banks[p,0]=np.roll(bins[0],3*p)
    cursors=np.stack([raw.copy() for _ in range(count)])
    with _greedy.TrainingSession(bins,y,cf,cb,sample_weight=w,group_offsets=offsets,initial_predictions=raw,
        **options(objective,policy)) as session:
        session.configure_permutations(banks,cursors)
        for i in range(3):
            session.select_permutation(i%count);tree=session.step()
            for p in range(count):
                ids=route(tree,banks[p])
                point,masses,_=leaf_reference(y,cursors[p],w,offsets,ids,len(tree.leaf_values),
                    objective=objective,l2_leaf_reg=2.3,leaf_estimation_iterations=3,query_beta=.7,query_lambda=.03)
                cursors[p]+=(point*np.float32(.18))[ids]
                if p==count-1: np.testing.assert_allclose(tree.leaf_values,point*np.float32(.18),atol=3e-6,rtol=5e-5)
            np.testing.assert_allclose(session.permutation_state['predictions'],cursors,atol=4e-6,rtol=8e-5)
            np.testing.assert_array_equal(session.predictions(),session.permutation_state['predictions'][-1])


@pytest.mark.parametrize('policy',POLICIES)
@pytest.mark.parametrize('beta,lam,method',[(0,.01,'Newton'),(-.7,.03,'Gradient'),(1,-1,'Newton'),(1,1e30,'Gradient')])
def test_literal_softmax_parameters_and_regularized_signed_curvature(policy,beta,lam,method):
    bins=np.array([[0,1]],np.uint8);y=np.array([0,1],np.float32);c=np.array([0],np.uint32);offsets=np.array([0,2],np.uint32)
    result=_greedy.train(bins,y,c,c,group_offsets=offsets,**options('QuerySoftMax',policy,
        iterations=1,depth=1,max_leaves=2,query_beta=beta,query_lambda=lam,l2_leaf_reg=1,
        leaf_estimation_iterations=1,leaf_estimation_method=method,learning_rate=1))
    ids=route(result.trees[0],bins)
    point,_,_=leaf_reference(y,np.zeros(2),np.ones(2),offsets,ids,len(result.trees[0].leaf_values),
        objective='QuerySoftMax',l2_leaf_reg=1,query_beta=beta,query_lambda=lam,leaf_estimation_method=method)
    np.testing.assert_allclose(result.trees[0].leaf_values,point,atol=2e-6)


@pytest.mark.parametrize('policy',POLICIES)
@pytest.mark.parametrize('score',['NewtonL2','NewtonCosine'])
def test_invalid_query_structure_curvature_is_rejected(policy,score):
    bins,y,w,offsets,raw,cf,cb=problem('QuerySoftMax')
    with pytest.raises(RuntimeError,match='(?i)curvature'):
        _greedy.train(bins,y,cf,cb,sample_weight=w,group_offsets=offsets,
            **options('QuerySoftMax',policy,score_function=score,query_lambda=-5))


@pytest.mark.parametrize('changes',[{'group_offsets':None},{'group_offsets':[0,2,2,521]},
    {'query_beta':float('nan')},{'query_lambda':float('inf')},{'leaf_estimation_method':'Exact'}])
def test_bad_query_arguments_fail_before_loading_gpu(monkeypatch,changes):
    bins,y,w,offsets,raw,cf,cb=problem('QuerySoftMax')
    def forbidden(): pytest.fail('invalid arguments reached Metal')
    monkeypatch.setattr(_greedy,'build_library',forbidden)
    with pytest.raises(ValueError):
        _greedy.TrainingSession(bins,y,cf,cb,**(options('QuerySoftMax','Lossguide')|dict(group_offsets=offsets)|changes))


def test_counted_query_cabi_rejects_bad_length_before_reading_pointer():
    lib=_greedy._load(_greedy.build_library());error=ct.create_string_buffer(2048);handle=ct.c_void_p()
    p=_greedy.Params();o=ObjectiveOptions(12,0,1,0);q=QueryOptions(2,1,.01,0)
    # Non-null sentinel must never be read when its counted geometry is wrong.
    offset=ct.cast(ct.c_void_p(1),ct.POINTER(ct.c_uint32))
    code=lib.cbm_greedy_session_create_query(ct.byref(p),ct.byref(o),ct.byref(q),offset,2,
        None,None,None,None,None,None,None,ct.byref(handle),error,len(error))
    assert code and not handle.value and b'count' in error.value


@pytest.mark.parametrize('policy',POLICIES)
@pytest.mark.parametrize('mode',['AnyImprovement','Armijo'])
def test_query_softmax_rejects_overshooting_gradient_trial(policy,mode):
    bins=np.array([[0,1]],np.uint8);y=np.array([.5,.5],np.float32);c=np.array([0],np.uint32)
    off=np.array([0,2],np.uint32);raw=np.array([-2,2],np.float32)
    result=_greedy.train(bins,y,c,c,group_offsets=off,initial_predictions=raw,
        **options('QuerySoftMax',policy,iterations=1,depth=1,max_leaves=2,query_beta=10,
            l2_leaf_reg=.01,learning_rate=1,leaf_estimation_method='Gradient',
            leaf_estimation_iterations=2,leaf_estimation_backtracking=mode))
    tree=result.trees[0];ids=route(tree,bins)
    point,_,trace=leaf_reference(y,raw,np.ones(2),off,ids,len(tree.leaf_values),objective='QuerySoftMax',
        l2_leaf_reg=.01,query_beta=10,query_lambda=.03,leaf_estimation_method='Gradient',
        leaf_estimation_iterations=2,leaf_estimation_backtracking=mode)
    assert trace[0]==(1.,False) and any(accepted for _,accepted in trace)
    np.testing.assert_allclose(tree.leaf_values,point,atol=2e-6,rtol=2e-6)
