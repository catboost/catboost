"""Batched pairwise candidate projection, complete solves and quadratic scores."""
import ctypes as ct
import hashlib
from pathlib import Path
import platform
import subprocess
import numpy as np
import pytest
from test_pairwise_matrix_kernels import fixture, edge_reference
from test_pairwise_score_kernels import stabilized


class Params(ct.Structure):
    _fields_ = [(k, ct.c_uint32) for k in ('rows','pairs','parent_leaves','candidates','features','first_candidate','leaf_method','reserved')]


@pytest.fixture(scope='module')
def candidates():
    if platform.system() != 'Darwin' or platform.machine() != 'arm64': pytest.skip('Apple GPU required')
    source = Path(__file__).with_name('pairwise_candidate_probe.mm'); root = source.parent.parent
    dependencies = [source, *(root/'native'/n for n in ('metal_pairwise_candidate_kernels.h',
        'metal_pairwise_matrix_kernels.h','metal_pairwise_score_kernels.h','metal_leaf_matrix_kernels.h',
        'metal_sort.h','metal_sort.mm','metal_sort_kernels.h'))]
    digest = hashlib.sha256(b''.join(p.read_bytes() for p in dependencies)).hexdigest()[:20]
    output = root/'.build'/('pairwise_candidates_'+digest+'.dylib')
    if not output.exists():
        subprocess.run(['xcrun','clang++','-std=c++17','-O2','-fobjc-arc','-dynamiclib',
            '-framework','Foundation','-framework','Metal',str(source),str(root/'native/metal_sort.mm'),'-o',str(output)],
            capture_output=True,text=True,check=True)
    lib = ct.CDLL(str(output))
    lib.cbm_pairwise_candidate_probe.argtypes = [ct.POINTER(Params),ct.c_uint32,ct.c_float,ct.c_float]+[ct.c_void_p]*15+[ct.c_uint32]
    lib.cbm_pairwise_candidate_probe.restype = ct.c_int
    def run(bins, leaf_ids, point, winners, losers, weights, features, borders, types, *,
            parent_leaves, method='Newton', first=0, count=None, l2=3., non_diag=.2):
        arrays = [np.ascontiguousarray(a,dtype) for a,dtype in zip(
            (bins,leaf_ids,point,winners,losers,weights,features,borders,types),
            (np.uint8,np.uint32,np.float32,np.uint32,np.uint32,np.float32,np.uint32,np.uint32,np.uint8))]
        bins,ids,point,win,lose,w,features,borders,types = arrays
        assert bins.ndim == 2 and bins.shape[1] == len(ids) == len(point)
        assert len(win) == len(lose) == len(w) and len(features) == len(borders) == len(types)
        count = len(features)-first if count is None else count
        leaves = parent_leaves*2
        g,d = np.zeros((count,leaves),np.float32),np.zeros((count,leaves),np.float32)
        h = np.zeros((count,leaves,leaves),np.float32)
        scores,flag = np.zeros((count,2),np.float32),np.zeros(count,np.uint32)
        p = Params(len(ids),len(win),parent_leaves,count,len(bins),first,method=='Gradient',0)
        error = ct.create_string_buffer(4096)
        code = lib.cbm_pairwise_candidate_probe(ct.byref(p),len(features),l2,non_diag,
            *(a.ctypes.data for a in (*arrays,g,h,d,scores,flag)),error,len(error))
        if code: raise ValueError(error.value.decode())
        return dict(gradient=g,hessian=h,direction=d,score=scores.sum(axis=1,dtype=float),status=flag)
    return run


def problem(parents=4, pairs=2053, count=9):
    point,win,lose,weight,ids = fixture(parents,pairs=pairs,rows=131)
    rng = np.random.default_rng(517)
    bins = rng.integers(0,4,(3,len(ids)),dtype=np.uint8)
    features = np.arange(count,dtype=np.uint32)%3
    borders = np.arange(count,dtype=np.uint32)//3%3
    types = (np.arange(count)%2).astype(np.uint8)
    return bins,ids,point,win,lose,weight,features,borders,types


def reference(args,parents,method,first,count,l2=3.,non_diag=.2):
    bins,ids,point,win,lose,weights,features,borders,types = args
    edges = edge_reference(point,win,lose,weights); leaves=2*parents
    result=[]
    for candidate in range(first,first+count):
        values = bins[features[candidate]]
        predicate = values != borders[candidate] if types[candidate] else values > borders[candidate]
        children = 2*ids + predicate.astype(np.uint32)
        a,b = children[win],children[lose]; mask=a!=b
        a,b = a[mask],b[mask];g=edges[mask,0];w=edges[mask,2 if method=='Gradient' else 1]
        gradient = np.zeros(leaves);h=np.zeros((leaves,leaves))
        np.add.at(gradient,a,g);np.add.at(gradient,b,-g)
        np.add.at(h,(a,a),w);np.add.at(h,(b,b),w);np.add.at(h,(a,b),-w);np.add.at(h,(b,a),-w)
        gradient=gradient.astype(np.float32);h=h.astype(np.float32)
        matrix = stabilized(h,False,l2,non_diag)
        beta=np.r_[np.linalg.solve(matrix[:-1,:-1],gradient[:-1]),0.]
        beta-=beta.mean()
        result.append((gradient,h,beta,beta@gradient-.5*beta@h@beta))
    return result


@pytest.mark.parametrize('parents',[1,3,8,32,128])
@pytest.mark.parametrize('method',['Newton','Gradient'])
def test_batched_projection_and_solutions_match_independent_edge_matrices(candidates,parents,method):
    args=problem(parents)
    actual=candidates(*args,parent_leaves=parents,method=method)
    expected=reference(args,parents,method,0,9)
    assert not actual['status'].any()
    for index,(g,h,d,score) in enumerate(expected):
        for name,value in [('gradient',g),('hessian',h),('direction',d)]:
            np.testing.assert_allclose(actual[name][index],value,atol=5e-5,rtol=3e-5,err_msg=name)
        assert actual['score'][index]==pytest.approx(score,rel=3e-5,abs=2e-5)
    np.testing.assert_array_equal(actual['hessian'],actual['hessian'].transpose(0,2,1))


@pytest.mark.parametrize('tile',[1,2,4,7,16,64])
def test_candidate_tiling_preserves_ordered_solutions_and_scores_exactly(candidates,tile):
    args=problem(count=64); full=candidates(*args,parent_leaves=4)
    chunks=[candidates(*args,parent_leaves=4,first=i,count=min(tile,64-i)) for i in range(0,64,tile)]
    for name in full:
        np.testing.assert_array_equal(np.concatenate([part[name] for part in chunks]),full[name])


@pytest.mark.parametrize('method',['Newton','Gradient'])
def test_zero_pairs_and_repeated_high_degree_edges(candidates,method):
    args=problem(pairs=0)
    empty=candidates(*args,parent_leaves=4,method=method)
    for values in empty.values(): np.testing.assert_array_equal(values,0)
    args=problem(pairs=65539,count=3)
    actual=candidates(*args,parent_leaves=4,method=method)
    expected=reference(args,4,method,0,3)
    for index,(g,h,d,score) in enumerate(expected):
        np.testing.assert_allclose(actual['gradient'][index],g,rtol=2e-5,atol=3e-4)
        np.testing.assert_allclose(actual['hessian'][index],h,rtol=2e-5,atol=1e-3)
        np.testing.assert_allclose(actual['direction'][index],d,rtol=2e-5,atol=3e-6)
        assert actual['score'][index]==pytest.approx(score,rel=3e-5,abs=2e-5)


@pytest.mark.parametrize('field', ['feature','border','type'])
def test_invalid_candidate_is_isolated_from_other_batch_members(candidates,field):
    args=list(problem());position={'feature':6,'border':7,'type':8}[field]
    args[position][4]={'feature':3,'border':256,'type':2}[field]
    result=candidates(*args,parent_leaves=4)
    assert result['status'][4] != 0
    assert not np.delete(result['status'],4).any()


@pytest.mark.parametrize('options',[dict(parent_leaves=0),dict(parent_leaves=129),dict(count=65),dict(first=8,count=2),dict(l2=-1)])
def test_invalid_dimensions_rejected(candidates,options):
    args=problem();config=dict(parent_leaves=4);config.update(options)
    with pytest.raises(ValueError,match='candidate dimensions'):
        candidates(*args,**config)
