"""QueryCrossEntropy full candidate matrices and query-tile invariance on Metal."""
import ctypes as ct
import hashlib
from pathlib import Path
import platform,subprocess
import numpy as np
import pytest
from test_pairwise_score_kernels import stabilized


class Params(ct.Structure):
    _fields_=[(n,ct.c_uint32) for n in ('rows','groups','parents','candidates','features','first','group_begin','group_count')]


@pytest.fixture(scope='module')
def probe():
    if platform.system()!='Darwin' or platform.machine()!='arm64':pytest.skip('Apple GPU required')
    source=Path(__file__).with_name('qce_candidate_probe.mm');root=source.parent.parent
    dependencies=[source,*(root/'native'/name for name in ('metal_query_cross_entropy_kernels.h',
        'metal_leaf_matrix_kernels.h','metal_pairwise_score_kernels.h','metal_qce_candidate_kernels.h'))]
    digest=hashlib.sha256(b''.join(p.read_bytes() for p in dependencies)).hexdigest()[:20]
    output=root/'.build'/('qce_candidates_'+digest+'.dylib')
    if not output.exists():
        subprocess.run(['xcrun','clang++','-std=c++17','-O2','-fobjc-arc','-dynamiclib','-framework','Foundation',
            '-framework','Metal',str(source),'-o',str(output)],check=True,capture_output=True,text=True)
    lib=ct.CDLL(str(output));lib.cbm_qce_candidate_probe.argtypes=[ct.POINTER(Params),ct.c_uint32,ct.c_float,ct.c_float]+[ct.c_void_p]*15+[ct.c_uint32]
    lib.cbm_qce_candidate_probe.restype=ct.c_int
    def run(args,*,parents=4,query_tile=None,l2=.7,non_diag=.2):
        stats,groups,offsets,bins,ids,features,borders,types,active=args
        arrays=[np.ascontiguousarray(a,d) for a,d in zip(args,(np.float32,np.float32,np.uint32,np.uint8,np.uint32,np.uint32,np.uint32,np.uint8,np.uint8))]
        count=len(features);leaves=2*parents
        g=np.zeros((count,leaves),np.float32);h=np.zeros((count,leaves,leaves),np.float32)
        direction=np.zeros_like(g);score=np.zeros((count,2),np.float32);status=np.zeros(count,np.uint32)
        p=Params(len(ids),len(groups),parents,count,len(bins),0,0,0);error=ct.create_string_buffer(4096)
        code=lib.cbm_qce_candidate_probe(ct.byref(p),len(groups) if query_tile is None else query_tile,l2,non_diag,
            *(a.ctypes.data for a in (*arrays,g,h,direction,score,status)),error,len(error))
        if code:raise ValueError(error.value.decode())
        return dict(gradient=g,hessian=h,direction=direction,score=score,status=status)
    return run


def problem(parents=4,queries=17,candidates=6,seed=32631):
    rng=np.random.default_rng(seed);sizes=rng.integers(1,32,queries);sizes[0]=1;sizes[-1]=256
    offsets=np.r_[0,sizes.cumsum()].astype(np.uint32);rows=int(offsets[-1])
    # Independent finite sufficient statistics; both terms and signs matter.
    stats=np.column_stack((rng.normal(0,1,rows),rng.lognormal(-1,.7,rows),rng.lognormal(0,2,rows),rng.uniform(0,2,rows))).astype(np.float32)
    stats[::23]=0;stats[:1,2]=0
    groups=np.array([[0,stats[a:b,2].sum(dtype=float),stats[a:b,3].sum(dtype=float),b-a] for a,b in zip(offsets[:-1],offsets[1:])],np.float32)
    bins=rng.integers(0,4,(3,rows),dtype=np.uint8);ids=rng.integers(0,parents,rows,dtype=np.uint32)
    features=np.arange(candidates,dtype=np.uint32)%3;borders=np.arange(candidates,dtype=np.uint32)//3%4
    types=(features==1).astype(np.uint8);active=np.ones(queries,np.uint8)
    return stats,groups,offsets,bins,ids,features,borders,types,active


def reference(args,parents,l2=.7,non_diag=.2):
    stats,groups,offsets,bins,ids,features,borders,types,active=args
    leaves=2*parents;out=[]
    for f,b,t in zip(features,borders,types):
        child=2*ids+((bins[f]!=b) if t else (bins[f]>b))
        g=np.zeros(leaves);h=np.zeros((leaves,leaves))
        for q,(start,end) in enumerate(zip(offsets[:-1],offsets[1:])):
            if not active[q]:continue
            rows=stats[start:end].astype(float);current=child[start:end]
            for a in range(leaves):
                select=current==a;ac=rows[select,2].sum();other=rows[~select,2].sum();g[a]+=rows[select,0].sum()
                h[a,a]+=rows[select,1].sum()
                if groups[q,1]<=1e-20:continue
                h[a,a]+=ac*other/float(groups[q,1])
                for b in range(a):
                    value=-ac*rows[current==b,2].sum()/float(groups[q,1]);h[a,b]+=value;h[b,a]+=value
        g=g.astype(np.float32);h=h.astype(np.float32)
        matrix=stabilized(h,True,l2,non_diag);direction=np.linalg.solve(matrix,g).astype(np.float32)
        score=direction.astype(float)@g-.5*direction.astype(float)@h@direction
        out.append((g,h,direction,score))
    return out


@pytest.mark.parametrize('parents',[1,3,8,32,128])
@pytest.mark.parametrize('sampling',['all','alternate','none'])
def test_full_diagonal_plus_query_laplacian_candidate_scores(probe,parents,sampling):
    args=list(problem(parents))
    if sampling=='alternate':args[-1][::2]=0
    elif sampling=='none':args[-1][:]=0
    actual=probe(args,parents=parents,query_tile=5);expected=reference(args,parents)
    assert not actual['status'].any()
    for index,(g,h,d,score) in enumerate(expected):
        for name,value in [('gradient',g),('hessian',h),('direction',d)]:
            np.testing.assert_allclose(actual[name][index],value,rtol=4e-5,atol=2e-5,err_msg=name)
        assert actual['score'][index].sum(dtype=float)==pytest.approx(score,rel=5e-5,abs=2e-5)
    np.testing.assert_array_equal(actual['hessian'],actual['hessian'].transpose(0,2,1))


@pytest.mark.parametrize('tile',[1,2,3,5,8,16,17])
def test_query_tiles_are_bit_identical(probe,tile):
    args=problem();expected=probe(args);actual=probe(args,query_tile=tile)
    for name in expected:np.testing.assert_array_equal(actual[name],expected[name])


def test_single_child_queries_cancel_laplacian_without_losing_point_diagonal(probe):
    args=list(problem(parents=1));args[3][:]=0;args[4][:]=0
    result=probe(args,parents=1,query_tile=3);expected=reference(args,1)
    for index,(g,h,d,score) in enumerate(expected):
        np.testing.assert_allclose(result['hessian'][index],h,atol=1e-6,rtol=2e-6)
        assert np.count_nonzero(result['hessian'][index])==1
        np.testing.assert_allclose(result['direction'][index],d,atol=1e-6,rtol=3e-6)
        assert result['direction'][index].mean()!=0 # QCE retains the absolute mean.


@pytest.mark.parametrize('bad',['feature','border','type','parent','stats','groups'])
def test_candidate_or_gpu_statistics_errors_are_flagged(probe,bad):
    args=list(problem())
    if bad=='feature':args[5][2]=3
    elif bad=='border':args[6][2]=256
    elif bad=='type':args[7][2]=2
    elif bad=='parent':args[4][0]=4
    elif bad=='stats':args[0][0,1]=-1
    else:args[1][0,1]=np.nan
    result=probe(args)
    if bad in ('feature','border','type'):
        assert result['status'][2]!=0 and not np.delete(result['status'],2).any()
    else:assert result['status'].all()
