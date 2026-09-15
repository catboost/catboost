"""Sampled PFound weak matrices and independently regenerated fixed leaf pairs."""
import ctypes as ct
import hashlib,platform,subprocess
from pathlib import Path
import numpy as np
import pytest
from test_bootstrap import seed_for_item,next_word,uniforms,mix_seed
from test_pfound_pair_kernels import reference as pair_reference
from test_pairwise_matrix_kernels import reference as leaf_reference
from test_pairwise_score_kernels import stabilized

class Params(ct.Structure):
    _fields_=[(n,ct.c_uint32) for n in ('rows','groups','features','candidates','parents','leaves','permutations','iterations',
        'leaf_method','score_method','bootstrap','seed_low','seed_high','absolute','group_sampling','passes')]
    _fields_ += [(n,ct.c_float) for n in ('l2','non_diag','temperature','subsample')]
    _fields_ += [('compact',ct.c_uint32)]

@pytest.fixture(scope='module')
def runtime():
    if platform.system()!='Darwin' or platform.machine()!='arm64':pytest.skip('Apple GPU required')
    root=Path(__file__).resolve().parents[1];source=Path(__file__).with_name('pfound_pair_runtime_probe.mm')
    deps=[source,*sorted((root/'native').glob('*.h')),root/'native/metal_sort.mm']
    digest=hashlib.sha256(b''.join(p.read_bytes() for p in deps)).hexdigest()[:20]
    output=root/'.build'/('pfound_runtime_'+digest+'.dylib')
    if not output.exists():
        result=subprocess.run(['xcrun','clang++','-std=c++17','-O2','-fobjc-arc','-dynamiclib','-framework','Foundation',
            '-framework','Metal',str(source),str(root/'native/metal_sort.mm'),'-o',str(output)],capture_output=True,text=True)
        assert result.returncode==0,result.stdout+result.stderr
    lib=ct.CDLL(str(output));lib.cbm_pfound_runtime_probe.argtypes=[ct.POINTER(Params),ct.c_uint64]+[ct.c_void_p]*23+[ct.c_uint32]
    lib.cbm_pfound_runtime_probe.restype=ct.c_int
    def run(problem,*,bootstrap=0,seed=738,absolute=0,subsample=.5,temperature=.7,group_sampling=False,
            method='Newton',score='Newton',passes=2,iterations=3,permutations=7,budget=2**30,compact=True):
        bins,y,w,x,off=problem;n=len(y);groups=len(off)-1;leaves=8;count=9
        parent=np.arange(n,dtype=np.uint32)%4;ids=np.arange(n,dtype=np.uint32)%leaves
        rows=np.argsort(ids,kind='stable').astype(np.uint32);loff=np.searchsorted(ids[rows],np.arange(leaves+1)).astype(np.uint32)
        features=np.arange(count,dtype=np.uint32)%3;borders=np.arange(count,dtype=np.uint32)//3;types=(np.arange(count)%2).astype(np.uint8)
        arrays=[np.ascontiguousarray(a,d) for a,d in zip((bins,parent,ids,x,y,w,off,rows,loff,features,borders,types),
            (np.uint8,np.uint32,np.uint32,np.float32,np.float32,np.float32,np.uint32,np.uint32,np.uint32,np.uint32,np.uint32,np.uint8))]
        cap=min(2*n//groups+8,1023);pc=max(1,sum(min(int(s),cap)*(min(int(s),cap)-1)//2 for s in np.diff(off)))
        shapes=np.zeros((passes,2,4),np.uint32);docs=np.zeros((passes,2,n),np.uint32);pairs=np.zeros((passes,2,pc,2),np.uint32)
        edges=np.zeros((passes,2,pc,4),np.float32);matrix=np.zeros((passes,2,pc),np.float32)
        scores=np.zeros((passes,count,2),np.float32);values=np.zeros((passes,leaves),np.float32);weights=np.zeros_like(values)
        allocated=ct.c_uint64();seconds=ct.c_double();error=ct.create_string_buffer(4096)
        p=Params(n,groups,len(bins),count,4,leaves,permutations,iterations,method=='Gradient',score=='Gradient',bootstrap,seed&0xffffffff,seed>>32,
            absolute,group_sampling,passes,3.,.2,temperature,subsample,compact)
        code=lib.cbm_pfound_runtime_probe(ct.byref(p),budget,*(a.ctypes.data for a in (*arrays,shapes,docs,pairs,edges,matrix,scores,values,weights)),
            ct.byref(allocated),ct.byref(seconds),error,len(error))
        if code:raise ValueError(error.value.decode())
        return dict(shapes=shapes,docs=docs,pairs=pairs,edges=edges,matrix=matrix,scores=scores.sum(axis=-1,dtype=float),values=values,weights=weights,
            allocated=allocated.value,seconds=seconds.value,parent=parent,ids=ids,features=features,borders=borders,types=types)
    return run


def problem(sizes=(7,13,19,31),constant=False):
    n=sum(sizes);rng=np.random.default_rng(n+739)
    y=rng.uniform(0,1,n).astype(np.float32)
    if constant:y.fill(.5)
    w=rng.uniform(.1,3,n).astype(np.float32);w[::19]=0;x=rng.normal(0,1,n).astype(np.float32)
    bins=rng.integers(0,4,(3,n),dtype=np.uint8)
    return bins,y,w,x,np.r_[0,np.cumsum(sizes)].astype(np.uint32)


def target_reference(data,*,bootstrap=0,seed=738,absolute=0,subsample=.5,temperature=.7,group_sampling=False,permutations=7,fixed=False,dataset=0):
    _,y,w,x,offsets=data;n=len(y);groups=len(offsets)-1;f=np.float32
    domain=(0x50464c00 if fixed else 0x50465700)+4*dataset
    keys=np.array([next_word(seed_for_item(i,seed=seed,iteration=absolute,stream=domain))[1] for i in range(n)],np.uint32)
    mask=uniforms(groups,seed=seed,iteration=absolute,stream=domain+1)<f(subsample) if not fixed and group_sampling and bootstrap==2 else np.ones(groups,bool)
    fraction=f(subsample) if not fixed and not group_sampling and bootstrap==2 else f(1)
    doc_parts=[];sizes=[];cap=min(2*n//groups+8,1023)
    for query,(begin,end) in enumerate(zip(offsets[:-1],offsets[1:])):
        if not mask[query]:continue
        take=min(cap,max(min(2,int(end-begin)),int(np.ceil(f(fraction*f(end-begin))))))
        docs=np.arange(begin,end);selected=docs[np.argsort(keys[docs],kind='stable')[:take]]
        doc_parts.append(np.sort(selected));sizes.append(len(selected))
    if not sizes:return dict(docs=np.empty(0,np.uint32),groups=0,active=0,matrix=np.empty(0,np.float32),pairs=np.empty((0,2),np.uint32),edges=np.empty((0,4),np.float32))
    docs=np.concatenate(doc_parts).astype(np.uint32);off=np.r_[0,np.cumsum(sizes)]
    dataset_seed=mix_seed((dataset<<32)|0x50464453) if dataset else 0
    oracle_seed=mix_seed(seed^mix_seed(absolute)^dataset_seed^(0x50464c454146 if fixed else 0x50465745414b))
    raw=pair_reference(y[docs],w[docs],x[docs],off,seed=oracle_seed,permutations=permutations,document_ids=docs)
    a,b=raw['pairs'].T;sampled_a,sampled_b=np.searchsorted(docs,a),np.searchsorted(docs,b)
    matrix=raw['matrix'];mass=matrix.copy()
    if fixed or bootstrap==1:
        u=uniforms(len(mass),seed=seed,iteration=absolute,stream=domain+2)
        mass=f(mass*np.power(-np.log(f(u+f(1e-20))),f(1 if fixed else temperature),dtype=np.float32))
    active=int(np.count_nonzero(abs(mass)>f(1e-20)))
    mass=np.where(abs(mass)>f(1e-20),f(mass*w[a]),f(0)).astype(np.float32)
    ax,ay=f(raw['exponent'][sampled_a]+f(1e-20)),f(raw['exponent'][sampled_b]+f(1e-20))
    gradient=f(f(mass*np.where(y[a]>y[b],ay,-ax))/f(ax+ay))
    return dict(docs=docs,groups=len(sizes),active=active,matrix=matrix,pairs=raw['pairs'],edges=np.column_stack((gradient,mass,mass,np.zeros(len(mass),np.float32))))


def score_reference(data,actual,weak):
    bins=data[0];parent=actual['parent'];pair=weak['pairs'];edges=weak['edges'];result=[]
    for feature,border,kind in zip(actual['features'],actual['borders'],actual['types']):
        child=2*parent+(bins[feature]!=border if kind else bins[feature]>border).astype(np.uint32)
        a,b=child[pair[:,0]],child[pair[:,1]];keep=a!=b;a,b=a[keep],b[keep];g=edges[keep,0];w=edges[keep,2]
        gradient=np.zeros(8);h=np.zeros((8,8));np.add.at(gradient,a,g);np.add.at(gradient,b,-g)
        np.add.at(h,(a,a),w);np.add.at(h,(b,b),w);np.add.at(h,(a,b),-w);np.add.at(h,(b,a),-w)
        gradient=gradient.astype(np.float32);h=h.astype(np.float32);m=stabilized(h,False,3.,.2)
        values=np.r_[np.linalg.solve(m[:-1,:-1],gradient[:-1]),0.];values-=values.mean()
        result.append(values@gradient-.5*values@h@values)
    return result


@pytest.mark.parametrize('bootstrap,group',[(0,False),(1,False),(2,False),(2,True)])
@pytest.mark.parametrize('method',['Newton','Gradient'])
@pytest.mark.parametrize('score',['Newton','Gradient'])
def test_sampled_weak_target_and_separate_bayesian_leaf_target(runtime,bootstrap,group,method,score):
    data=problem();kwargs=dict(bootstrap=bootstrap,group_sampling=group,seed=0xf0123abcdef,absolute=17)
    actual=runtime(data,method=method,score=score,**kwargs)
    for step in range(2):
        expected=[]
        for fixed in (False,True):
            ref=target_reference(data,**{**kwargs,'absolute':17+step},fixed=fixed);expected.append(ref)
            shape=actual['shapes'][step,int(fixed)];count=len(ref['matrix'])
            np.testing.assert_array_equal(shape,[len(ref['docs']),ref['groups'],count,ref['active']])
            np.testing.assert_array_equal(actual['docs'][step,int(fixed),:shape[0]],ref['docs'])
            for key in ('matrix','pairs','edges'):
                np.testing.assert_allclose(actual[key][step,int(fixed),:count],ref[key],rtol=5e-5,atol=3e-7,err_msg=key)
        np.testing.assert_allclose(actual['scores'][step],score_reference(data,actual,expected[0]),rtol=5e-5,atol=2e-6)
        fixed=expected[1];a,b=fixed['pairs'].T;y=data[1];swap=y[a]<y[b];win=np.where(swap,b,a);lose=np.where(swap,a,b)
        values=np.zeros(8,np.float32)
        for _ in range(3):
            ref=leaf_reference(np.float32(data[3]+values[actual['ids']]),win,lose,fixed['edges'][:,2],actual['ids'],8,method)
            values=np.float32(values+ref['direction']);values[-1]=0
        values=np.float32(values-values.mean(dtype=np.float64))
        np.testing.assert_allclose(actual['values'][step],values,rtol=5e-5,atol=3e-6)
        np.testing.assert_allclose(actual['weights'][step],np.bincount(actual['ids'],weights=data[2],minlength=8),rtol=2e-6)
    assert 0<actual['seconds']<10 and actual['allocated']<2**30


@pytest.mark.parametrize('sizes',[(1,),(1,)*513,(2,)*512,(4097,),(17,)*65])
@pytest.mark.parametrize('bootstrap,group',[(0,False),(2,False),(2,True)])
def test_empty_targets_caps_and_repeated_resident_shapes(runtime,sizes,bootstrap,group):
    data=problem(sizes,constant=True);kwargs=dict(bootstrap=bootstrap,group_sampling=group,subsample=.000001,passes=2,iterations=1,permutations=1)
    actual=runtime(data,**kwargs)
    for step in range(2):
        for fixed in (False,True):
            ref=target_reference(data,**{k:v for k,v in kwargs.items() if k not in ('passes','iterations')},absolute=step,fixed=fixed)
            shape=actual['shapes'][step,int(fixed)]
            np.testing.assert_array_equal(shape,[len(ref['docs']),ref['groups'],len(ref['matrix']),ref['active']])
            np.testing.assert_array_equal(actual['docs'][step,int(fixed),:shape[0]],ref['docs'])
        np.testing.assert_array_equal(actual['values'][step],0);np.testing.assert_array_equal(actual['scores'][step],0)


def test_budget_is_exact_and_rejected_before_large_matrix_allocation(runtime):
    data=problem();first=runtime(data,passes=1,iterations=1)
    exact=runtime(data,passes=1,iterations=1,budget=first['allocated'])
    assert exact['allocated']==first['allocated']
    with pytest.raises(ValueError,match='budget'):runtime(data,budget=1024)


@pytest.mark.parametrize('kwargs',[{'bootstrap':3},{'bootstrap':4},{'subsample':0},{'temperature':-1},{'permutations':0}])
def test_reject_unsupported_bootstrap_and_parameters(runtime,kwargs):
    with pytest.raises(ValueError):runtime(problem(),**kwargs)


@pytest.mark.parametrize('method',['Newton','Gradient'])
@pytest.mark.parametrize('bootstrap,group',[(0,False),(1,False),(2,False),(2,True)])
def test_zero_pair_compaction_preserves_scores_and_fixed_leaf_walk(runtime,method,bootstrap,group):
    data=problem((64,)*7);kwargs=dict(method=method,bootstrap=bootstrap,group_sampling=group,permutations=3,passes=2)
    dense=runtime(data,**kwargs,compact=False);compact=runtime(data,**kwargs)
    assert dense['allocated']==compact['allocated']
    np.testing.assert_array_equal(dense['shapes'][...,:3],compact['shapes'][...,:3])
    assert np.all(compact['shapes'][...,3]<dense['shapes'][...,3])
    for key in ('matrix','pairs','edges','docs','weights'):
        np.testing.assert_array_equal(compact[key],dense[key])
    for key in ('scores','values'):
        np.testing.assert_allclose(compact[key],dense[key],rtol=4e-6,atol=2e-7)


def test_raw_weight_filter_precedes_original_zero_document_weights(runtime):
    data=list(problem((64,)));data[2][::2]=0
    actual=runtime(data,passes=1,permutations=3)
    for fixed in (False,True):
        ref=target_reference(data,fixed=fixed,permutations=3)
        assert actual['shapes'][0,int(fixed),3]==ref['active']
        assert ref['active']>np.count_nonzero(ref['edges'][:,2])
