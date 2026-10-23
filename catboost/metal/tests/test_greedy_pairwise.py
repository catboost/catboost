"""Original supplied-edge PairLogit on Metal variable-node tree policies."""
import ctypes as ct
import hashlib
from pathlib import Path
import platform
import subprocess
import numpy as np
import pytest
from catboost import CatBoost
from catboost_metal import _greedy
from catboost_metal._native import ObjectiveOptions,PairOptions
from test_pairwise_training import problem,training_terms,leaf_reference
from test_pairwise_kernels import reference
from test_greedy_training import route,POLICIES

pytestmark=pytest.mark.skipif(platform.system()!='Darwin' or platform.machine()!='arm64',reason='Apple GPU required')


@pytest.fixture(scope='module')
def difference_probe():
 source=Path(__file__).with_name('pairwise_difference_probe.mm');root=source.parent.parent
 dependencies=[source,root/'native/metal_pairwise_runtime.h',root/'native/metal_pairwise_kernels.h']
 digest=hashlib.sha256(b''.join(path.read_bytes() for path in dependencies)).hexdigest()[:16]
 destination=root/'.build'/f'pairwise_difference_probe_{digest}.dylib'
 destination.parent.mkdir(exist_ok=True)
 if not destination.exists():
  subprocess.run(['xcrun','clang++','-std=c++17','-O2','-fobjc-arc','-dynamiclib',
   '-framework','Foundation','-framework','Metal',str(source),'-o',str(destination)],check=True,capture_output=True,text=True)
 library=ct.CDLL(str(destination));function=library.cbm_pairwise_difference_probe
 function.argtypes=[ct.c_uint32]*4+[ct.c_void_p]*9+[ct.c_uint32];function.restype=ct.c_int
 def run(cursor,winners,losers,weights,ids,current,trial,groups=7):
  buffers=[np.ascontiguousarray(value,np.uint32 if i in (1,2,4) else np.float32)
   for i,value in enumerate((cursor,winners,losers,weights,ids,current,trial))]
  result=ct.c_double();error=ct.create_string_buffer(2048)
  code=function(len(cursor),len(winners),len(current),groups,*(x.ctypes.data for x in buffers),
   ct.byref(result),error,len(error))
  if code:raise RuntimeError(error.value.decode())
  return result.value
 return run


def difference_reference(cursor,winners,losers,weights,ids,current,trial):
 # Float32 point additions are part of the training contract; loss arithmetic
 # is independently evaluated in float64, including endpoint subtraction.
 before=(np.asarray(cursor,np.float32)+np.asarray(current,np.float32)[ids]).astype(float)
 after=(np.asarray(cursor,np.float32)+np.asarray(trial,np.float32)[ids]).astype(float)
 weights=np.asarray(weights,np.float32).astype(float)
 return np.dot(weights,np.logaddexp(0,-(before[winners]-before[losers]))
  -np.logaddexp(0,-(after[winners]-after[losers])))


@pytest.mark.parametrize('seed',range(5))
@pytest.mark.parametrize('step',[1e-6,1e-3,.5,4.])
def test_pair_backtracking_signed_difference_matches_independent_loss(difference_probe,seed,step):
 rng=np.random.default_rng(138+seed);rows=79;pairs=1027;leaves=11
 cursor=rng.normal(0,2,rows).astype(np.float32);ids=rng.integers(0,leaves,rows,dtype=np.uint32)
 winners=rng.integers(0,rows,pairs,dtype=np.uint32)
 losers=(winners+rng.integers(1,rows,pairs,dtype=np.uint32))%rows
 weights=rng.lognormal(0,.7,pairs).astype(np.float32);weights[::17]=0
 current=rng.normal(0,.2,leaves).astype(np.float32)
 trial=(current+step*rng.normal(size=leaves)).astype(np.float32)
 expected=difference_reference(cursor,winners,losers,weights,ids,current,trial)
 actual=difference_probe(cursor,winners,losers,weights,ids,current,trial)
 assert actual==pytest.approx(expected,rel=8e-6,abs=1e-9)
 assert difference_probe(cursor,winners,losers,weights,ids,current,current)==0


@pytest.mark.parametrize('shift',[1e-4,-1e-4])
def test_pair_backtracking_resolves_improvements_below_absolute_loss_precision(difference_probe,shift):
 # Opposing edges nearly balance at margin 0.75. Both directions produce an
 # improvement smaller than one ULP of the absolute float32 loss. The sign
 # changes across the optimum and must drive the Armijo acceptance decision.
 cursor=np.array([.75-1e-4,0],np.float32);ids=np.arange(2,dtype=np.uint32)
 winners=np.array([0,1],np.uint32);losers=winners[::-1].copy()
 weights=np.array([1,np.exp(-.75)],np.float32);current=np.zeros(2,np.float32)
 trial=np.array([shift,0],np.float32)
 expected=difference_reference(cursor,winners,losers,weights,ids,current,trial)
 actual=difference_probe(cursor,winners,losers,weights,ids,current,trial)
 before=cursor.astype(float);after=(cursor+trial).astype(float)
 edge_changes=weights.astype(float)*(np.logaddexp(0,-(before[winners]-before[losers]))
  -np.logaddexp(0,-(after[winners]-after[losers])))
 # Opposite signed terms of order 3e-5 cancel to 1e-9. Scale the float32
 # forward-error tolerance by term magnitudes, not the nearly zero sum.
 bound=4*np.finfo(np.float32).eps*np.abs(edge_changes).sum()
 assert abs(actual-expected)<=bound
 assert np.sign(actual)==np.sign(shift)
 initial=reference(cursor,winners,losers,weights)['objective'][0]
 assert abs(expected)<np.spacing(np.float32(initial))


@pytest.mark.parametrize('cursor,shift',[
 ([1000.,-1000.],[1.,0.]),([-1000.,1000.],[1.,0.]),
 ([-10000000.,10000000.],[2.,0.]),([.25,-.25],[1e-9,1e-9]),
 ([1024.,.01],[0.,1e-4]),([.01,1024.],[1e-4,0.]),
 ([3e38,-3e38],[0.,0.]),([3e38,-3e38],[-3e38,3e38]),
 ([0.,0.],[3e38,-3e38]),
 ([-1000.,1000.],[3e38,-3e38]),
])
def test_pair_backtracking_point_rounding_and_large_margins(difference_probe,cursor,shift):
 ids=np.arange(2,dtype=np.uint32);winners=np.array([0],np.uint32);losers=np.array([1],np.uint32)
 current=np.zeros(2,np.float32);weights=np.ones(1,np.float32)
 expected=difference_reference(cursor,winners,losers,weights,ids,current,shift)
 actual=difference_probe(cursor,winners,losers,weights,ids,current,shift)
 assert actual==pytest.approx(expected,rel=3e-6,abs=1e-12)


def test_pair_backtracking_nonfinite_trials_reject_and_zero_edges_are_inert(difference_probe):
 ids=np.arange(3,dtype=np.uint32);cursor=np.zeros(3,np.float32);current=cursor.copy()
 winners=np.array([0,2],np.uint32);losers=np.array([1,1],np.uint32)
 trial=np.array([.1,0,np.inf],np.float32)
 inert=difference_probe(cursor,winners,losers,[1,0],ids,current,trial)
 assert inert>0 and np.isfinite(inert)
 assert not np.isfinite(difference_probe(cursor,winners,losers,[1,1],ids,current,trial))

@pytest.fixture(autouse=True)
def no_cpu(monkeypatch):
 def forbidden(*a,**kw):raise AssertionError('CPU CatBoost fitting is forbidden')
 monkeypatch.setattr(CatBoost,'_fit',forbidden)


def args(policy='Lossguide',**extra):
 return problem(grow_policy=policy,max_leaves=6,depth=3)|extra


def expected_root(a,types):
 terms=training_terms(a['initial_predictions'],a['pair_winners'],a['pair_losers'],a['pair_weights'])
 g=terms['gradients'].astype(np.float32).astype(float)
 score=a['score_function'];l2=a['l2_leaf_reg']
 weight=terms['curvature' if score.startswith('Newton') else 'incident_weights'].astype(np.float32).astype(float)
 def value(sums,masses):
  if score=='SolarL2':return sum(-s*s*(1+2*np.log1p(m))/m for s,m in zip(sums,masses) if m>1e-20)
  if score=='LOOL2':return sum(-s*s/m*(m/(m-1))**2 for s,m in zip(sums,masses) if m>1)
  if score.endswith('L2'):return sum(-s*s/(m+l2) for s,m in zip(sums,masses) if m>1e-20)
  d=[s/(m+l2) if m>0 else 0 for s,m in zip(sums,masses)]
  return -np.dot(sums,d)/np.sqrt(1e-10+np.dot(masses,np.square(d)))
 parent=value([g.sum()],[weight.sum()]);gains=[]
 for f,b,t in zip(a['candidate_features'],a['candidate_bins'],types):
  right=a['bins'][f]==b if t else a['bins'][f]>b;masses=[weight[~right].sum(),weight[right].sum()]
  gains.append(0 if min(masses)<1e-20 else value([g[~right].sum(),g[right].sum()],masses)-parent)
 return int(np.argmin(gains)),min(gains)


@pytest.mark.parametrize('policy',POLICIES)
@pytest.mark.parametrize('method',['Newton','Gradient'])
@pytest.mark.parametrize('score',['L2','Cosine','NewtonL2','NewtonCosine','SolarL2','LOOL2'])
@pytest.mark.parametrize('mode',['No','AnyImprovement','Armijo'])
@pytest.mark.parametrize('onehot',[False,True])
def test_pair_scores_every_tree_leaf_centering_and_metric(policy,method,score,mode,onehot):
 a=args(policy,leaf_estimation_method=method,score_function=score,leaf_estimation_backtracking=mode)
 types=np.zeros(len(a['candidate_features']),np.uint8)
 if onehot:types[a['candidate_features']==2]=1
 a['candidate_types']=types;raw=a['initial_predictions'].copy();winner,gain=expected_root(a,types)
 result=_greedy.train(**a)
 if policy!='Depthwise' or gain<0:
  assert result.trees[0].nodes[0,:3].tolist()==[int(a['candidate_features'][winner]),int(a['candidate_bins'][winner]),int(types[winner])]
 terms=reference(raw,a['pair_winners'],a['pair_losers'],a['pair_weights'])
 assert result.loss[0]==pytest.approx(terms['objective'][0]/terms['objective'][1],rel=3e-6)
 for tree in result.trees:
  ids=route(tree,a['bins']);values,masses,_=leaf_reference(a,raw,ids,len(tree.leaf_values))
  np.testing.assert_allclose(tree.leaf_values,values,atol=2e-5,rtol=4e-4)
  np.testing.assert_allclose(tree.leaf_weights,masses,atol=3e-4,rtol=3e-6)
  assert abs(tree.leaf_values.mean(dtype=float))<3e-8
  raw+=tree.leaf_values[ids]
  terms=reference(raw,a['pair_winners'],a['pair_losers'],a['pair_weights'])
  assert tree.loss==pytest.approx(terms['objective'][0]/terms['objective'][1],rel=3e-6)
 np.testing.assert_array_equal(result.predictions,raw)


@pytest.mark.parametrize('policy',POLICIES)
@pytest.mark.parametrize('count',[1,4,7])
@pytest.mark.parametrize('sampling',['No','Bayesian','Bernoulli','Poisson'])
def test_pair_dataset_cursors_are_independently_estimated_and_resume_exactly(policy,count,sampling):
 a=args(policy,iterations=4,bootstrap_type=sampling,random_strength=.3,random_seed=22,leaf_estimation_backtracking='Armijo')
 if sampling in ('Bernoulli','Poisson'):a['subsample']=.7
 banks=np.stack([a['bins'].copy() for _ in range(count)])
 for p in range(1,count):banks[p,0]=np.roll(banks[p,0],3*p)
 cursors=np.stack([a['initial_predictions'].copy() for _ in range(count)])
 with _greedy.TrainingSession(**a) as s:
  s.configure_permutations(banks,cursors)
  for i in range(4):
   s.select_permutation(i%count);tree=s.step()
   for p in range(count):
    ids=route(tree,banks[p]);values,_,_=leaf_reference(a,cursors[p],ids,len(tree.leaf_values));cursors[p]+=values[ids]
    if p==count-1:np.testing.assert_allclose(tree.leaf_values,values,atol=3e-5,rtol=5e-4)
   np.testing.assert_allclose(s.permutation_state['predictions'],cursors,atol=5e-5,rtol=8e-4)
   if i==1:state=s.permutation_state;bootstrap=s.bootstrap_state
  full=s.result()
 with _greedy.TrainingSession(**(a|dict(iterations=2,iteration_offset=2,initial_predictions=state['predictions'][-1]))) as s:
  s.configure_permutations(banks,state['predictions'])
  for i in range(2,4):
   s.select_permutation(i%count);tree=s.step()
   for k in ('nodes','leaf_values','leaf_weights'):np.testing.assert_array_equal(getattr(tree,k),getattr(full.trees[i],k))
  np.testing.assert_array_equal(s.predictions(),full.predictions)


@pytest.mark.parametrize('policy',POLICIES)
@pytest.mark.parametrize('mode',['No','AnyImprovement','Armijo'])
def test_unpaired_documents_and_duplicate_zero_weight_edges(policy,mode):
 a=args(policy,leaf_estimation_backtracking=mode)
 keep=(a['pair_winners']<230)&(a['pair_losers']<230)
 for k in ('pair_winners','pair_losers','pair_weights'):a[k]=a[k][keep]
 a['pair_winners']=np.r_[a['pair_winners'],a['pair_winners'][:10]].astype(np.uint32)
 a['pair_losers']=np.r_[a['pair_losers'],a['pair_losers'][:10]].astype(np.uint32)
 a['pair_weights']=np.r_[a['pair_weights'],np.zeros(10,np.float32)]
 a['group_offsets']=None;raw=a['initial_predictions'].copy();r=_greedy.train(**a)
 for tree in r.trees:
  ids=route(tree,a['bins']);v,w,_=leaf_reference(a,raw,ids,len(tree.leaf_values))
  np.testing.assert_allclose(tree.leaf_values,v,atol=2e-5,rtol=4e-4)
  assert sum(tree.leaf_weights)==pytest.approx(2*a['pair_weights'].sum(dtype=float),rel=3e-6)
  raw+=tree.leaf_values[ids]
 np.testing.assert_array_equal(raw,r.predictions)


@pytest.mark.parametrize('bad',[{'pair_winners':None},{'pair_weights':[-1.]},{'group_offsets':[0,1,261]},
 {'sample_weight':np.ones(261)},{'leaf_estimation_method':'Exact'}])
def test_invalid_pair_inputs_fail_before_gpu(monkeypatch,bad):
 def forbidden():pytest.fail('invalid pair data reached GPU build')
 monkeypatch.setattr(_greedy,'build_library',forbidden)
 with pytest.raises(ValueError):_greedy.TrainingSession(**(args()|bad))


@pytest.mark.parametrize('pair_count,offset_count',[(0,0),(2,0),(1,2)])
def test_pair_cabi_rejects_bad_counts_before_pointer_reads(pair_count,offset_count):
 lib=_greedy._load(_greedy.build_library());error=ct.create_string_buffer(2048);handle=ct.c_void_p()
 p=_greedy.Params();p.objective=14;o=ObjectiveOptions(14,0,1,0);pairs=PairOptions(1,0,0,0)
 sentinel=ct.cast(ct.c_void_p(1),ct.POINTER(ct.c_uint32))
 code=lib.cbm_greedy_session_create_pair(ct.byref(p),ct.byref(o),ct.byref(pairs),sentinel,sentinel,None,pair_count,
  sentinel,offset_count,None,None,None,None,None,None,ct.byref(handle),error,len(error))
 assert code and not handle.value and b'count' in error.value
