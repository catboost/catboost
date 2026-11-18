"""Vector variable-tree GPU training, fixed-topology algebra and exact recovery."""
import ctypes as ct
import numpy as np
import pytest
from catboost_metal import _multiclass, _greedy
from test_multiclass import reference as class_math
from test_greedy_training import POLICIES, tree_depths, MISSING
from test_greedy_vector_scores import score_terms

OBJECTIVES=('MultiClass','MultiClassOneVsAll','RMSEWithUncertainty')
SCORES=('L2','Cosine','SolarL2','LOOL2','SatL2')

@pytest.fixture(autouse=True)
def no_cpu_fit(monkeypatch):
    from catboost import CatBoost
    def forbidden(*a,**k): raise AssertionError('CPU CatBoost fitting is forbidden')
    monkeypatch.setattr(CatBoost,'_fit',forbidden)


def problem(objective='MultiClass',policy='Depthwise',count=1,classes=3,**overrides):
    rng=np.random.default_rng(906513);rows=263
    bins=rng.integers(0,8,(4,rows),dtype=np.uint8)
    y=((bins[0].astype(int)+2*(bins[1]>3)+bins[2])%classes).astype(np.uint32)
    if objective=='RMSEWithUncertainty':
        classes=2;y=np.float32((bins[0].astype(float)-3)/4+rng.normal(0,.3,rows)*(1+bins[1]/8))
    w=rng.uniform(.1,3,rows).astype(np.float32);w[::19]=0
    cf=np.repeat(np.arange(4,dtype=np.uint32),7);cb=np.tile(np.arange(7,dtype=np.uint32),4)
    types=np.uint8(cf==2)
    banks=np.stack([bins,*[rng.integers(0,8,bins.shape,dtype=np.uint8) for _ in range(count-1)]])
    initial=rng.normal(0,.1,(count,rows,classes)).astype(np.float32)
    if objective=='MultiClass':initial+=rng.uniform(-2,2,(count,rows,1)).astype(np.float32)
    args=dict(bins=bins,targets=y,sample_weight=w,candidate_features=cf,candidate_bins=cb,candidate_types=types,
        classes=classes,objective=objective,grow_policy=policy,max_leaves=7,depth=3,iterations=3,
        initial_predictions=initial[0],learning_rate=.19,l2_leaf_reg=2.3,leaf_estimation_iterations=3,
        leaf_estimation_method='Newton',score_function='L2',random_seed=617735,subsample=.71)
    args.update(overrides)
    return args,banks,initial


def route(tree,bins):
    ids=np.zeros(bins.shape[1],np.uint32)
    for row in range(len(ids)):
        node=0
        for _ in range(len(tree.nodes)):
            f,b,t,left,right,leaf=map(int,tree.nodes[node])
            if leaf!=MISSING:ids[row]=leaf;break
            node=right if (bins[f,row]==b if t else bins[f,row]>b) else left
        else:raise AssertionError('Invalid cyclic topology')
    return ids


def algebra(args,active,ids,leaves):
    obj=args['objective'];w=args['sample_weight'].astype(float);y=args['targets']
    if obj!='RMSEWithUncertainty':
        return class_math(active,y,w,args['classes'],obj,ids,leaves,args['l2_leaf_reg'],args['leaf_estimation_method'])
    error=y.astype(float)-active[0];norm=error**2*np.exp(np.minimum(-2*active[1],70))
    g=np.array([w*error,w*(norm-1)]);h=np.array([w,2*w*norm]);values=np.zeros((leaves,2))
    for leaf in range(leaves):
        take=ids==leaf
        if w[take].sum()<=1e-20:continue
        den=h[:,take].sum(axis=1) if args['leaf_estimation_method']=='Newton' else w[take].sum()
        values[leaf]=g[:,take].sum(axis=1)/(den+args['l2_leaf_reg'])
    loss=w*(.9189385332046+active[1]+.5*norm)
    return g,h,loss,values


def fixed(args,tree,bins,active):
    ids=route(tree,bins);leaves=len(tree.leaf_weights);base=active.astype(float);values=np.zeros((leaves,len(base)))
    for _ in range(args['leaf_estimation_iterations']):
        values+=algebra(args,base+values[ids].T,ids,leaves)[-1]
    full=np.zeros((leaves,args['classes']));full[:,:len(base)]=values*args['learning_rate']
    return full,np.bincount(ids,weights=args['sample_weight'].astype(float),minlength=leaves),ids


@pytest.mark.parametrize('objective',OBJECTIVES)
@pytest.mark.parametrize('policy',POLICIES)
@pytest.mark.parametrize('score',SCORES)
@pytest.mark.parametrize('method',['Newton','Gradient'])
def test_variable_tree_leaf_iterations_match_independent_math(objective,policy,score,method):
    args,banks,initial=problem(objective,policy,score_function=score,leaf_estimation_method=method)
    with _multiclass.Session(**args) as session:
        for iteration in range(args['iterations']):
            active=session.optimization_predictions();before=session.predictions()
            tree=session.step();values,weights,ids=fixed(args,tree,banks[0],active)
            np.testing.assert_allclose(tree.leaf_values,values,rtol=7e-5,atol=4e-6)
            np.testing.assert_allclose(tree.leaf_weights,weights,rtol=3e-6,atol=1e-5)
            np.testing.assert_allclose(session.predictions(),before+values[ids],rtol=7e-5,atol=4e-6)
            raw=session.optimization_predictions()
            expected_loss=algebra(args,raw.astype(float),ids,len(weights))[2].sum()/args['sample_weight'].astype(float).sum()
            np.testing.assert_allclose(tree.loss,expected_loss,rtol=2e-5,atol=2e-6)
            depths=tree_depths(tree);assert max(depths.values())<=3
            assert len(tree.nodes)==2*len(depths)-1
            assert len(depths)<=({'Depthwise':8,'Lossguide':7,'Region':4}[policy])
            if policy=='Region':
                for node in tree.nodes:
                    if node[5]==MISSING:assert sum(tree.nodes[child,5]==MISSING for child in node[3:5])<=1
            if objective=='MultiClass':
                np.testing.assert_array_equal(session.predictions()[:,-1],initial[0,:,-1]);assert not tree.leaf_values[:,-1].any()
        assert session.result().completed_iterations==3
        assert session.result().stats['kernel_dispatches']>0


@pytest.mark.parametrize('objective',OBJECTIVES)
@pytest.mark.parametrize('score',SCORES)
def test_root_selection_matches_cuda_equations(objective,score):
    args,_,_=problem(objective,'Lossguide',depth=1,max_leaves=2,iterations=1,score_function=score)
    with _multiclass.Session(**args) as session:
        active=session.optimization_predictions().astype(float)
        g=algebra(args,active,np.zeros(args['bins'].shape[1],np.uint32),1)[0]
        w=args['sample_weight'].astype(float);parents=g.sum(axis=1)
        missing=objective=='MultiClass'
        kind={'L2':0,'Cosine':1,'SolarL2':4,'LOOL2':5,'SatL2':6}[score]
        def terms(grads,weight):
            values=list(grads)
            if missing:values.append(-sum(values))
            return [(v,weight) for v in values]
        before=score_terms(terms(parents,w.sum()),kind,0,args['l2_leaf_reg']);scores=[]
        for f,b,t in zip(args['candidate_features'],args['candidate_bins'],args['candidate_types']):
            take=args['bins'][f]==b if t else args['bins'][f]<=b
            a=g[:,take].sum(axis=1);wa=w[take].sum();wb=w[~take].sum()
            # CUDA interleaves left/right coordinates when accumulating score.
            left=terms(a,wa);right=terms(parents-a,wb)
            after=score_terms([v for pair in zip(left,right) for v in pair],kind,0,args['l2_leaf_reg'])
            scores.append(0 if min(wa,wb)<1e-20 else after-before)
        winner=int(np.argmin(scores));tree=session.step()
        np.testing.assert_array_equal(tree.nodes[0,:3],[args['candidate_features'][winner],args['candidate_bins'][winner],args['candidate_types'][winner]])


@pytest.mark.parametrize('objective',OBJECTIVES)
@pytest.mark.parametrize('policy',POLICIES)
@pytest.mark.parametrize('sampler',['No','Bayesian','Bernoulli','Poisson'])
def test_each_permutation_estimates_fixed_tree_on_own_optimizer(objective,policy,sampler):
    args,banks,_=problem(objective,policy,count=4,bootstrap_type=sampler,random_strength=.27,score_function='Cosine')
    with _multiclass.Session(**args) as session:
        session.configure_permutations(banks)
        for iteration,search in enumerate([3,0,2]):
            cursors=session.permutation_state['predictions'];active=session.optimization_predictions(all_permutations=True)
            with _multiclass.Session(**(args|dict(bins=banks[search],iterations=1,iteration_offset=iteration,
                    initial_predictions=cursors[search],initial_optimization_predictions=active[search]))) as single:
                searched=single.step()
            session.select_permutation(search);tree=session.step();state=session.permutation_state
            np.testing.assert_array_equal(tree.nodes,searched.nodes)
            for p,bank in enumerate(banks):
                values,weights,ids=fixed(args,tree,bank,active[p])
                np.testing.assert_allclose(state['predictions'][p],cursors[p]+values[ids],rtol=1e-4,atol=6e-6)
                if p==3:
                    np.testing.assert_allclose(tree.leaf_values,values,rtol=1e-4,atol=6e-6)
                    np.testing.assert_allclose(tree.leaf_weights,weights,rtol=3e-6,atol=1e-5)
            np.testing.assert_array_equal(session.predictions(),state['predictions'][-1])


def equal_tree(a,b):
    for key in ('nodes','leaf_values','leaf_weights','loss'):np.testing.assert_array_equal(getattr(a,key),getattr(b,key))


@pytest.mark.parametrize('objective',OBJECTIVES)
@pytest.mark.parametrize('policy',POLICIES)
@pytest.mark.parametrize('mode',['No','AnyImprovement','Armijo'])
@pytest.mark.parametrize('count',[1,4])
def test_exact_optimizer_and_published_recovery(objective,policy,mode,count):
    args,banks,initial=problem(objective,policy,count=count,iterations=4,leaf_estimation_backtracking=mode,
        bootstrap_type='Bernoulli',score_function='Cosine',random_strength=.4)
    def setup(options,cursors,active=None):
        s=_multiclass.Session(**options)
        s.configure_permutations(banks,initial_predictions=cursors,optimization_predictions=active)
        return s
    with setup(args,initial) as full:
        for i in range(4):
            full.select_permutation(i%count);full.step()
            if i==1:prefix=full.permutation_state
        expected=full.result();end=full.permutation_state
    with setup(args|dict(iterations=2,iteration_offset=2),prefix['predictions'],prefix['optimization_predictions']) as resumed:
        for i in range(2,4):
            resumed.select_permutation(i%count);equal_tree(resumed.step(),expected.trees[i])
        np.testing.assert_array_equal(resumed.predictions(),expected.predictions)
        for key in end:np.testing.assert_array_equal(resumed.permutation_state[key],end[key])
        np.testing.assert_array_equal(resumed.result().loss,expected.loss[2:])


@pytest.mark.parametrize('policy',POLICIES)
@pytest.mark.parametrize('count',[1,4])
def test_singular_failure_restores_all_cursors_and_iteration(policy,count):
    args,banks,initial=problem('RMSEWithUncertainty',policy,count=count,iterations=1,l2_leaf_reg=0,depth=0)
    args['targets']=np.zeros(len(args['targets']),np.float32);initial[:]=0
    args['initial_predictions']=initial[0]
    with _multiclass.Session(**args) as session:
        session.configure_permutations(banks,initial_predictions=initial)
        before=session.permutation_state
        for _ in range(2):
            with pytest.raises(RuntimeError,match='leaf solve failed'):session.step()
            assert session.completed_iterations==0
            for key in before:np.testing.assert_array_equal(session.permutation_state[key],before[key])


@pytest.mark.parametrize('objective',['MultiRMSE','MultiLogloss','MultiCrossEntropy'])
def test_unregistered_cuda_vector_greedy_objective_rejected_before_load(monkeypatch,objective):
    args,_,_=problem();args['objective']=objective
    monkeypatch.setattr(_multiclass,'build_library',lambda:pytest.fail('Unsupported objective reached native load'))
    with pytest.raises(ValueError,match='CUDA greedy vector objectives'):_multiclass.Session(**args)


@pytest.mark.parametrize('objective',OBJECTIVES)
@pytest.mark.parametrize('policy',POLICIES)
@pytest.mark.parametrize('empty',['depth','candidates'])
def test_root_only_vector_tree_keeps_all_weight_and_class_coordinates(objective,policy,empty):
    args,banks,_=problem(objective,policy,depth=0 if empty=='depth' else 3,classes=64)
    if empty=='candidates':args.update(candidate_features=[],candidate_bins=[],candidate_types=None)
    with _multiclass.Session(**args) as session:
        active=session.optimization_predictions();tree=session.step()
        values,weights,_=fixed(args,tree,banks[0],active)
        assert tree.nodes.tolist()==[[0,0,0,0,0,0]]
        np.testing.assert_allclose(tree.leaf_values,values,rtol=5e-5,atol=4e-6)
        np.testing.assert_allclose(tree.leaf_weights,weights,rtol=3e-6)


@pytest.mark.parametrize('objective',OBJECTIVES)
@pytest.mark.parametrize('policy',POLICIES)
@pytest.mark.parametrize('count',[1,2,7,64])
def test_identical_banks_preserve_single_dataset_bits(objective,policy,count):
    args,banks,initial=problem(objective,policy,iterations=2,bootstrap_type='Poisson',random_strength=.4,score_function='Cosine')
    expected=_multiclass.train(**args)
    with _multiclass.Session(**args) as session:
        session.configure_permutations(np.repeat(banks,count,axis=0))
        for i in range(2):
            session.select_permutation((count-1)*i);equal_tree(session.step(),expected.trees[i])
        np.testing.assert_array_equal(session.predictions(),expected.predictions)


@pytest.mark.parametrize('objective',OBJECTIVES)
@pytest.mark.parametrize('selected',[0,3])
def test_lossguide_path_depth_is_bounded_by_leaf_capacity(objective,selected):
    args,_,_=problem(objective,'Lossguide',iterations=1,depth=2**32-1,max_leaves=41)
    args.update(bins=np.zeros((1,263),np.uint8),candidate_features=[0],candidate_bins=[0],candidate_types=[0])
    with _multiclass.Session(**args) as session:
        session.configure_permutations(np.repeat(args['bins'][None],4,axis=0));session.select_permutation(selected)
        active=session.optimization_predictions();tree=session.step()
        assert max(tree_depths(tree).values())==40 and len(tree.leaf_weights)==41
        values,weights,_=fixed(args,tree,args['bins'],active)
        np.testing.assert_allclose(tree.leaf_values,values,rtol=5e-5,atol=4e-6)
        np.testing.assert_allclose(tree.leaf_weights,weights,rtol=3e-6)


@pytest.mark.parametrize('options',[dict(grow_policy='Depthwise',depth=17),dict(grow_policy='Region',depth=65536),
    dict(grow_policy='Lossguide',depth=2**32),dict(max_leaves=65537),dict(max_leaves=0),dict(min_data_in_leaf=-1),
    dict(min_data_in_leaf=1.2),dict(grow_policy='bad')])
def test_malformed_grow_limits_rejected_before_native_load(monkeypatch,options):
    args,_,_=problem();args.update(options)
    monkeypatch.setattr(_multiclass,'build_library',lambda:pytest.fail('Invalid limits reached GPU load'))
    with pytest.raises(ValueError):_multiclass.Session(**args)


def test_step_abi_cannot_mix_symmetric_and_variable_tree_buffers():
    args,_,_=problem()
    with _multiclass.Session(**args) as session:
        error=ct.create_string_buffer(1024)
        result=session._lib.cbm_multiclass_session_step(session._handle,None,None,None,None,None,None,None,error,len(error))
        assert result and b'step_greedy' in error.value
        with pytest.raises(RuntimeError,match='Dynamic feature penalties'):
            session.configure_feature_penalties(np.zeros(4,np.uint32))
    args['grow_policy']='SymmetricTree'
    with _multiclass.Session(**args) as session:
        error=ct.create_string_buffer(1024)
        result=session._lib.cbm_multiclass_session_step_greedy(session._handle,None,None,None,None,error,len(error))
        assert result and b'greedy session' in error.value
