"""Complete QCE forests against independent optimized-query matrix oracles."""
import platform
import numpy as np
import pytest
from catboost_metal import _query_cross_entropy
from test_query_cross_entropy_kernels import reference as target_reference
from test_qce_candidate_kernels import reference as candidate_reference
from test_pairwise_training import leaf_ids
from test_bootstrap import uniforms

pytestmark=pytest.mark.skipif(platform.system()!='Darwin' or platform.machine()!='arm64',reason='Apple GPU required')

@pytest.fixture(autouse=True)
def no_cpu_fit(monkeypatch):
    from catboost import CatBoost
    def forbidden(*a,**kw):raise AssertionError('CPU CatBoost training is forbidden')
    monkeypatch.setattr(CatBoost,'_fit',forbidden)


def problem(**extra):
    rng=np.random.default_rng(84291);offsets=np.r_[0,np.cumsum([1,4,7,3,16,2,11,5])].astype(np.uint32);rows=int(offsets[-1])
    result=dict(bins=rng.integers(0,4,(3,rows),dtype=np.uint8),targets=rng.integers(0,2,rows).astype(np.float32),
        candidate_features=np.repeat(np.arange(3,dtype=np.uint32),3),candidate_bins=np.tile(np.arange(3,dtype=np.uint32),3),
        candidate_types=np.repeat(np.array([0,1,0],np.uint8),3),group_offsets=offsets,
        query_scales=rng.uniform(.5,1.6,len(offsets)-1).astype(np.float32),alpha=.95,
        sample_weight=rng.uniform(.2,2,rows).astype(np.float32),initial_predictions=rng.normal(0,.4,rows).astype(np.float32),
        iterations=3,depth=3,learning_rate=.17,l2_leaf_reg=1.,non_diagonal_regularization=.2,leaf_estimation_iterations=3,
        score_function='NewtonL2',leaf_estimation_method='Newton',bootstrap_type='No',subsample=.7,random_seed=718,
        leaf_estimation_backtracking='No')
    result['sample_weight'][::7]=0;result.update(extra);return result


def target(args,cursor,ids,leaves):
    return target_reference(args['targets'],args['sample_weight'],cursor,args['group_offsets'],ids,
        alpha=args['alpha'],scales=args['query_scales'],leaf_count=leaves)


def direction(args,cursor,ids,leaves):
    _,qs,_,g,h=target(args,cursor,ids,leaves)
    # CUDA leaf oracle: empty diagonal becomes 10, then full L2 and Laplace ridge.
    matrix=h.astype(np.float32).astype(float);g=g.astype(np.float32)
    for i in range(leaves):
        if matrix[i,i]==0:matrix[i,i]=10
    matrix+=np.eye(leaves)*(args['l2_leaf_reg']+args['non_diagonal_regularization'])-args['non_diagonal_regularization']/leaves
    return np.linalg.solve(matrix,g).astype(np.float32),g,qs[:,2].sum()


def leaf_walk(args,cursor,ids,leaves):
    values=np.zeros(leaves,np.float32);weights=np.bincount(ids,weights=args['sample_weight'].astype(float),minlength=leaves)
    count=args['leaf_estimation_iterations'];mode=args['leaf_estimation_backtracking'];rejects=0
    if mode=='No' or count==1:
        for _ in range(count):
            step,_,_=direction(args,np.float32(cursor+values[ids]),ids,leaves)
            values=np.float32(values+step);values[weights<1e-20]=0
    else:
        current=target(args,cursor,ids,leaves)[1][:,2].sum();new=True;updated=False;attempt=0;scale=1.
        while attempt<count or (not updated and attempt<100):
            if new:
                step,g,_=direction(args,np.float32(cursor+values[ids]),ids,leaves)
                dot=step.astype(float)@g
            trial=np.float32(values+np.float32(scale)*step);trial[weights<1e-20]=0
            value=target(args,np.float32(cursor+trial[ids]),ids,leaves)[1][:,2].sum()
            if value<=current-(1e-5*scale*dot if mode=='Armijo' else 0):
                values=trial;current=value;new=True;updated=True;scale=1.
            else:scale*=.5;new=False;rejects+=1
            attempt+=1
    return np.float32(values*np.float32(args['learning_rate'])),weights,rejects


def oracle(args,cursor,iteration):
    ids=np.zeros(len(cursor),np.uint32);chosen=[]
    stats,groups,*_=target(args,cursor,ids,1)
    active=np.ones(len(groups),np.uint8) if args['bootstrap_type']=='No' else (uniforms(len(groups),seed=args['random_seed'],iteration=iteration)<np.float32(args['subsample'])).astype(np.uint8)
    for depth in range(args['depth']):
        if not len(args['candidate_features']):break
        ref=candidate_reference((stats.astype(np.float32),groups.astype(np.float32),args['group_offsets'],args['bins'],ids,
            args['candidate_features'],args['candidate_bins'],args['candidate_types'],active),1<<depth,args['l2_leaf_reg'],args['non_diagonal_regularization'])
        c=int(np.argmax([r[3] for r in ref]));chosen.append(c)
        f,b,t=(args[k][c] for k in ('candidate_features','candidate_bins','candidate_types'))
        ids|=(((args['bins'][f]==b) if t else (args['bins'][f]>b)).astype(np.uint32)<<depth)
    leaves=1<<len(chosen);values,weights,rejects=leaf_walk(args,cursor,ids,leaves)
    return chosen,ids,values,weights,rejects


@pytest.mark.parametrize('score',['Cosine','NewtonL2','NewtonCosine','SolarL2','LOOL2','SatL2'])
@pytest.mark.parametrize('kind',['No','Bernoulli'])
def test_full_forests_match_original_query_leaf_matrices_and_frozen_query_sampling(score,kind):
    args=problem(score_function=score,bootstrap_type=kind);cursor=args['initial_predictions'].copy()
    with _query_cross_entropy.Session(**args) as session:
        for iteration in range(args['iterations']):
            chosen,ids,values,weights,_=oracle(args,cursor,iteration);tree=session.step()
            for attr,key in [('split_features','candidate_features'),('split_bins','candidate_bins'),('split_types','candidate_types')]:
                np.testing.assert_array_equal(getattr(tree,attr),args[key][chosen])
            np.testing.assert_array_equal(leaf_ids(tree,args['bins']),ids)
            np.testing.assert_allclose(tree.leaf_values,values,rtol=2e-4,atol=2e-6)
            np.testing.assert_allclose(tree.leaf_weights,weights,rtol=3e-6,atol=1e-6)
            cursor=np.float32(cursor+tree.leaf_values[ids]);np.testing.assert_array_equal(session.predictions(),cursor)
            assert tree.loss==pytest.approx(target(args,cursor,ids,len(values))[1][:,2].sum()/args['sample_weight'].sum(dtype=float),rel=3e-6)
        assert session.result().stats['kernel_dispatches']>0


@pytest.mark.parametrize('alpha',[0,.37,.95,1])
@pytest.mark.parametrize('depth',[0,2])
def test_soft_labels_and_scale_keep_point_diagonal_and_absolute_leaf_mean(alpha,depth):
    args=problem(alpha=alpha,depth=depth,iterations=1)
    args['targets']=np.linspace(.12,.86,len(args['targets']),dtype=np.float32)
    expected=oracle(args,args['initial_predictions'],0)
    with _query_cross_entropy.Session(**args) as session:tree=session.step()
    np.testing.assert_allclose(tree.leaf_values,expected[2],rtol=3e-4,atol=3e-6)
    if alpha<1:assert abs(tree.leaf_values.mean())>1e-4


@pytest.mark.parametrize('mode',['AnyImprovement','Armijo'])
@pytest.mark.parametrize('count',[1,2,5,10])
def test_full_matrix_backtracking_matches_independent_objective_acceptance(mode,count):
    args=problem(iterations=1,leaf_estimation_iterations=count,leaf_estimation_backtracking=mode)
    expected=oracle(args,args['initial_predictions'],0)
    with _query_cross_entropy.Session(**args) as session:tree=session.step()
    np.testing.assert_allclose(tree.leaf_values,expected[2],rtol=3e-4,atol=3e-6)


@pytest.mark.parametrize('kind',['No','Bernoulli'])
@pytest.mark.parametrize('mode',['No','AnyImprovement','Armijo'])
def test_resume_and_staged_search_preserve_query_sampler_and_leaf_walk(kind,mode):
    args=problem(iterations=4,bootstrap_type=kind,leaf_estimation_backtracking=mode)
    with _query_cross_entropy.Session(**args) as session:
        trees=[session.step() for _ in range(4)];expected=session.predictions()
    with _query_cross_entropy.Session(**{**args,'iterations':2}) as session:
        for _ in range(2):session.step()
        cursor=session.predictions()
    with _query_cross_entropy.Session(**{**args,'iterations':2,'initial_predictions':cursor,'iteration_offset':2}) as session:
        for index in range(2,4):
            session.begin_tree()
            for depth in range(args['depth']):assert session.grow_tree()['depth']==depth+1
            tree=session.finish_tree()
            for attr in ('split_features','split_bins','split_types','leaf_values','leaf_weights'):
                np.testing.assert_array_equal(getattr(tree,attr),getattr(trees[index],attr))
        np.testing.assert_array_equal(session.predictions(),expected)


@pytest.mark.parametrize('changes',[
    {'alpha':-1},{'alpha':np.nan},{'alpha':2},{'non_diagonal_regularization':-1},
    {'leaf_estimation_method':'Gradient'},{'leaf_estimation_method':'Exact'},{'score_function':'L2'},
    {'bootstrap_type':'Bayesian'},{'bootstrap_type':'Poisson'},{'bootstrap_type':'MVS'},{'depth':9},
    {'query_scales':[1]},{'query_scales':[np.inf]*8},{'targets':[-1]*49},
])
def test_rejects_unsupported_cuda_methods_or_invalid_metadata(changes):
    with pytest.raises(ValueError):_query_cross_entropy.Session(**problem(**changes))


@pytest.mark.parametrize('mode',['No','AnyImprovement','Armijo'])
def test_one_candidate_retains_repeated_splits_to_depth_eight(mode):
    args=problem(iterations=1,depth=8,leaf_estimation_iterations=2,leaf_estimation_backtracking=mode,
        candidate_features=np.array([0],np.uint32),candidate_bins=np.array([1],np.uint32),candidate_types=np.array([0],np.uint8))
    with _query_cross_entropy.Session(**args) as session:tree=session.step()
    assert len(tree.split_features)==8
    ids=leaf_ids(tree,args['bins']);values,weights,_=leaf_walk(args,args['initial_predictions'],ids,256)
    np.testing.assert_allclose(tree.leaf_values,values,rtol=3e-4,atol=3e-6)
    np.testing.assert_allclose(tree.leaf_weights,weights,rtol=3e-6)


@pytest.mark.parametrize('mode',['AnyImprovement','Armijo'])
@pytest.mark.parametrize('count',[2,5,12])
def test_saturated_soft_targets_force_multiple_step_halvings(mode,count):
    args=problem(iterations=1,depth=1,leaf_estimation_iterations=count,leaf_estimation_backtracking=mode,
        bins=np.array([[0,1]],np.uint8),targets=np.array([.9,.1],np.float32),group_offsets=np.array([0,2],np.uint32),
        query_scales=np.array([1.2],np.float32),sample_weight=np.ones(2,np.float32),initial_predictions=np.array([-5,5],np.float32),
        candidate_features=np.array([0],np.uint32),candidate_bins=np.array([0],np.uint32),candidate_types=np.array([0],np.uint8),
        l2_leaf_reg=.002,non_diagonal_regularization=.001,alpha=.7)
    expected,_,rejections=leaf_walk(args,args['initial_predictions'],np.array([0,1],np.uint32),2)
    assert rejections>=2
    with _query_cross_entropy.Session(**args) as session:tree=session.step()
    np.testing.assert_allclose(tree.leaf_values,expected,rtol=3e-4,atol=1e-5)
