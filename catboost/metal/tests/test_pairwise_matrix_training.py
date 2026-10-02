"""Full resident trees against independent edge matrices, without CPU training."""
import platform
import numpy as np
import pytest
from catboost_metal import _pair_matrix
from test_pairwise_candidate_kernels import reference as candidate_reference
from test_pairwise_matrix_kernels import reference as leaf_reference
from test_pairwise_matrix_runtime import multipliers
from test_pairwise_training import leaf_ids

pytestmark=pytest.mark.skipif(platform.system()!='Darwin' or platform.machine()!='arm64',reason='Apple GPU required')

@pytest.fixture(autouse=True)
def no_cpu_fit(monkeypatch):
    from catboost import CatBoost
    def forbidden(*a,**kw):raise AssertionError('CPU CatBoost training is forbidden')
    monkeypatch.setattr(CatBoost,'_fit',forbidden)


def problem(**extra):
    rng=np.random.default_rng(72293);rows=79;pairs=601
    winners=rng.integers(0,rows,pairs,dtype=np.uint32)
    losers=(winners+rng.integers(1,rows,pairs,dtype=np.uint32))%rows
    result=dict(bins=rng.integers(0,4,(3,rows),dtype=np.uint8),
        candidate_features=np.repeat(np.arange(3,dtype=np.uint32),3),
        candidate_bins=np.tile(np.arange(3,dtype=np.uint32),3),
        candidate_types=np.repeat(np.array([0,1,0],np.uint8),3),
        pair_winners=winners,pair_losers=losers,pair_weights=rng.uniform(.2,2,pairs).astype(np.float32),
        sample_weight=rng.uniform(.2,2,rows).astype(np.float32),
        initial_predictions=rng.normal(0,.4,rows).astype(np.float32),iterations=3,depth=3,
        learning_rate=.17,l2_leaf_reg=3.,non_diagonal_regularization=.2,leaf_estimation_iterations=3,
        score_function='NewtonL2',leaf_estimation_method='Newton',bootstrap_type='No',random_seed=718)
    result['sample_weight'][::7]=0
    result.update(extra);return result


def oracle(args,cursor,iteration,feature_weights=None):
    count=len(args['candidate_features']);ids=np.zeros(len(cursor),np.uint32);selected=[]
    kind={'No':0,'Bayesian':1,'Bernoulli':2,'Poisson':3}[args['bootstrap_type']]
    sampled=np.float32(args['pair_weights']*multipliers(len(args['pair_weights']),kind,args['random_seed'],iteration))
    previous=0.
    for depth in range(args['depth']):
        if not count:break
        candidates=(args['bins'],ids,cursor,args['pair_winners'],args['pair_losers'],sampled,
                    args['candidate_features'],args['candidate_bins'],args['candidate_types'])
        result=candidate_reference(candidates,1<<depth,'Gradient' if args['score_function']=='L2' else 'Newton',
            0,count,args['l2_leaf_reg'],args['non_diagonal_regularization'])
        scores=np.array([v[3] for v in result]);gains=scores+previous
        if feature_weights is not None:gains*=feature_weights[args['candidate_features']]
        chosen=int(np.argmax(gains));previous=-scores[chosen];selected.append(chosen)
        f,b,t=(args[k][chosen] for k in ('candidate_features','candidate_bins','candidate_types'))
        predicate=(args['bins'][f]==b) if t else (args['bins'][f]>b)
        ids|=predicate.astype(np.uint32)<<depth
    leaves=1<<len(selected);values=np.zeros(leaves,np.float32)
    weights=np.bincount(ids,weights=args['sample_weight'].astype(float),minlength=leaves)
    for _ in range(args['leaf_estimation_iterations']):
        ref=leaf_reference(np.float32(cursor+values[ids]),args['pair_winners'],args['pair_losers'],args['pair_weights'],
            ids,leaves,args['leaf_estimation_method'],args['l2_leaf_reg'],args['non_diagonal_regularization'])
        values=np.float32(values+ref['direction']);values[weights<=1e-20]=0;values[-1]=0
    values=np.float32(values-values.mean(dtype=float));values=np.float32(values*np.float32(args['learning_rate']))
    return selected,ids,values,weights


@pytest.mark.parametrize('score',['L2','Cosine','NewtonL2','NewtonCosine','SolarL2','LOOL2','SatL2'])
@pytest.mark.parametrize('method',['Newton','Gradient'])
def test_complete_forest_matches_independent_coupled_search_and_leaf_walk(score,method):
    args=problem(score_function=score,leaf_estimation_method=method)
    cursor=args['initial_predictions'].copy()
    with _pair_matrix.Session(**args) as session:
        for iteration in range(args['iterations']):
            chosen,ids,values,weights=oracle(args,cursor,iteration)
            tree=session.step()
            for attr,key in [('split_features','candidate_features'),('split_bins','candidate_bins'),('split_types','candidate_types')]:
                np.testing.assert_array_equal(getattr(tree,attr),args[key][chosen])
            np.testing.assert_array_equal(leaf_ids(tree,args['bins']),ids)
            np.testing.assert_allclose(tree.leaf_values,values,rtol=1e-4,atol=2e-7)
            np.testing.assert_allclose(tree.leaf_weights,weights,rtol=3e-6,atol=1e-6)
            cursor=np.float32(cursor+tree.leaf_values[ids]);np.testing.assert_array_equal(session.predictions(),cursor)
            differences=cursor[args['pair_winners']].astype(float)-cursor[args['pair_losers']]
            expected=np.average(np.logaddexp(0.,-differences),weights=args['pair_weights'])
            assert tree.loss==pytest.approx(expected,rel=2e-6)
        assert session.result().stats['kernel_dispatches']>0


@pytest.mark.parametrize('kind',['No','Bayesian','Bernoulli','Poisson'])
@pytest.mark.parametrize('score',['L2','NewtonL2'])
def test_edge_sampling_is_frozen_for_search_and_original_edges_restore_for_leaves(kind,score):
    args=problem(iterations=1,bootstrap_type=kind,bagging_temperature=.7,subsample=.7,score_function=score)
    chosen,ids,values,weights=oracle(args,args['initial_predictions'],0)
    with _pair_matrix.Session(**args) as session:tree=session.step()
    np.testing.assert_array_equal(tree.split_features,args['candidate_features'][chosen])
    np.testing.assert_array_equal(tree.split_bins,args['candidate_bins'][chosen])
    np.testing.assert_allclose(tree.leaf_values,values,rtol=1e-4,atol=2e-7)
    np.testing.assert_allclose(tree.leaf_weights,weights,rtol=3e-6)


@pytest.mark.parametrize('kind',['No','Bayesian','Bernoulli','Poisson'])
def test_resume_and_incremental_search_are_identical(kind):
    args=problem(iterations=4,bootstrap_type=kind,bagging_temperature=.7,subsample=.7)
    with _pair_matrix.Session(**args) as session:
        expected=[session.step() for _ in range(4)];final=session.predictions()
    with _pair_matrix.Session(**{**args,'iterations':2}) as session:
        for _ in range(2):session.step()
        cursor=session.predictions()
    with _pair_matrix.Session(**{**args,'iterations':2,'initial_predictions':cursor,'iteration_offset':2}) as session:
        for index in range(2,4):
            session.begin_tree()
            for depth in range(args['depth']):
                status=session.grow_tree();assert status['depth']==depth+1
            tree=session.finish_tree()
            for attr in ('split_features','split_bins','split_types','leaf_values','leaf_weights'):
                np.testing.assert_array_equal(getattr(tree,attr),getattr(expected[index],attr))
        np.testing.assert_array_equal(session.predictions(),final)


@pytest.mark.parametrize('depth',[0,1,4,8])
def test_duplicate_winner_retained_to_full_depth_and_empty_leaves_centered(depth):
    args=problem(iterations=1,depth=depth,candidate_features=np.array([0],np.uint32),
        candidate_bins=np.array([1],np.uint32),candidate_types=np.array([0],np.uint8))
    with _pair_matrix.Session(**args) as session:tree=session.step()
    assert tree.depth==depth
    np.testing.assert_array_equal(tree.split_features,np.zeros(depth,np.uint32))
    np.testing.assert_array_equal(tree.split_bins,np.ones(depth,np.uint32))
    _,_,values,weights=oracle(args,args['initial_predictions'],0)
    np.testing.assert_allclose(tree.leaf_values,values,rtol=1e-4,atol=2e-7)
    np.testing.assert_allclose(tree.leaf_weights,weights,rtol=3e-6)
    assert abs(tree.leaf_values.mean(dtype=float))<1e-8


@pytest.mark.parametrize('score',['L2','Cosine','NewtonL2','NewtonCosine','SolarL2','LOOL2','SatL2'])
def test_cuda_pairwise_search_ignores_score_noise(score):
    args=problem(iterations=1,score_function=score)
    with _pair_matrix.Session(**args) as session:expected=session.step()
    with _pair_matrix.Session(**{**args,'random_strength':1000.}) as session:actual=session.step()
    for attr in ('split_features','split_bins','split_types','leaf_values','leaf_weights'):
        np.testing.assert_array_equal(getattr(actual,attr),getattr(expected,attr))


@pytest.mark.parametrize('bad',[{'depth':9},{'bootstrap_type':'MVS'},
    {'leaf_estimation_method':'Exact'},{'non_diagonal_regularization':-1},{'non_diagonal_regularization':float('nan')},
    {'group_offsets':np.array([0,40,79],np.uint32)}])
def test_unconnected_or_invalid_configuration_is_rejected(bad):
    with pytest.raises(ValueError):_pair_matrix.Session(**problem(**bad))


def backtracking_oracle(args,cursor,ids,count):
    values=np.zeros(count,np.float32)
    weights=np.bincount(ids,weights=args['sample_weight'].astype(float),minlength=count)
    winner,loser=args['pair_winners'],args['pair_losers']
    mask=ids[winner]!=ids[loser]
    def objective(point):
        raw=np.float32(cursor+point[ids])
        diff=np.float32(raw[winner[mask]]-raw[loser[mask]])
        return -np.dot(args['pair_weights'][mask].astype(float),np.logaddexp(0.,-diff.astype(float)))
    current=objective(values);new=True;updated=False;step=np.float32(1);attempts=0;rejects=0
    while attempts<args['leaf_estimation_iterations'] or (not updated and attempts<100):
        if new:
            ref=leaf_reference(np.float32(cursor+values[ids]),winner,loser,args['pair_weights'],ids,count,
                args['leaf_estimation_method'],args['l2_leaf_reg'],args['non_diagonal_regularization'])
            direction=ref['direction'].astype(np.float32)
            dot=np.dot(ref['gradient'].astype(np.float32).astype(float),direction.astype(float))
        trial=np.float32(values.astype(float)+float(step)*direction.astype(float))
        trial[weights<1e-20]=0;trial[-1]=0
        score=objective(trial);threshold=current+(1e-5*float(step)*dot if args['leaf_estimation_backtracking']=='Armijo' else 0)
        if np.isfinite(score) and score>=threshold:
            values=trial;current=score;updated=True;new=True;step=np.float32(1)
        else:step=np.float32(step*.5);new=False;rejects+=1
        attempts+=1
    values=np.float32(values-values.mean(dtype=float))*np.float32(args['learning_rate'])
    return values,rejects


@pytest.mark.parametrize('mode',['AnyImprovement','Armijo'])
@pytest.mark.parametrize('method',['Newton','Gradient'])
@pytest.mark.parametrize('iterations',[2,5,10])
def test_coupled_backtracking_matches_independent_acceptance_and_retry_budget(mode,method,iterations):
    args=problem(iterations=2,leaf_estimation_method=method,leaf_estimation_iterations=iterations,
        leaf_estimation_backtracking=mode,l2_leaf_reg=.02,non_diagonal_regularization=.01)
    args['initial_predictions']*=12
    cursor=args['initial_predictions'].copy()
    with _pair_matrix.Session(**args) as session:
        for _ in range(2):
            tree=session.step();ids=leaf_ids(tree,args['bins'])
            expected,rejections=backtracking_oracle(args,cursor,ids,1<<tree.depth)
            np.testing.assert_allclose(tree.leaf_values,expected,rtol=2e-4,atol=5e-6)
            cursor=np.float32(cursor+tree.leaf_values[ids])


@pytest.mark.parametrize('mode',['AnyImprovement','Armijo'])
def test_one_leaf_iteration_bypasses_backtracking_like_cuda(mode):
    args=problem(iterations=1,leaf_estimation_iterations=1)
    with _pair_matrix.Session(**args) as session:expected=session.step()
    with _pair_matrix.Session(**{**args,'leaf_estimation_backtracking':mode}) as session:actual=session.step()
    np.testing.assert_array_equal(actual.leaf_values,expected.leaf_values)


@pytest.mark.parametrize('mode',['AnyImprovement','Armijo'])
def test_pairwise_backtracking_resume_is_exact(mode):
    args=problem(iterations=4,leaf_estimation_iterations=6,leaf_estimation_backtracking=mode,
        bootstrap_type='Bernoulli',subsample=.7,l2_leaf_reg=.02,non_diagonal_regularization=.01)
    args['initial_predictions']*=10
    with _pair_matrix.Session(**args) as session:
        trees=[session.step() for _ in range(4)];expected=session.predictions()
    with _pair_matrix.Session(**{**args,'iterations':2}) as session:
        for _ in range(2):session.step()
        partial=session.predictions()
    with _pair_matrix.Session(**{**args,'iterations':2,'initial_predictions':partial,'iteration_offset':2}) as session:
        for index in range(2,4):
            actual=session.step();np.testing.assert_array_equal(actual.leaf_values,trees[index].leaf_values)
        np.testing.assert_array_equal(session.predictions(),expected)

@pytest.mark.parametrize('mode',['AnyImprovement','Armijo'])
@pytest.mark.parametrize('iterations',[2,5,12])
def test_saturated_conflicting_edges_force_halving_before_acceptance(mode,iterations):
    args=problem(iterations=1,depth=1,leaf_estimation_iterations=iterations,leaf_estimation_backtracking=mode,
        bins=np.array([[0,1]],np.uint8),candidate_features=np.array([0],np.uint32),
        candidate_bins=np.array([0],np.uint32),candidate_types=np.array([0],np.uint8),
        pair_winners=np.array([0]*9+[1],np.uint32),pair_losers=np.array([1]*9+[0],np.uint32),
        pair_weights=np.ones(10,np.float32),sample_weight=np.ones(2,np.float32),
        initial_predictions=np.array([-5,5],np.float32),l2_leaf_reg=.02,non_diagonal_regularization=.01)
    with _pair_matrix.Session(**args) as session:tree=session.step()
    expected,rejections=backtracking_oracle(args,args['initial_predictions'],np.array([0,1],np.uint32),2)
    assert rejections>=2
    np.testing.assert_allclose(tree.leaf_values,expected,rtol=3e-4,atol=1e-5)
