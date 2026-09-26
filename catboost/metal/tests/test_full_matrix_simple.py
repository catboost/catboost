"""CUDA Simple leaves reuse the winning sampled split solve and raw diagonal."""
import platform
import numpy as np
import pytest
from catboost_metal import _pair_matrix, _query_cross_entropy, CatBoostMetalRanker
from test_pairwise_matrix_training import problem as pair_problem
from test_query_cross_entropy_training import problem as qce_problem, target as qce_target
from test_pairwise_candidate_kernels import reference as pair_candidates
from test_qce_candidate_kernels import reference as qce_candidates
from test_pairwise_matrix_runtime import multipliers
from test_pairwise_training import leaf_ids
from test_bootstrap import uniforms

pytestmark=pytest.mark.skipif(platform.system()!='Darwin' or platform.machine()!='arm64',reason='Apple GPU required')

@pytest.fixture(autouse=True)
def no_cpu_fit(monkeypatch):
    from catboost import CatBoost
    def forbidden(*a,**kw):raise AssertionError('CPU CatBoost training is forbidden')
    monkeypatch.setattr(CatBoost,'_fit',forbidden)


def problem(target,score='NewtonL2',kind='No',one_hot=False,**extra):
    args=(pair_problem if target=='pair' else qce_problem)(leaf_estimation_method='Simple',leaf_estimation_iterations=1,
        iterations=3,depth=3,score_function=score,bootstrap_type=kind,subsample=.7,bagging_temperature=.7)
    # Pair tiles hold 32 candidates; QCE tiles hold eight. Include multiple tiles
    # so the selected matrix need not be the one left at the end of scoring.
    rng=np.random.default_rng(3917);rows=args['bins'].shape[1];features=13 if target=='pair' else 4
    args.update(bins=rng.integers(0,4,(features,rows),dtype=np.uint8),
        candidate_features=np.repeat(np.arange(features,dtype=np.uint32),3),
        candidate_bins=np.tile(np.arange(3,dtype=np.uint32),features),
        candidate_types=np.full(features*3,int(one_hot),np.uint8))
    args.update(extra);return args


def oracle(args,cursor,iteration,target):
    ids=np.zeros(len(cursor),np.uint32);chosen=[];count=len(args['candidate_features'])
    if target=='pair':
        kind={'No':0,'Bayesian':1,'Bernoulli':2,'Poisson':3}[args['bootstrap_type']]
        sampled=np.float32(args['pair_weights']*multipliers(len(args['pair_weights']),kind,args['random_seed'],iteration))
    else:
        stats,groups,*_=qce_target(args,cursor,ids,1)
        active=np.ones(len(groups),np.uint8) if args['bootstrap_type']=='No' else (uniforms(len(groups),seed=args['random_seed'],iteration=iteration)<np.float32(args['subsample'])).astype(np.uint8)
    for depth in range(args['depth']):
        if target=='pair':
            values=pair_candidates((args['bins'],ids,cursor,args['pair_winners'],args['pair_losers'],sampled,
                args['candidate_features'],args['candidate_bins'],args['candidate_types']),1<<depth,
                'Gradient' if args['score_function']=='L2' else 'Newton',0,count,args['l2_leaf_reg'],args['non_diagonal_regularization'])
        else:
            values=qce_candidates((stats.astype(np.float32),groups.astype(np.float32),args['group_offsets'],args['bins'],ids,
                args['candidate_features'],args['candidate_bins'],args['candidate_types'],active),1<<depth,
                args['l2_leaf_reg'],args['non_diagonal_regularization'])
        c=int(np.argmax([v[3] for v in values]));chosen.append(c)
        f,b,t=(args[k][c] for k in ('candidate_features','candidate_bins','candidate_types'))
        ids|=((args['bins'][f]==b) if t else (args['bins'][f]>b)).astype(np.uint32)<<depth
    _,h,d,_=values[c];leaves=1<<args['depth'];index=np.arange(leaves)
    solver=2*(index%(leaves//2))+((index//(leaves//2))^int(t))
    leaf_values=np.float32(np.float32(d[solver])*np.float32(args['learning_rate']))
    return chosen,ids,leaf_values,np.diag(h)[solver]


@pytest.mark.parametrize('target,score,kind',[
    *[('pair',score,kind) for score in ('L2','Cosine','NewtonL2','NewtonCosine','SolarL2','LOOL2','SatL2') for kind in ('No','Bayesian','Bernoulli','Poisson')],
    *[('qce',score,kind) for score in ('Cosine','NewtonL2','NewtonCosine','SolarL2','LOOL2','SatL2') for kind in ('No','Bernoulli')],
])
@pytest.mark.parametrize('one_hot',[False,True])
def test_simple_forests_match_sampled_candidate_solutions_and_matrix_weights(target,score,kind,one_hot):
    args=problem(target,score,kind,one_hot);cursor=args['initial_predictions'].copy()
    factory=_pair_matrix.Session if target=='pair' else _query_cross_entropy.Session
    with factory(**args) as session:
        for iteration in range(args['iterations']):
            chosen,ids,values,weights=oracle(args,cursor,iteration,target);tree=session.step()
            for attr,key in [('split_features','candidate_features'),('split_bins','candidate_bins'),('split_types','candidate_types')]:
                np.testing.assert_array_equal(getattr(tree,attr),args[key][chosen])
            np.testing.assert_array_equal(leaf_ids(tree,args['bins']),ids)
            np.testing.assert_allclose(tree.leaf_values,values,rtol=4e-4,atol=5e-6)
            np.testing.assert_allclose(tree.leaf_weights,weights,rtol=4e-5,atol=3e-6)
            cursor=np.float32(cursor+tree.leaf_values[ids]);np.testing.assert_array_equal(session.predictions(),cursor)
            assert tree.leaf_weights.sum()!=pytest.approx(args['sample_weight'].sum())


@pytest.mark.parametrize('target',['pair','qce'])
@pytest.mark.parametrize('depth',[1,8])
@pytest.mark.parametrize('mode',['No','AnyImprovement','Armijo'])
def test_simple_extreme_depth_and_backtracking_options_preserve_single_solve(target,depth,mode):
    args=problem(target,depth=depth,iterations=1,leaf_estimation_backtracking=mode)
    # A single split candidate intentionally repeats, including empty leaves.
    for key in ('candidate_features','candidate_bins','candidate_types'):args[key]=args[key][:1]
    factory=_pair_matrix.Session if target=='pair' else _query_cross_entropy.Session
    expected=oracle(args,args['initial_predictions'],0,target)
    with factory(**args) as session:tree=session.step()
    np.testing.assert_allclose(tree.leaf_values,expected[2],rtol=4e-4,atol=5e-6)
    np.testing.assert_allclose(tree.leaf_weights,expected[3],rtol=4e-5,atol=3e-6)


@pytest.mark.parametrize('target',['pair','qce'])
@pytest.mark.parametrize('extra',[{'depth':0},{'leaf_estimation_iterations':2},{'candidate_features':np.array([],np.uint32),'candidate_bins':np.array([],np.uint32),'candidate_types':np.array([],np.uint8)}])
def test_simple_rejects_missing_structure_or_multiple_leaf_iterations(target,extra):
    args=problem(target,**extra);factory=_pair_matrix.Session if target=='pair' else _query_cross_entropy.Session
    with pytest.raises(ValueError,match='Simple'):factory(**args)


@pytest.mark.parametrize('objective',['PairLogitPairwise','QueryCrossEntropy'])
def test_public_simple_defaults_one_iteration_and_rejects_invalid_configs(objective):
    model=CatBoostMetalRanker(loss_function=objective,leaf_estimation_method='Simple')
    assert model.leaf_estimation_method=='Simple' and model.leaf_estimation_iterations==1
    for extra in ({'depth':0},{'leaf_estimation_iterations':2}):
        with pytest.raises(ValueError,match='Simple'):CatBoostMetalRanker(loss_function=objective,leaf_estimation_method='Simple',**extra)
