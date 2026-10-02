"""Complete PFound forests, explicit full matrices, and absolute-iteration replay."""
import platform
import numpy as np
import pytest
from catboost_metal import _yeti_pair
from test_pfound_pair_runtime import target_reference
from test_pairwise_matrix_kernels import reference as leaf_reference
from test_pairwise_score_kernels import stabilized
from test_pairwise_training import leaf_ids

pytestmark=pytest.mark.skipif(platform.system()!='Darwin' or platform.machine()!='arm64',reason='Apple GPU required')
@pytest.fixture(autouse=True)
def no_cpu_fit(monkeypatch):
    from catboost import CatBoost
    monkeypatch.setattr(CatBoost,'_fit',lambda *a,**kw:pytest.fail('CPU CatBoost training is forbidden'))


def problem(**extra):
    rng=np.random.default_rng(8771);rows=79;features=4
    result=dict(bins=rng.integers(0,4,(features,rows),dtype=np.uint8),targets=rng.uniform(0,1,rows).astype(np.float32),
        group_offsets=np.array([0,5,18,32,61,79],np.uint32),sample_weight=rng.uniform(.2,2,rows).astype(np.float32),
        candidate_features=np.repeat(np.arange(features,dtype=np.uint32),3),candidate_bins=np.tile(np.arange(3,dtype=np.uint32),features),
        candidate_types=np.repeat(np.array([0,1,0,1],np.uint8),3),initial_predictions=rng.normal(0,.4,rows).astype(np.float32),
        iterations=3,depth=3,learning_rate=.17,l2_leaf_reg=3.,non_diagonal_regularization=.2,leaf_estimation_iterations=1,
        leaf_estimation_method='Simple',bootstrap_type='No',random_seed=0x12345678abcdef,score_function='NewtonL2',
        bagging_temperature=.7,subsample=.7,permutations=7,sampling_unit='Object')
    result['sample_weight'][::7]=0;result.update(extra);return result


def oracle(args,cursor,iteration,dataset=0):
    data=(args['bins'],args['targets'],args['sample_weight'],cursor,args['group_offsets'])
    opts=dict(bootstrap={'No':0,'Bayesian':1,'Bernoulli':2}[args['bootstrap_type']],seed=args['random_seed'],absolute=iteration,
        subsample=args['subsample'],temperature=args['bagging_temperature'],group_sampling=args['sampling_unit']=='Group',permutations=args['permutations'],dataset=dataset)
    weak=target_reference(data,**opts);ids=np.zeros(len(cursor),np.uint32);chosen=[]
    pairs=weak['pairs'];edges=weak['edges'];count=len(args['candidate_features']);last=None;t=0
    for depth in range(args['depth']):
        if not count:break
        leaves=2<<depth;results=[]
        for feature,border,kind in zip(args['candidate_features'],args['candidate_bins'],args['candidate_types']):
            predicate=args['bins'][feature]!=border if kind else args['bins'][feature]>border;children=2*ids+predicate.astype(np.uint32)
            a,b=children[pairs[:,0]],children[pairs[:,1]];mask=a!=b;a,b=a[mask],b[mask];g=edges[mask,0];w=edges[mask,2]
            gradient=np.zeros(leaves);h=np.zeros((leaves,leaves));np.add.at(gradient,a,g);np.add.at(gradient,b,-g)
            np.add.at(h,(a,a),w);np.add.at(h,(b,b),w);np.add.at(h,(a,b),-w);np.add.at(h,(b,a),-w)
            gradient=gradient.astype(np.float32);h=h.astype(np.float32);matrix=stabilized(h,False,args['l2_leaf_reg'],args['non_diagonal_regularization'])
            values=np.r_[np.linalg.solve(matrix[:-1,:-1],gradient[:-1]),0.];values-=values.mean()
            results.append((values@gradient-.5*values@h@values,values,h))
        c=int(np.argmax([v[0] for v in results]));chosen.append(c);last=results[c]
        f,b,t=(args[k][c] for k in ('candidate_features','candidate_bins','candidate_types'))
        ids|=((args['bins'][f]==b) if t else (args['bins'][f]>b)).astype(np.uint32)<<depth
    leaves=1<<len(chosen)
    if args['leaf_estimation_method']=='Simple':
        idx=np.arange(leaves);solver=2*(idx%(leaves//2))+((idx//(leaves//2))^int(t))
        values=last[1][solver].astype(np.float32);weights=np.diag(last[2])[solver]
    else:
        target=target_reference(data,**opts,fixed=True);a,b=target['pairs'].T;swap=args['targets'][a]<args['targets'][b]
        win,lose=np.where(swap,b,a),np.where(swap,a,b);weights=np.bincount(ids,weights=args['sample_weight'],minlength=leaves);values=np.zeros(leaves,np.float32)
        for _ in range(args['leaf_estimation_iterations']):
            ref=leaf_reference(np.float32(cursor+values[ids]),win,lose,target['edges'][:,2],ids,leaves,args['leaf_estimation_method'],args['l2_leaf_reg'],args['non_diagonal_regularization'])
            values=np.float32(values+ref['direction']);values[weights<=1e-20]=0;values[-1]=0
        values=np.float32(values-values.mean(dtype=float))
    return chosen,ids,np.float32(values*np.float32(args['learning_rate'])),weights


@pytest.mark.parametrize('kind,unit',[('No','Object'),('Bayesian','Object'),('Bernoulli','Object'),('Bernoulli','Group')])
@pytest.mark.parametrize('method',['Simple','Newton','Gradient'])
@pytest.mark.parametrize('score',['L2','Cosine','NewtonL2','NewtonCosine','SolarL2','LOOL2','SatL2'])
def test_complete_generated_pair_forests_match_independent_matrices(kind,unit,method,score):
    args=problem(bootstrap_type=kind,sampling_unit=unit,leaf_estimation_method=method,leaf_estimation_iterations=1 if method=='Simple' else 3,score_function=score)
    cursor=args['initial_predictions'].copy()
    with _yeti_pair.Session(**args) as session:
        for iteration in range(args['iterations']):
            chosen,ids,values,weights=oracle(args,cursor,iteration);tree=session.step()
            for attr,key in [('split_features','candidate_features'),('split_bins','candidate_bins'),('split_types','candidate_types')]:
                np.testing.assert_array_equal(getattr(tree,attr),args[key][chosen])
            np.testing.assert_array_equal(leaf_ids(tree,args['bins']),ids)
            np.testing.assert_allclose(tree.leaf_values,values,rtol=2e-4,atol=3e-6)
            np.testing.assert_allclose(tree.leaf_weights,weights,rtol=4e-5,atol=2e-6)
            cursor=np.float32(cursor+tree.leaf_values[ids]);np.testing.assert_array_equal(session.predictions(),cursor)
            assert tree.loss==0
        assert session.result().stats['gpu_seconds']>0


@pytest.mark.parametrize('method',['Simple','Newton','Gradient'])
@pytest.mark.parametrize('kind,unit',[('No','Object'),('Bayesian','Object'),('Bernoulli','Object'),('Bernoulli','Group')])
def test_resume_replays_seed_domains_and_incremental_topology_exactly(method,kind,unit):
    args=problem(iterations=4,leaf_estimation_method=method,leaf_estimation_iterations=1 if method=='Simple' else 3,bootstrap_type=kind,sampling_unit=unit)
    with _yeti_pair.Session(**args) as session:
        expected=[session.step() for _ in range(4)];final=session.predictions()
    with _yeti_pair.Session(**{**args,'iterations':2}) as session:
        for _ in range(2):session.step()
        cursor=session.predictions()
    with _yeti_pair.Session(**{**args,'iterations':2,'initial_predictions':cursor,'iteration_offset':2}) as session:
        for index in range(2,4):
            session.begin_tree()
            for _ in range(args['depth']):session.grow_tree()
            tree=session.finish_tree()
            for attr in ('split_features','split_bins','split_types','leaf_values','leaf_weights'):
                np.testing.assert_array_equal(getattr(tree,attr),getattr(expected[index],attr))
        np.testing.assert_array_equal(session.predictions(),final)


@pytest.mark.parametrize('method',['Newton','Gradient'])
@pytest.mark.parametrize('depth',[0,1,8])
def test_degenerate_topologies_and_empty_group_samples_are_valid(method,depth):
    args=problem(iterations=1,depth=depth,leaf_estimation_method=method,leaf_estimation_iterations=3,
        candidate_features=np.array([0],np.uint32),candidate_bins=np.array([1],np.uint32),candidate_types=np.array([0],np.uint8),
        bootstrap_type='Bernoulli',sampling_unit='Group',subsample=1e-8)
    expected=oracle(args,args['initial_predictions'],0)
    with _yeti_pair.Session(**args) as session:tree=session.step()
    np.testing.assert_allclose(tree.leaf_values,expected[2],rtol=2e-4,atol=3e-6)
    np.testing.assert_allclose(tree.leaf_weights,expected[3],rtol=3e-6,atol=2e-6)


@pytest.mark.parametrize('extra',[{'depth':0},{'leaf_estimation_iterations':2},{'depth':9},{'bootstrap_type':'Poisson'},
    {'bootstrap_type':'MVS'},{'leaf_estimation_backtracking':'Armijo'},{'sampling_unit':'Row'},{'permutations':True},{'decay':1.1}])
def test_invalid_session_configuration_is_rejected(extra):
    with pytest.raises(ValueError):_yeti_pair.Session(**problem(**extra))
