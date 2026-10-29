"""Per-dataset generated PFound targets, fixed matrices and exact continuation."""
import platform
import numpy as np
import pytest
from catboost_metal import _yeti_pair
from catboost_metal._data import cuda_search_permutation
from test_yeti_pair_training import problem, oracle
from test_pfound_pair_runtime import target_reference
from test_pairwise_matrix_kernels import reference as leaf_reference
from test_pairwise_training import leaf_ids

pytestmark = pytest.mark.skipif(platform.system()!='Darwin' or platform.machine()!='arm64',reason='Apple GPU required')
SAMPLERS = [('No','Object'),('Bayesian','Object'),('Bernoulli','Object'),('Bernoulli','Group')]
METHODS = ['Simple','Newton','Gradient']


@pytest.fixture(autouse=True)
def no_cpu_fit(monkeypatch):
    from catboost import CatBoost
    def forbidden(*args,**kwargs):raise AssertionError('CPU CatBoost fitting is forbidden')
    monkeypatch.setattr(CatBoost,'_fit',forbidden)


def inputs(method,kind='No',unit='Object',count=4,**extra):
    args=problem(iterations=4,depth=2,leaf_estimation_method=method,
        leaf_estimation_iterations=1 if method=='Simple' else 3,bootstrap_type=kind,sampling_unit=unit)
    args.update(extra);rng=np.random.default_rng(65431)
    banks=np.array([args['bins'],*[rng.integers(0,4,args['bins'].shape,dtype=np.uint8) for _ in range(count-1)]])
    return args,banks


def fixed_leaves(args,cursor,ids,leaves,iteration,dataset):
    data=(args['bins'],args['targets'],args['sample_weight'],cursor,args['group_offsets'])
    target=target_reference(data,seed=args['random_seed'],absolute=iteration,permutations=args['permutations'],fixed=True,dataset=dataset)
    a,b=target['pairs'].T;swap=args['targets'][a]<args['targets'][b]
    win,lose=np.where(swap,b,a),np.where(swap,a,b)
    weights=np.bincount(ids,weights=args['sample_weight'],minlength=leaves)
    values=np.zeros(leaves,np.float32)
    for _ in range(args['leaf_estimation_iterations']):
        step=leaf_reference(np.float32(cursor+values[ids]),win,lose,target['edges'][:,2],ids,leaves,
            args['leaf_estimation_method'],args['l2_leaf_reg'],args['non_diagonal_regularization'])['direction']
        values=np.float32(values+step);values[weights<=1e-20]=0;values[-1]=0
    values=np.float32(values-values.mean(dtype=float))
    return np.float32(values*np.float32(args['learning_rate'])),weights


@pytest.mark.parametrize('method',METHODS)
@pytest.mark.parametrize('kind,unit',SAMPLERS)
@pytest.mark.parametrize('schedule',['cuda','all'])
def test_distinct_cursors_use_selected_weak_target_and_independent_fixed_pairs(method,kind,unit,schedule):
    args,banks=inputs(method,kind,unit);cursors=np.tile(args['initial_predictions'],(4,1))
    with _yeti_pair.Session(**args) as session:
        session.configure_permutations(banks)
        for iteration in range(4):
            selected=cuda_search_permutation(args['random_seed'],iteration,4) if schedule=='cuda' else [3,2,0,1][iteration]
            chosen,_,weak_values,weak_weights=oracle(args|dict(bins=banks[selected]),cursors[selected],iteration,dataset=selected)
            session.select_permutation(selected);tree=session.step()
            for attr,key in [('split_features','candidate_features'),('split_bins','candidate_bins'),('split_types','candidate_types')]:
                np.testing.assert_array_equal(getattr(tree,attr),args[key][chosen])
            for p in range(4):
                ids=leaf_ids(tree,banks[p])
                if method=='Simple':values,weights=weak_values,weak_weights
                else:values,weights=fixed_leaves(args,cursors[p],ids,1<<tree.depth,iteration,p)
                cursors[p]=np.float32(cursors[p]+values[ids])
                if p==3:
                    np.testing.assert_allclose(tree.leaf_values,values,rtol=5e-4,atol=5e-6)
                    np.testing.assert_allclose(tree.leaf_weights,weights,rtol=4e-5,atol=4e-6)
            np.testing.assert_allclose(session.permutation_state['predictions'],cursors,rtol=5e-4,atol=8e-6)


@pytest.mark.parametrize('method',METHODS)
@pytest.mark.parametrize('kind,unit',SAMPLERS)
def test_automatic_driver_recovers_every_cursor_and_metric_bitwise(method,kind,unit):
    args,banks=inputs(method,kind,unit)
    with _yeti_pair.TrainingSession(**args) as full:
        full.configure_permutations(banks);trees=[full.step() for _ in range(4)];final=full.permutation_state
        assert full.result().stats['yeti_pair_rng']=='item_iteration_dataset_domains_v2'
    with _yeti_pair.TrainingSession(**(args|dict(iterations=2))) as partial:
        partial.configure_permutations(banks);partial.step();partial.step();state=partial.permutation_state
    with _yeti_pair.TrainingSession(**(args|dict(iterations=2,iteration_offset=2))) as resumed:
        resumed.configure_permutations(banks,initial_predictions=state['predictions'])
        for i in (2,3):
            tree=resumed.step()
            for attr in ('split_features','split_bins','split_types','leaf_values','leaf_weights'):
                np.testing.assert_array_equal(getattr(tree,attr),getattr(trees[i],attr))
            assert tree.loss==trees[i].loss
        for key in final:np.testing.assert_array_equal(resumed.permutation_state[key],final[key])


@pytest.mark.parametrize('method',METHODS)
@pytest.mark.parametrize('count',[1,2,4,7,64])
def test_identical_banks_reuse_simple_model_or_estimate_distinct_fixed_targets(method,count):
    args,_=inputs(method,iterations=2,depth=1)
    cursors=np.tile(args['initial_predictions'],(count,1))
    with _yeti_pair.Session(**args) as many:
        many.configure_permutations([args['bins']]*count)
        for iteration in range(2):
            chosen,_,values,weights=oracle(args,cursors[0],iteration)
            actual=many.step()
            for attr,key in [('split_features','candidate_features'),('split_bins','candidate_bins'),('split_types','candidate_types')]:
                np.testing.assert_array_equal(getattr(actual,attr),args[key][chosen])
            ids=leaf_ids(actual,args['bins'])
            for p in range(count):
                if method=='Simple':value,weight=actual.leaf_values,actual.leaf_weights
                else:value,weight=fixed_leaves(args,cursors[p],ids,1<<actual.depth,iteration,p)
                cursors[p]=np.float32(cursors[p]+value[ids])
                if p==count-1:
                    np.testing.assert_allclose(actual.leaf_values,value,rtol=5e-4,atol=5e-6)
                    np.testing.assert_allclose(actual.leaf_weights,weight,rtol=4e-5,atol=4e-6)
            np.testing.assert_allclose(many.permutation_state['predictions'],cursors,rtol=5e-4,atol=8e-6)
            if method=='Simple':
                np.testing.assert_allclose(actual.leaf_values,values,rtol=5e-4,atol=5e-6)
                np.testing.assert_array_equal(many.permutation_state['predictions'],cursors)
            elif count>1:assert not np.array_equal(cursors[0],cursors[-1])


@pytest.mark.parametrize('method',['Newton','Gradient'])
@pytest.mark.parametrize('empty',[False,True])
def test_empty_structure_and_zero_sampled_queries_keep_all_cursors_valid(method,empty):
    args,banks=inputs(method,'Bernoulli','Group',depth=0 if empty else 2,subsample=1e-8)
    with _yeti_pair.Session(**args) as session:
        session.configure_permutations(banks);session.select_permutation(3)
        session.begin_tree()
        while not session.grow_tree()['finished']:pass
        tree=session.finish_tree()
        assert np.isfinite(tree.leaf_values).all()
        for p,cursor in enumerate(session.permutation_state['predictions']):
            ids=leaf_ids(tree,banks[p]);value,_=fixed_leaves(args,args['initial_predictions'],ids,1<<tree.depth,0,p)
            np.testing.assert_allclose(cursor,args['initial_predictions']+value[ids],rtol=5e-4,atol=8e-6)
