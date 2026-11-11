"""CUDA Plain multi-cursor semantics for deterministic full-matrix objectives."""
import platform
import numpy as np
import pytest
from catboost_metal import _pair_matrix, _query_cross_entropy
from catboost_metal._data import cuda_search_permutation
from test_pairwise_matrix_training import problem as pair_problem
from test_pairwise_matrix_kernels import reference as pair_direction
from test_query_cross_entropy_training import problem as qce_problem, leaf_walk
from test_pairwise_training import leaf_ids

pytestmark = pytest.mark.skipif(platform.system() != 'Darwin' or platform.machine() != 'arm64', reason='Apple GPU required')
CASES = [('PairLogitPairwise', method) for method in ('Simple', 'Newton', 'Gradient')] + [
    ('QueryCrossEntropy', method) for method in ('Simple', 'Newton')]


@pytest.fixture(autouse=True)
def forbid_cpu_fits(monkeypatch):
    from catboost import CatBoost
    def forbidden(*args, **kwargs): raise AssertionError('CPU CatBoost fitting is forbidden')
    monkeypatch.setattr(CatBoost, '_fit', forbidden)


def inputs(loss, method, kind='No', iterations=4):
    source = pair_problem if loss == 'PairLogitPairwise' else qce_problem
    args = source(iterations=iterations, depth=2, leaf_estimation_method=method,
        leaf_estimation_iterations=1 if method == 'Simple' else 3, bootstrap_type=kind,
        leaf_estimation_backtracking='No', subsample=.7)
    cls = _pair_matrix.Session if loss == 'PairLogitPairwise' else _query_cross_entropy.Session
    rng = np.random.default_rng(61743)
    matrices = np.array([args['bins'], *[rng.integers(0,4,args['bins'].shape,dtype=np.uint8) for _ in range(3)]])
    return args, cls, matrices


def fixed_values(loss, args, cursor, ids, leaves):
    if loss == 'QueryCrossEntropy':
        return leaf_walk(args, cursor, ids, leaves)[:2]
    values = np.zeros(leaves, np.float32)
    weights = np.bincount(ids, weights=args['sample_weight'].astype(float), minlength=leaves)
    for _ in range(args['leaf_estimation_iterations']):
        step = pair_direction(np.float32(cursor + values[ids]), args['pair_winners'], args['pair_losers'],
            args['pair_weights'], ids, leaves, args['leaf_estimation_method'],
            args['l2_leaf_reg'], args['non_diagonal_regularization'])['direction']
        values = np.float32(values + step); values[weights <= 1e-20] = 0; values[-1] = 0
    values = np.float32(values - values.mean(dtype=float))
    return np.float32(values * np.float32(args['learning_rate'])), weights


@pytest.mark.parametrize('loss,method', CASES)
@pytest.mark.parametrize('kind', ['No', 'Bernoulli'])
@pytest.mark.parametrize('schedule', ['cuda', 'all'])
def test_distinct_permutation_cursors_match_shared_simple_or_independent_leaf_estimation(loss, method, kind, schedule):
    args, cls, matrices = inputs(loss, method, kind)
    cursors = np.tile(args['initial_predictions'], (4,1))
    with cls(**args) as session:
        session.configure_permutations(matrices, initial_predictions=cursors)
        for iteration in range(4):
            selected = cuda_search_permutation(args['random_seed'], iteration, 4) if schedule == 'cuda' else [0, 3, 2, 1][iteration]
            # Search a fresh P1 target at the selected cursor. Independent
            # equations below estimate the selected structure on other banks.
            search_args = args | dict(bins=matrices[selected], initial_predictions=cursors[selected],
                iterations=1, iteration_offset=iteration)
            with cls(**search_args) as search: structure = search.step()
            session.select_permutation(selected); actual = session.step()
            for key in ('split_features', 'split_bins', 'split_types'):
                np.testing.assert_array_equal(getattr(actual,key), getattr(structure,key))
            expected_weights = None
            for permutation in range(4):
                ids = leaf_ids(structure, matrices[permutation])
                if method == 'Simple' or permutation == selected:
                    values, weights = structure.leaf_values, structure.leaf_weights
                else:
                    values, weights = fixed_values(loss,args,cursors[permutation],ids,1<<structure.depth)
                cursors[permutation] = np.float32(cursors[permutation] + values[ids])
                if permutation == 3: expected_weights = weights; expected_values = values
            state = session.permutation_state
            np.testing.assert_allclose(state['predictions'], cursors, rtol=5e-4, atol=4e-6)
            np.testing.assert_allclose(actual.leaf_values, expected_values, rtol=5e-4, atol=4e-6)
            np.testing.assert_allclose(actual.leaf_weights, expected_weights, rtol=5e-6, atol=4e-6)
            if method == 'Simple':
                np.testing.assert_array_equal(actual.leaf_values, structure.leaf_values)
                np.testing.assert_array_equal(actual.leaf_weights, structure.leaf_weights)
                np.testing.assert_array_equal(state['predictions'], cursors)


@pytest.mark.parametrize('loss,method', CASES)
@pytest.mark.parametrize('count', [1,2,4,7,64])
def test_identical_dataset_banks_match_p1_bitwise(loss, method, count):
    args, cls, _ = inputs(loss, method, iterations=3)
    with cls(**args) as one, cls(**args) as many:
        many.configure_permutations([args['bins']] * count)
        for iteration in range(3):
            many.select_permutation(count - 1 if iteration == 1 else iteration % count)
            expected = one.step(); actual = many.step()
            for key in ('split_features','split_bins','split_types','leaf_values','leaf_weights'):
                np.testing.assert_array_equal(getattr(actual,key),getattr(expected,key))
            np.testing.assert_array_equal(many.permutation_state['predictions'], np.tile(one.predictions(), (count,1)))


@pytest.mark.parametrize('loss,method', CASES)
@pytest.mark.parametrize('kind', ['No','Bernoulli'])
def test_restored_all_cursors_produce_the_same_staged_forest(loss, method, kind):
    args, cls, matrices = inputs(loss, method, kind)
    with cls(**args) as full:
        full.configure_permutations(matrices); trees=[]
        for iteration in range(4):
            full.select_permutation(cuda_search_permutation(args['random_seed'],iteration,4)); trees.append(full.step())
        final = full.permutation_state
    with cls(**(args | dict(iterations=2))) as prefix:
        prefix.configure_permutations(matrices)
        for iteration in range(2):
            prefix.select_permutation(cuda_search_permutation(args['random_seed'],iteration,4)); prefix.step()
        state = prefix.permutation_state
    with cls(**(args | dict(iterations=2,iteration_offset=2))) as resumed:
        resumed.configure_permutations(matrices,initial_predictions=state['predictions'])
        for iteration in range(2,4):
            resumed.select_permutation(cuda_search_permutation(args['random_seed'],iteration,4))
            resumed.begin_tree()
            for _ in range(args['depth']): resumed.grow_tree()
            actual=resumed.finish_tree()
            for key in ('split_features','split_bins','split_types','leaf_values','leaf_weights'):
                np.testing.assert_array_equal(getattr(actual,key),getattr(trees[iteration],key))
        np.testing.assert_array_equal(resumed.permutation_state['predictions'],final['predictions'])
