"""Independent per-cursor YetiRank equations and CUDA host seed ordering."""
import json
import platform
import numpy as np
import pytest
from catboost_metal import _yeti
from catboost_metal._data import cuda_search_permutation
from catboost_metal._yeti_rng import YetiRankRng
from test_ordered_rng import ReferenceMt64
from test_yeti_rank_training import problem, oracle_leaves
from test_pairwise_training import leaf_ids

pytestmark = pytest.mark.skipif(platform.system() != 'Darwin' or platform.machine() != 'arm64', reason='Apple GPU required')


@pytest.fixture(autouse=True)
def no_cpu_fit(monkeypatch):
    from catboost import CatBoost
    def forbidden(*args, **kwargs): raise AssertionError('CPU CatBoost fitting is forbidden')
    monkeypatch.setattr(CatBoost, '_fit', forbidden)


def inputs(kind='No', leaf_iterations=3, count=4, **extra):
    args = problem(random_seed=817, iterations=4, bootstrap_type=kind,
                   leaf_estimation_iterations=leaf_iterations, **extra)
    if kind in ('Bernoulli', 'Poisson', 'MVS'): args['subsample'] = .7
    rng = np.random.default_rng(44173)
    banks = np.array([args['bins'], *[rng.integers(0,4,args['bins'].shape,dtype=np.uint8) for _ in range(count-1)]])
    return args, banks


@pytest.mark.parametrize('kind', ['No','Bayesian','Bernoulli','Poisson','MVS'])
@pytest.mark.parametrize('leaf_iterations', [1,3])
@pytest.mark.parametrize('schedule', ['cuda','all'])
def test_distinct_dataset_leaf_walks_match_independent_oracles(kind, leaf_iterations, schedule):
    args, banks = inputs(kind, leaf_iterations)
    cursors = np.tile(args['initial_predictions'], (4,1)); lambdas = [None]*4
    chunk = leaf_iterations + int(leaf_iterations > 1)
    with _yeti.Session(**args) as session:
        session.configure_permutations(banks)
        for iteration in range(4):
            selected = cuda_search_permutation(817, iteration, 4) if schedule == 'cuda' else [3,2,0,1][iteration]
            seeds = [0xdef0123400000011 + iteration*97 + j for j in range(1+4*chunk)]
            search_args = args | dict(bins=banks[selected], initial_predictions=cursors[selected],
                iterations=1, iteration_offset=iteration, initial_mvs_lambda=lambdas[selected])
            with _yeti.Session(**search_args) as search:
                structure = search.step([seeds[0], *seeds[1+selected*chunk:1+(selected+1)*chunk]])
            session.select_permutation(selected); actual = session.step(seeds)
            for key in ('split_features','split_bins','split_types'):
                np.testing.assert_array_equal(getattr(actual,key),getattr(structure,key))
            for p in range(4):
                ids = leaf_ids(structure, banks[p])
                values, weights = oracle_leaves(args, cursors[p], ids, 1<<structure.depth,
                    seeds[1+p*chunk:1+p*chunk+leaf_iterations])
                cursors[p] = np.float32(cursors[p]+values[ids])
                if kind == 'MVS': lambdas[p] = np.float32(np.abs(values.astype(float)).mean()**2)
                if p == 3:
                    np.testing.assert_allclose(actual.leaf_values,values,rtol=2e-4,atol=2e-6)
                    np.testing.assert_allclose(actual.leaf_weights,weights,rtol=5e-6,atol=1e-5)
            np.testing.assert_allclose(session.permutation_state['predictions'],cursors,rtol=3e-4,atol=3e-6)


@pytest.mark.parametrize('kind', ['No','Bayesian','Bernoulli','Poisson','MVS'])
@pytest.mark.parametrize('leaf_iterations', [1,3])
@pytest.mark.parametrize('count', [1,2,4,7,64])
def test_host_rng_preserves_complete_leaf_walks_in_dataset_order(kind, leaf_iterations, count):
    rng = YetiRankRng(817,kind,leaf_iterations,dataset_permutations=count)
    source = ReferenceMt64(817); source.next()
    for iteration, attempts in enumerate([0,2,1]):
        assert rng.begin() == source.next()
        if not iteration and kind != 'No': source.advance(65537)
        source.advance(attempts)
        expected = [source.next() for p in range(count) for _ in range(leaf_iterations+int(leaf_iterations>1))]
        assert rng.leaves(attempts) == expected
        rng.complete(); state = json.loads(json.dumps(rng.state()))
        assert state['words'] == source.words and state['index'] == source.index
        rng = YetiRankRng(817,kind,leaf_iterations,dataset_permutations=count,
            iteration_offset=iteration+1,initial_state=state)


@pytest.mark.parametrize('kind', ['No','Bayesian','Bernoulli','Poisson','MVS'])
@pytest.mark.parametrize('leaf_iterations', [1,3])
def test_automatic_controller_restores_all_cursors_and_rng_bitwise(kind, leaf_iterations):
    args, banks = inputs(kind, leaf_iterations, random_strength=.7)
    with _yeti.TrainingSession(**args, dataset_permutations=4) as full:
        full.configure_permutations(banks); trees = [full.step() for _ in range(4)]
        final = full.permutation_state; final_rng = full.rng.state()
    with _yeti.TrainingSession(**(args | dict(iterations=2)), dataset_permutations=4) as partial:
        partial.configure_permutations(banks); partial.step(); partial.step()
        state = partial.permutation_state; rng_state = partial.rng.state()
    with _yeti.TrainingSession(**(args | dict(iterations=2,iteration_offset=2)),
            dataset_permutations=4,initial_rng_state=rng_state) as resumed:
        resumed.configure_permutations(banks,initial_predictions=state['predictions'],
            mvs_lambdas=state['mvs_lambdas'],mvs_valid=state['mvs_valid'])
        for i in (2,3):
            actual = resumed.step()
            for key in ('split_features','split_bins','split_types','leaf_values','leaf_weights'):
                np.testing.assert_array_equal(getattr(actual,key),getattr(trees[i],key))
            assert actual.loss == trees[i].loss
        for key in final: np.testing.assert_array_equal(resumed.permutation_state[key],final[key])
        assert resumed.rng.state() == final_rng


@pytest.mark.parametrize('count', [1,2,4,7,64])
@pytest.mark.parametrize('depth,empty', [(0,False),(2,True),(2,False)])
def test_identical_banks_with_identical_seed_chunks_match_p1_staged(count, depth, empty):
    args, _ = inputs(depth=depth)
    if empty: args.update(candidate_features=np.empty(0,np.uint32),candidate_bins=np.empty(0,np.uint32))
    with _yeti.Session(**args) as one, _yeti.Session(**args) as many:
        many.configure_permutations([args['bins']]*count)
        for iteration in range(4):
            seeds = [100+iteration*17+j for j in range(5)]
            expected = one.step(seeds)
            many.select_permutation(count-1 if iteration%2 else 0)
            many.begin_tree([seeds[0]])
            while not many.grow_tree()['finished']: pass
            actual = many.finish_tree(seeds[1:]*count)
            for key in ('split_features','split_bins','split_types','leaf_values','leaf_weights'):
                np.testing.assert_array_equal(getattr(actual,key),getattr(expected,key))
            np.testing.assert_array_equal(many.permutation_state['predictions'],np.tile(one.predictions(),(count,1)))


def test_missing_banks_or_wrong_rng_geometry_fail_before_rng_is_consumed():
    args, banks = inputs()
    with _yeti.TrainingSession(**args,dataset_permutations=4) as session:
        state = session.rng.state()
        with pytest.raises(ValueError,match='every'): session.step()
        with pytest.raises(ValueError,match='match'): session.configure_permutations(banks[:2])
        assert session.rng.state() == state
        session.configure_permutations(banks); session.step()
        state = session.rng.state()
        for count in (1,2,7):
            with pytest.raises(ValueError,match='dataset_permutations'):
                YetiRankRng(817,'No',3,initial_state=state,iteration_offset=1,dataset_permutations=count)


def test_permutation_geometry_cannot_change_after_supplying_seeds():
    args, banks = inputs()
    with _yeti.Session(**args) as session:
        session.set_oracle_seeds([1,2,3,4,5])
        with pytest.raises(RuntimeError,match='before supplying'): session.configure_permutations(banks)
        session.step([1,2,3,4,5])
