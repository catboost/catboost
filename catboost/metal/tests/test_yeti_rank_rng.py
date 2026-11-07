"""Independent MT64 interpreter checks for CUDA YetiRank orchestration draws."""
import json
import pytest
from catboost_metal._yeti_rng import YetiRankRng
from test_ordered_rng import ReferenceMt64


@pytest.mark.parametrize('seed', [0, 5489, 2**64-1])
@pytest.mark.parametrize('kind', ['No', 'Bayesian', 'Bernoulli', 'Poisson', 'MVS'])
@pytest.mark.parametrize('iterations', [1, 2, 7])
def test_host_sequence_matches_cuda_target_bootstrap_search_and_leaf_walk(seed, kind, iterations):
    source = ReferenceMt64(seed)
    source.next()  # TDocParallelBoosting::BaseIterationSeed
    actual = YetiRankRng(seed, kind, iterations)
    for tree, attempts in enumerate([0, 1, 4, 2, 16, 0]):
        assert actual.begin() == source.next()
        if tree == 0 and kind != 'No':
            source.advance(65537)
        source.advance(attempts)
        # TNewtonLikeWalker: initial evaluation then one per update, except
        # its one-iteration fast path, which never evaluates the moved point.
        expected = [source.next() for _ in range(1 if iterations == 1 else iterations + 1)]
        assert actual.leaves(attempts) == expected
        actual.complete()
        state = actual.state()
        assert state['words'] == source.words
        assert state['index'] == source.index
        assert state['completed_iterations'] == tree + 1
        assert state['bootstrap_initialized'] == (kind != 'No')
        actual = YetiRankRng(seed, kind, iterations, initial_state=json.loads(json.dumps(state)),
                             iteration_offset=tree + 1)


@pytest.mark.parametrize('field,value', [('version', True), ('random_seed', 12),
    ('leaf_iterations', 2), ('bootstrap_type', 'No'), ('words', [0]*312),
    ('words', [1]*311), ('words', [-1]*312), ('words', [2**64]*312), ('index', 313),
    ('index', True), ('bootstrap_initialized', False), ('completed_iterations', 2)])
def test_corrupt_or_incompatible_state_is_rejected(field, value):
    rng = YetiRankRng(817, 'MVS', 3)
    rng.begin(); rng.leaves(2); rng.complete()
    state = rng.state(); state[field] = value
    with pytest.raises((TypeError, ValueError)):
        YetiRankRng(817, 'MVS', 3, initial_state=state, iteration_offset=1)


def test_only_completed_iterations_can_be_serialized():
    rng = YetiRankRng(1, 'No', 1)
    with pytest.raises(RuntimeError): rng.leaves(1)
    with pytest.raises(RuntimeError): rng.complete()
    rng.begin()
    with pytest.raises(RuntimeError): rng.begin()
    with pytest.raises(RuntimeError): rng.state()
    rng.leaves(1)
    with pytest.raises(RuntimeError): rng.state()
    with pytest.raises(RuntimeError): rng.leaves(1)
    rng.complete()
    assert rng.state()['completed_iterations'] == 1


def test_nonzero_offset_cannot_invent_missing_rng_history():
    with pytest.raises(ValueError, match='saved RNG'):
        YetiRankRng(1, 'No', 1, iteration_offset=1)
