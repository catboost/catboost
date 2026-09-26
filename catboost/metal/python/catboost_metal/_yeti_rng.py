"""Classic YetiRank's single-device numeric DocParallel host seed protocol.

The tree learner uses separately documented Metal bootstrap/noise streams.
These draws preserve CUDA's stochastic target stream, including unused draws.
"""
import numpy as np
from ._data import _CudaMersenne64, _unsigned_integer


class YetiRankRng:
    def __init__(self, seed, bootstrap_type, leaf_iterations, *, initial_state=None, iteration_offset=0,
                 dataset_permutations=1):
        self.seed = _unsigned_integer(seed, 'random_seed', (1 << 64) - 1)
        if bootstrap_type not in ('No', 'Bayesian', 'Bernoulli', 'Poisson', 'MVS'):
            raise ValueError('Invalid YetiRank bootstrap type.')
        self.bootstrap_type = bootstrap_type
        self.leaf_iterations = _unsigned_integer(leaf_iterations, 'leaf_iterations', 1000)
        if not self.leaf_iterations:
            raise ValueError('YetiRank leaf_iterations must be positive.')
        self.dataset_permutations = _unsigned_integer(dataset_permutations, 'dataset_permutations', 64)
        if not self.dataset_permutations:
            raise ValueError('YetiRank dataset_permutations must be positive.')
        self.completed_iterations = _unsigned_integer(iteration_offset, 'iteration_offset', (1 << 32) - 1)
        self.random = _CudaMersenne64(self.seed)
        self.random.next()  # TDocParallelBoosting constructor's BaseIterationSeed.
        self.bootstrap_initialized = False
        self._phase = 'idle'
        if initial_state is None:
            if self.completed_iterations:
                raise ValueError('YetiRank continuation requires its saved RNG state.')
            return
        state = initial_state
        if not isinstance(state, dict):
            raise ValueError('Invalid YetiRank RNG state.')
        for name, expected in (('version', 1), ('random_seed', self.seed),
                               ('leaf_iterations', self.leaf_iterations),
                               ('completed_iterations', self.completed_iterations)):
            actual = _unsigned_integer(state.get(name), 'YetiRank RNG ' + name, (1 << 64) - 1)
            if actual != expected:
                raise ValueError('YetiRank RNG state conflicts with ' + name + '.')
        if state.get('bootstrap_type') != bootstrap_type:
            raise ValueError('YetiRank RNG state conflicts with bootstrap_type.')
        if _unsigned_integer(state.get('dataset_permutations', 1), 'dataset_permutations', 64) != self.dataset_permutations:
            raise ValueError('YetiRank RNG state conflicts with dataset_permutations.')
        words = state.get('words')
        if not isinstance(words, (list, tuple, np.ndarray)) or len(words) != 312:
            raise ValueError('YetiRank RNG needs 312 uint64 words.')
        words = [_unsigned_integer(v, 'YetiRank RNG word', (1 << 64) - 1) for v in words]
        if not any(words):
            raise ValueError('YetiRank RNG words cannot all be zero.')
        index = _unsigned_integer(state.get('index'), 'YetiRank RNG index', 312)
        initialized = state.get('bootstrap_initialized')
        if not isinstance(initialized, bool) or initialized != (self.completed_iterations > 0 and bootstrap_type != 'No'):
            raise ValueError('Invalid YetiRank RNG bootstrap initialization state.')
        self.random.state, self.random.index = words, index
        self.bootstrap_initialized = initialized

    def begin(self):
        if self._phase != 'idle':
            raise RuntimeError('Complete the pending YetiRank RNG iteration first.')
        self._phase = 'search'
        return self.random.next()

    def leaves(self, search_attempts):
        attempts = _unsigned_integer(search_attempts, 'YetiRank search_attempts', 16)
        if self._phase != 'search':
            raise RuntimeError('Begin YetiRank search before requesting leaf seeds.')
        # BootstrapAndFilter skips GetGpuSeeds entirely for No. Otherwise the
        # first StripeMapping cache consumes its base draw plus 65536 FillSeeds.
        if self.bootstrap_type != 'No' and not self.bootstrap_initialized:
            self.random.advance(65537)
            self.bootstrap_initialized = True
        self.random.advance(attempts)  # One numeric feature-set seed per attempt.
        count = self.dataset_permutations * (self.leaf_iterations + int(self.leaf_iterations > 1))
        values = [self.random.next() for _ in range(count)]
        self._phase = 'leaves'
        return values

    def complete(self):
        if self._phase != 'leaves':
            raise RuntimeError('Supply YetiRank leaf seeds before completing the iteration.')
        self.completed_iterations += 1
        self._phase = 'idle'

    def state(self):
        if self._phase != 'idle':
            raise RuntimeError('Cannot snapshot an incomplete YetiRank iteration.')
        result = dict(version=1, random_seed=self.seed, bootstrap_type=self.bootstrap_type,
                    leaf_iterations=self.leaf_iterations, words=list(self.random.state),
                    index=self.random.index, bootstrap_initialized=self.bootstrap_initialized,
                    completed_iterations=self.completed_iterations)
        if self.dataset_permutations != 1:
            result['dataset_permutations'] = self.dataset_permutations
        return result
