"""Classic YetiRank's single-device FeatureParallel Ordered host draws.

The chooser, weak query oracles, bootstrap cache, split searches and leaf
oracles share CUDA's host stream. Metal GPU bootstrap/noise streams retain
their existing adaptation. Seed counts come from the runtime's actual folds.
"""

import numpy as np

from ._data import _CudaMersenne64, _unsigned_integer


class OrderedYetiRankRng:
    def __init__(self, seed, permutations, leaf_iterations, *, score_sets=1,
                 initial_state=None, iteration_offset=0):
        self.seed = _unsigned_integer(seed, "random_seed", (1 << 64) - 1)
        self.permutations = _unsigned_integer(permutations, "permutation_count", 64)
        self.leaf_iterations = _unsigned_integer(leaf_iterations, "leaf_iterations", 1000)
        self.score_sets = _unsigned_integer(score_sets, "score_sets", 2)
        if not self.permutations or not self.leaf_iterations or not self.score_sets:
            raise ValueError("Ordered YetiRank RNG dimensions must be positive.")
        if self.score_sets > 1 and self.permutations == 1:
            raise ValueError("Dependent simple CTR score draws require multiple permutations.")
        self.completed_iterations = _unsigned_integer(iteration_offset, "iteration_offset", (1 << 32) - 1)
        self.random = _CudaMersenne64(self.seed)
        self.bootstrap_initialized = False
        self._phase = "idle"
        if initial_state is None:
            if self.completed_iterations:
                raise ValueError("Ordered YetiRank continuation requires its saved selection_rng state.")
            return
        state = initial_state
        if not isinstance(state, dict) or state.get("policy") != "ordered_yeti_host_v1":
            raise ValueError("Invalid Ordered YetiRank selection RNG policy.")
        for name, expected in (("version", 1), ("random_seed", self.seed),
                               ("permutation_count", self.permutations), ("leaf_iterations", self.leaf_iterations),
                               ("score_sets", self.score_sets), ("completed_iterations", self.completed_iterations)):
            value = _unsigned_integer(state.get(name), "Ordered YetiRank RNG " + name, (1 << 64) - 1)
            if value != expected:
                raise ValueError("Ordered YetiRank RNG state conflicts with " + name + ".")
        words = state.get("words")
        if not isinstance(words, (list, tuple, np.ndarray)) or len(words) != 312:
            raise ValueError("Ordered YetiRank RNG words must contain 312 uint64 integers.")
        words = [_unsigned_integer(word, "Ordered YetiRank RNG word", (1 << 64) - 1) for word in words]
        if not any(words):
            raise ValueError("Ordered YetiRank RNG words cannot all be zero.")
        index = _unsigned_integer(state.get("index"), "Ordered YetiRank RNG index", 312)
        initialized = state.get("bootstrap_initialized")
        if not isinstance(initialized, bool) or initialized != (self.completed_iterations > 0):
            raise ValueError("Invalid Ordered YetiRank RNG bootstrap initialization state.")
        self.random.state, self.random.index = words, index
        self.bootstrap_initialized = initialized

    def select(self):
        if self._phase != "idle":
            raise RuntimeError("Finish the pending Ordered YetiRank iteration before selecting again.")
        if self.completed_iterations >= (1 << 32) - 1:
            raise ValueError("Ordered YetiRank RNG iteration_offset cannot exceed uint32.")
        self._phase = "selected"
        learn_count = self.permutations - 1 if self.permutations > 1 else 1
        return self.random.next() % (learn_count - 1) if learn_count > 1 else 0

    def weak(self, seed_count):
        count = _unsigned_integer(seed_count, "Ordered YetiRank weak seed count", (1 << 32) - 1)
        if not count or count % 2:
            raise ValueError("Ordered YetiRank needs a learn and quality seed for every selected fold.")
        if self._phase != "selected":
            raise RuntimeError("Select the Ordered YetiRank permutation before its weak seeds.")
        self._phase = "search"
        return [self.random.next() for _ in range(count)]

    def leaves(self, search_attempts, seed_count):
        attempts = _unsigned_integer(search_attempts, "Ordered YetiRank search_attempts", 16)
        count = _unsigned_integer(seed_count, "Ordered YetiRank leaf seed count", (1 << 32) - 1)
        evaluations = self.leaf_iterations + int(self.leaf_iterations > 1)
        if not count or count % evaluations:
            raise ValueError("Ordered YetiRank leaf seed count must cover every evaluation and active task.")
        if self._phase != "search":
            raise RuntimeError("Begin Ordered YetiRank search before requesting leaf seeds.")
        # FeatureParallel creates its MirrorMapping seed cache even for No.
        if not self.bootstrap_initialized:
            self.random.advance(65537)
            self.bootstrap_initialized = True
        self.random.advance(attempts * self.score_sets)
        self._phase = "leaves"
        # Runtime consumes this evaluation-major, then task-major packet.
        # The last evaluation when I>1 is unused for movement but still drawn.
        return [self.random.next() for _ in range(count)]

    def finish(self):
        if self._phase != "leaves":
            raise RuntimeError("Supply Ordered YetiRank leaf seeds before completing the iteration.")
        self.completed_iterations += 1
        self._phase = "idle"

    def state(self):
        if self._phase != "idle":
            raise RuntimeError("Cannot snapshot an incomplete Ordered YetiRank iteration.")
        return {"version": 1, "policy": "ordered_yeti_host_v1", "random_seed": self.seed,
                "permutation_count": self.permutations, "leaf_iterations": self.leaf_iterations,
                "score_sets": self.score_sets, "words": list(self.random.state), "index": self.random.index,
                "bootstrap_initialized": self.bootstrap_initialized,
                "completed_iterations": self.completed_iterations}
