"""Classic YetiRank target draws for variable-shape greedy tree searches."""

from ._data import _unsigned_integer
from ._yeti_rng import YetiRankRng


class GreedyYetiRankRng(YetiRankRng):
    """Persist the same target stream while counting actual greedy searches."""

    def __init__(self, *args, initial_state=None, **kwargs):
        if initial_state is not None and (
                not isinstance(initial_state, dict) or initial_state.get("learner") != "greedy_v1"):
            raise ValueError("Greedy YetiRank continuation requires its saved greedy RNG state.")
        super().__init__(*args, initial_state=initial_state, **kwargs)

    def leaves(self, search_attempts):
        # A Lossguide/Region tree can issue more than sixteen search calls.
        # Count actual runtime calls, including terminal unsuccessful searches.
        attempts = _unsigned_integer(search_attempts, "YetiRank search_attempts", (1 << 32) - 1)
        if self._phase != "search":
            raise RuntimeError("Begin YetiRank search before requesting leaf seeds.")
        if self.bootstrap_type != "No" and not self.bootstrap_initialized:
            self.random.advance(65537)
            self.bootstrap_initialized = True
        self.random.advance(attempts)
        count = self.dataset_permutations * (self.leaf_iterations + int(self.leaf_iterations > 1))
        values = [self.random.next() for _ in range(count)]
        self._phase = "leaves"
        return values

    def state(self):
        return {**super().state(), "learner": "greedy_v1"}
