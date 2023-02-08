# coding=utf-8
#
# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Most of this work is copyright (C) 2013-2019 David R. MacIver
# (david@drmaciver.com), but it contains contributions by others. See
# CONTRIBUTING.rst for a full list of people who may hold copyright, and
# consult the git log if you need to determine who owns an individual
# contribution.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.
#
# END HEADER

from __future__ import absolute_import, division, print_function

import sys
import warnings
from collections import defaultdict
from random import choice as random_choice

import hypothesis.internal.conjecture.utils as cu
from hypothesis._settings import (
    HealthCheck,
    Phase,
    Verbosity,
    not_set,
    note_deprecation,
    settings,
)
from hypothesis.control import _current_build_context, assume
from hypothesis.errors import (
    HypothesisException,
    NonInteractiveExampleWarning,
    UnsatisfiedAssumption,
)
from hypothesis.internal.compat import bit_length, hrange
from hypothesis.internal.conjecture.utils import (
    calc_label_from_cls,
    calc_label_from_name,
    combine_labels,
)
from hypothesis.internal.coverage import check_function
from hypothesis.internal.lazyformat import lazyformat
from hypothesis.internal.reflection import get_pretty_function_description
from hypothesis.internal.validation import check_type
from hypothesis.utils.conventions import UniqueIdentifier

try:
    from typing import Any, List, Callable, TypeVar, Generic, Optional  # noqa

    Ex = TypeVar("Ex", covariant=True)
    T = TypeVar("T")

    from hypothesis._settings import UniqueIdentifier  # noqa
    from hypothesis.internal.conjecture.data import ConjectureData  # noqa

except ImportError:  # pragma: no cover
    Ex = "key"  # type: ignore
    Generic = {Ex: object}  # type: ignore

calculating = UniqueIdentifier("calculating")

MAPPED_SEARCH_STRATEGY_DO_DRAW_LABEL = calc_label_from_name(
    "another attempted draw in MappedSearchStrategy"
)


def one_of_strategies(xs):
    """Helper function for unioning multiple strategies."""
    xs = tuple(xs)
    if not xs:
        raise ValueError("Cannot join an empty list of strategies")
    return OneOfStrategy(xs)


def recursive_property(name, default):
    """Handle properties which may be mutually recursive among a set of
    strategies.

    These are essentially lazily cached properties, with the ability to set
    an override: If the property has not been explicitly set, we calculate
    it on first access and memoize the result for later.

    The problem is that for properties that depend on each other, a naive
    calculation strategy may hit infinite recursion. Consider for example
    the property is_empty. A strategy defined as x = st.deferred(lambda: x)
    is certainly empty (in order to draw a value from x we would have to
    draw a value from x, for which we would have to draw a value from x,
    ...), but in order to calculate it the naive approach would end up
    calling x.is_empty in order to calculate x.is_empty in order to etc.

    The solution is one of fixed point calculation. We start with a default
    value that is the value of the property in the absence of evidence to
    the contrary, and then update the values of the property for all
    dependent strategies until we reach a fixed point.

    The approach taken roughly follows that in section 4.2 of Adams,
    Michael D., Celeste Hollenbeck, and Matthew Might. "On the complexity
    and performance of parsing with derivatives." ACM SIGPLAN Notices 51.6
    (2016): 224-236.
    """
    cache_key = "cached_" + name
    calculation = "calc_" + name
    force_key = "force_" + name

    def forced_value(target):
        try:
            return getattr(target, force_key)
        except AttributeError:
            return getattr(target, cache_key)

    def accept(self):
        try:
            return forced_value(self)
        except AttributeError:
            pass

        mapping = {}
        sentinel = object()
        hit_recursion = [False]

        # For a first pass we do a direct recursive calculation of the
        # property, but we block recursively visiting a value in the
        # computation of its property: When that happens, we simply
        # note that it happened and return the default value.
        def recur(strat):
            try:
                return forced_value(strat)
            except AttributeError:
                pass
            result = mapping.get(strat, sentinel)
            if result is calculating:
                hit_recursion[0] = True
                return default
            elif result is sentinel:
                mapping[strat] = calculating
                mapping[strat] = getattr(strat, calculation)(recur)
                return mapping[strat]
            return result

        recur(self)

        # If we hit self-recursion in the computation of any strategy
        # value, our mapping at the end is imprecise - it may or may
        # not have the right values in it. We now need to proceed with
        # a more careful fixed point calculation to get the exact
        # values. Hopefully our mapping is still pretty good and it
        # won't take a large number of updates to reach a fixed point.
        if hit_recursion[0]:
            needs_update = set(mapping)

            # We track which strategies use which in the course of
            # calculating their property value. If A ever uses B in
            # the course of calculating its value, then whenever the
            # value of B changes we might need to update the value of
            # A.
            listeners = defaultdict(set)
        else:
            needs_update = None

        def recur2(strat):
            def recur_inner(other):
                try:
                    return forced_value(other)
                except AttributeError:
                    pass
                listeners[other].add(strat)
                result = mapping.get(other, sentinel)
                if result is sentinel:
                    needs_update.add(other)
                    mapping[other] = default
                    return default
                return result

            return recur_inner

        count = 0
        seen = set()
        while needs_update:
            count += 1
            # If we seem to be taking a really long time to stabilize we
            # start tracking seen values to attempt to detect an infinite
            # loop. This should be impossible, and most code will never
            # hit the count, but having an assertion for it means that
            # testing is easier to debug and we don't just have a hung
            # test.
            # Note: This is actually covered, by test_very_deep_deferral
            # in tests/cover/test_deferred_strategies.py. Unfortunately it
            # runs into a coverage bug. See
            # https://bitbucket.org/ned/coveragepy/issues/605/
            # for details.
            if count > 50:  # pragma: no cover
                key = frozenset(mapping.items())
                assert key not in seen, (key, name)
                seen.add(key)
            to_update = needs_update
            needs_update = set()
            for strat in to_update:
                new_value = getattr(strat, calculation)(recur2(strat))
                if new_value != mapping[strat]:
                    needs_update.update(listeners[strat])
                    mapping[strat] = new_value

        # We now have a complete and accurate calculation of the
        # property values for everything we have seen in the course of
        # running this calculation. We simultaneously update all of
        # them (not just the strategy we started out with).
        for k, v in mapping.items():
            setattr(k, cache_key, v)
        return getattr(self, cache_key)

    accept.__name__ = name
    return property(accept)


class SearchStrategy(Generic[Ex]):
    """A SearchStrategy is an object that knows how to explore data of a given
    type.

    Except where noted otherwise, methods on this class are not part of
    the public API and their behaviour may change significantly between
    minor version releases. They will generally be stable between patch
    releases.
    """

    supports_find = True
    validate_called = False
    __label = None

    def available(self, data):
        """Returns whether this strategy can *currently* draw any
        values. This typically useful for stateful testing where ``Bundle``
        grows over time a list of value to choose from.

        Unlike ``empty`` property, this method's return value may change
        over time.
        Note: ``data`` parameter will only be used for introspection and no
        value drawn from it.
        """
        return not self.is_empty

    # Returns True if this strategy can never draw a value and will always
    # result in the data being marked invalid.
    # The fact that this returns False does not guarantee that a valid value
    # can be drawn - this is not intended to be perfect, and is primarily
    # intended to be an optimisation for some cases.
    is_empty = recursive_property("is_empty", True)

    # Returns True if values from this strategy can safely be reused without
    # this causing unexpected behaviour.
    has_reusable_values = recursive_property("has_reusable_values", True)

    # Whether this strategy is suitable for holding onto in a cache.
    is_cacheable = recursive_property("is_cacheable", True)

    def calc_is_cacheable(self, recur):
        return True

    def calc_is_empty(self, recur):
        # Note: It is correct and significant that the default return value
        # from calc_is_empty is False despite the default value for is_empty
        # being true. The reason for this is that strategies should be treated
        # as empty absent evidence to the contrary, but most basic strategies
        # are trivially non-empty and it would be annoying to have to override
        # this method to show that.
        return False

    def calc_has_reusable_values(self, recur):
        return False

    def example(self, random=not_set):
        # type: (UniqueIdentifier) -> Ex
        """Provide an example of the sort of value that this strategy
        generates. This is biased to be slightly simpler than is typical for
        values from this strategy, for clarity purposes.

        This method shouldn't be taken too seriously. It's here for interactive
        exploration of the API, not for any sort of real testing.

        This method is part of the public API.
        """
        if random is not not_set:
            note_deprecation("The random argument does nothing", since="2019-07-08")

        if getattr(sys, "ps1", None) is None:  # pragma: no branch
            # The other branch *is* covered in cover/test_examples.py; but as that
            # uses `pexpect` for an interactive session `coverage` doesn't see it.
            warnings.warn(
                "The `.example()` method is good for exploring strategies, but should "
                "only be used interactively.  We recommend using `@given` for tests - "
                "it performs better, saves and replays failures to avoid flakiness, "
                "and reports minimal examples. (strategy: %r)" % (self,),
                NonInteractiveExampleWarning,
            )

        context = _current_build_context.value
        if context is not None:
            if context.data is not None and context.data.depth > 0:
                raise HypothesisException(
                    "Using example() inside a strategy definition is a bad "
                    "idea. Instead consider using hypothesis.strategies.builds() "
                    "or @hypothesis.strategies.composite to define your strategy."
                    " See https://hypothesis.readthedocs.io/en/latest/data.html"
                    "#hypothesis.strategies.builds or "
                    "https://hypothesis.readthedocs.io/en/latest/data.html"
                    "#composite-strategies for more details."
                )
            else:
                raise HypothesisException(
                    "Using example() inside a test function is a bad "
                    "idea. Instead consider using hypothesis.strategies.data() "
                    "to draw more examples during testing. See "
                    "https://hypothesis.readthedocs.io/en/latest/data.html"
                    "#drawing-interactively-in-tests for more details."
                )

        from hypothesis.core import given

        # Note: this function has a weird name because it might appear in
        # tracebacks, and we want users to know that they can ignore it.
        @given(self)
        @settings(
            database=None,
            max_examples=10,
            deadline=None,
            verbosity=Verbosity.quiet,
            phases=(Phase.generate,),
            suppress_health_check=HealthCheck.all(),
        )
        def example_generating_inner_function(ex):
            examples.append(ex)

        examples = []  # type: List[Ex]
        example_generating_inner_function()
        return random_choice(examples)

    def map(self, pack):
        # type: (Callable[[Ex], T]) -> SearchStrategy[T]
        """Returns a new strategy that generates values by generating a value
        from this strategy and then calling pack() on the result, giving that.

        This method is part of the public API.
        """
        return MappedSearchStrategy(pack=pack, strategy=self)

    def flatmap(self, expand):
        # type: (Callable[[Ex], SearchStrategy[T]]) -> SearchStrategy[T]
        """Returns a new strategy that generates values by generating a value
        from this strategy, say x, then generating a value from
        strategy(expand(x))

        This method is part of the public API.
        """
        from hypothesis.strategies._internal.flatmapped import FlatMapStrategy

        return FlatMapStrategy(expand=expand, strategy=self)

    def filter(self, condition):
        # type: (Callable[[Ex], Any]) -> SearchStrategy[Ex]
        """Returns a new strategy that generates values from this strategy
        which satisfy the provided condition. Note that if the condition is too
        hard to satisfy this might result in your tests failing with
        Unsatisfiable.

        This method is part of the public API.
        """
        return FilteredStrategy(conditions=(condition,), strategy=self)

    def do_filtered_draw(self, data, filter_strategy):
        # Hook for strategies that want to override the behaviour of
        # FilteredStrategy. Most strategies don't, so by default we delegate
        # straight back to the default filtered-draw implementation.
        return filter_strategy.default_do_filtered_draw(data)

    @property
    def branches(self):
        # type: () -> List[SearchStrategy[Ex]]
        return [self]

    def __or__(self, other):
        """Return a strategy which produces values by randomly drawing from one
        of this strategy or the other strategy.

        This method is part of the public API.
        """
        if not isinstance(other, SearchStrategy):
            raise ValueError("Cannot | a SearchStrategy with %r" % (other,))
        return one_of_strategies((self, other))

    def validate(self):
        # type: () -> None
        """Throw an exception if the strategy is not valid.

        This can happen due to lazy construction
        """
        if self.validate_called:
            return
        try:
            self.validate_called = True
            self.do_validate()
            self.is_empty
            self.has_reusable_values
        except Exception:
            self.validate_called = False
            raise

    LABELS = {}  # type: dict

    @property
    def class_label(self):
        cls = self.__class__
        try:
            return cls.LABELS[cls]
        except KeyError:
            pass
        result = calc_label_from_cls(cls)
        cls.LABELS[cls] = result
        return result

    @property
    def label(self):
        if self.__label is calculating:
            return 0
        if self.__label is None:
            self.__label = calculating
            self.__label = self.calc_label()
        return self.__label

    def calc_label(self):
        return self.class_label

    def do_validate(self):
        pass

    def do_draw(self, data):
        # type: (ConjectureData) -> Ex
        raise NotImplementedError("%s.do_draw" % (type(self).__name__,))

    def __init__(self):
        pass


def is_simple_data(value):
    try:
        hash(value)
        return True
    except TypeError:
        return False


class SampledFromStrategy(SearchStrategy):
    """A strategy which samples from a set of elements. This is essentially
    equivalent to using a OneOfStrategy over Just strategies but may be more
    efficient and convenient.

    The conditional distribution chooses uniformly at random from some
    non-empty subset of the elements.
    """

    def __init__(self, elements):
        SearchStrategy.__init__(self)
        self.elements = cu.check_sample(elements, "sampled_from")
        assert self.elements

    def __repr__(self):
        return "sampled_from(%s)" % ", ".join(map(repr, self.elements))

    def calc_has_reusable_values(self, recur):
        return True

    def calc_is_cacheable(self, recur):
        return is_simple_data(self.elements)

    def do_draw(self, data):
        return cu.choice(data, self.elements)

    def do_filtered_draw(self, data, filter_strategy):
        # Set of indices that have been tried so far, so that we never test
        # the same element twice during a draw.
        known_bad_indices = set()

        def check_index(i):
            """Return ``True`` if the element at ``i`` satisfies the filter
            condition.
            """
            if i in known_bad_indices:
                return False
            ok = filter_strategy.condition(self.elements[i])
            if not ok:
                if not known_bad_indices:
                    filter_strategy.note_retried(data)
                known_bad_indices.add(i)
            return ok

        # Start with ordinary rejection sampling. It's fast if it works, and
        # if it doesn't work then it was only a small amount of overhead.
        for _ in hrange(3):
            i = cu.integer_range(data, 0, len(self.elements) - 1)
            if check_index(i):
                return self.elements[i]

        # If we've tried all the possible elements, give up now.
        max_good_indices = len(self.elements) - len(known_bad_indices)
        if not max_good_indices:
            return filter_not_satisfied

        # Figure out the bit-length of the index that we will write back after
        # choosing an allowed element.
        write_length = bit_length(len(self.elements))

        # Impose an arbitrary cutoff to prevent us from wasting too much time
        # on very large element lists.
        cutoff = 10000
        max_good_indices = min(max_good_indices, cutoff)

        # Before building the list of allowed indices, speculatively choose
        # one of them. We don't yet know how many allowed indices there will be,
        # so this choice might be out-of-bounds, but that's OK.
        speculative_index = cu.integer_range(data, 0, max_good_indices - 1)

        # Calculate the indices of allowed values, so that we can choose one
        # of them at random. But if we encounter the speculatively-chosen one,
        # just use that and return immediately.
        allowed_indices = []
        for i in hrange(min(len(self.elements), cutoff)):
            if check_index(i):
                allowed_indices.append(i)
                if len(allowed_indices) > speculative_index:
                    # Early-exit case: We reached the speculative index, so
                    # we just return the corresponding element.
                    data.draw_bits(write_length, forced=i)
                    return self.elements[i]

        # The speculative index didn't work out, but at this point we've built
        # the complete list of allowed indices, so we can just choose one of
        # them.
        if allowed_indices:
            i = cu.choice(data, allowed_indices)
            data.draw_bits(write_length, forced=i)
            return self.elements[i]
        # If there are no allowed indices, the filter couldn't be satisfied.

        return filter_not_satisfied


class OneOfStrategy(SearchStrategy):
    """Implements a union of strategies. Given a number of strategies this
    generates values which could have come from any of them.

    The conditional distribution draws uniformly at random from some
    non-empty subset of these strategies and then draws from the
    conditional distribution of that strategy.
    """

    def __init__(self, strategies):
        SearchStrategy.__init__(self)
        strategies = tuple(strategies)
        self.original_strategies = list(strategies)
        self.__element_strategies = None
        self.__in_branches = False

    def calc_is_empty(self, recur):
        return all(recur(e) for e in self.original_strategies)

    def calc_has_reusable_values(self, recur):
        return all(recur(e) for e in self.original_strategies)

    def calc_is_cacheable(self, recur):
        return all(recur(e) for e in self.original_strategies)

    @property
    def element_strategies(self):
        if self.__element_strategies is None:
            strategies = []
            for arg in self.original_strategies:
                check_strategy(arg)
                if not arg.is_empty:
                    strategies.extend([s for s in arg.branches if not s.is_empty])
            pruned = []
            seen = set()
            for s in strategies:
                if s is self:
                    continue
                if s in seen:
                    continue
                seen.add(s)
                pruned.append(s)
            self.__element_strategies = pruned
        return self.__element_strategies

    def calc_label(self):
        return combine_labels(
            self.class_label, *[p.label for p in self.original_strategies]
        )

    def do_draw(self, data):
        # type: (ConjectureData) -> Ex
        strategy = data.draw(
            SampledFromStrategy(self.element_strategies).filter(
                lambda s: s.available(data)
            )
        )
        return data.draw(strategy)

    def __repr__(self):
        return "one_of(%s)" % ", ".join(map(repr, self.original_strategies))

    def do_validate(self):
        for e in self.element_strategies:
            e.validate()

    @property
    def branches(self):
        if not self.__in_branches:
            try:
                self.__in_branches = True
                return self.element_strategies
            finally:
                self.__in_branches = False
        else:
            return [self]


class MappedSearchStrategy(SearchStrategy):
    """A strategy which is defined purely by conversion to and from another
    strategy.

    Its parameter and distribution come from that other strategy.
    """

    def __init__(self, strategy, pack=None):
        SearchStrategy.__init__(self)
        self.mapped_strategy = strategy
        if pack is not None:
            self.pack = pack

    def calc_is_empty(self, recur):
        return recur(self.mapped_strategy)

    def calc_is_cacheable(self, recur):
        return recur(self.mapped_strategy)

    def __repr__(self):
        if not hasattr(self, "_cached_repr"):
            self._cached_repr = "%r.map(%s)" % (
                self.mapped_strategy,
                get_pretty_function_description(self.pack),
            )
        return self._cached_repr

    def do_validate(self):
        self.mapped_strategy.validate()

    def pack(self, x):
        """Take a value produced by the underlying mapped_strategy and turn it
        into a value suitable for outputting from this strategy."""
        raise NotImplementedError("%s.pack()" % (self.__class__.__name__))

    def do_draw(self, data):
        # type: (ConjectureData) -> Ex
        for _ in range(3):
            i = data.index
            try:
                data.start_example(MAPPED_SEARCH_STRATEGY_DO_DRAW_LABEL)
                result = self.pack(data.draw(self.mapped_strategy))
                data.stop_example()
                return result
            except UnsatisfiedAssumption:
                data.stop_example(discard=True)
                if data.index == i:
                    raise
        raise UnsatisfiedAssumption()

    @property
    def branches(self):
        # type: () -> List[SearchStrategy[Ex]]
        return [
            MappedSearchStrategy(pack=self.pack, strategy=strategy)
            for strategy in self.mapped_strategy.branches
        ]


filter_not_satisfied = UniqueIdentifier("filter not satisfied")


class FilteredStrategy(SearchStrategy):
    def __init__(self, strategy, conditions):
        super(FilteredStrategy, self).__init__()
        if isinstance(strategy, FilteredStrategy):
            # Flatten chained filters into a single filter with multiple
            # conditions.
            self.flat_conditions = strategy.flat_conditions + conditions
            self.filtered_strategy = strategy.filtered_strategy
        else:
            self.flat_conditions = conditions
            self.filtered_strategy = strategy

        assert self.flat_conditions
        assert isinstance(self.flat_conditions, tuple)
        assert not isinstance(self.filtered_strategy, FilteredStrategy)

        self.__condition = None

    def calc_is_empty(self, recur):
        return recur(self.filtered_strategy)

    def calc_is_cacheable(self, recur):
        return recur(self.filtered_strategy)

    def __repr__(self):
        if not hasattr(self, "_cached_repr"):
            self._cached_repr = "%r%s" % (
                self.filtered_strategy,
                "".join(
                    ".filter(%s)" % get_pretty_function_description(cond)
                    for cond in self.flat_conditions
                ),
            )
        return self._cached_repr

    def do_validate(self):
        self.filtered_strategy.validate()

    @property
    def condition(self):
        if self.__condition is None:
            assert self.flat_conditions
            if len(self.flat_conditions) == 1:
                # Avoid an extra indirection in the common case of only one
                # condition.
                self.__condition = self.flat_conditions[0]
            else:
                self.__condition = lambda x: all(
                    cond(x) for cond in self.flat_conditions
                )
        return self.__condition

    def do_draw(self, data):
        # type: (ConjectureData) -> Ex
        result = self.filtered_strategy.do_filtered_draw(
            data=data, filter_strategy=self
        )
        if result is not filter_not_satisfied:
            return result

        data.note_event("Aborted test because unable to satisfy %r" % (self,))
        data.mark_invalid()
        raise AssertionError("Unreachable, for Mypy")  # pragma: no cover

    def note_retried(self, data):
        data.note_event(lazyformat("Retried draw from %r to satisfy filter", self))

    def default_do_filtered_draw(self, data):
        for i in hrange(3):
            start_index = data.index
            value = data.draw(self.filtered_strategy)
            if self.condition(value):
                return value
            else:
                if i == 0:
                    self.note_retried(data)
                # This is to guard against the case where we consume no data.
                # As long as we consume data, we'll eventually pass or raise.
                # But if we don't this could be an infinite loop.
                assume(data.index > start_index)

        return filter_not_satisfied

    @property
    def branches(self):
        # type: () -> List[SearchStrategy[Ex]]
        return [
            FilteredStrategy(strategy=strategy, conditions=self.flat_conditions)
            for strategy in self.filtered_strategy.branches
        ]


@check_function
def check_strategy(arg, name=""):
    check_type(SearchStrategy, arg, name)
