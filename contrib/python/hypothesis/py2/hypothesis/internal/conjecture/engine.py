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

from collections import Counter, defaultdict
from enum import Enum
from random import Random, getrandbits
from weakref import WeakKeyDictionary

import attr

from hypothesis import HealthCheck, Phase, Verbosity, settings as Settings
from hypothesis._settings import local_settings
from hypothesis.internal.cache import LRUReusedCache
from hypothesis.internal.compat import ceil, hbytes, int_from_bytes
from hypothesis.internal.conjecture.data import (
    ConjectureData,
    ConjectureResult,
    DataObserver,
    Overrun,
    Status,
    StopTest,
)
from hypothesis.internal.conjecture.datatree import (
    DataTree,
    PreviouslyUnseenBehaviour,
    TreeRecordingObserver,
)
from hypothesis.internal.conjecture.junkdrawer import clamp
from hypothesis.internal.conjecture.pareto import NO_SCORE, ParetoFront, ParetoOptimiser
from hypothesis.internal.conjecture.shrinker import Shrinker, sort_key
from hypothesis.internal.healthcheck import fail_health_check
from hypothesis.reporting import base_report

# Tell pytest to omit the body of this module from tracebacks
# https://docs.pytest.org/en/latest/example/simple.html#writing-well-integrated-assertion-helpers
__tracebackhide__ = True


MAX_SHRINKS = 500
CACHE_SIZE = 10000
MUTATION_POOL_SIZE = 100
MIN_TEST_CALLS = 10
BUFFER_SIZE = 8 * 1024


@attr.s
class HealthCheckState(object):
    valid_examples = attr.ib(default=0)
    invalid_examples = attr.ib(default=0)
    overrun_examples = attr.ib(default=0)
    draw_times = attr.ib(default=attr.Factory(list))


class ExitReason(Enum):
    max_examples = 0
    max_iterations = 1
    max_shrinks = 3
    finished = 4
    flaky = 5


class RunIsComplete(Exception):
    pass


class ConjectureRunner(object):
    def __init__(self, test_function, settings=None, random=None, database_key=None):
        self._test_function = test_function
        self.settings = settings or Settings()
        self.shrinks = 0
        self.call_count = 0
        self.event_call_counts = Counter()
        self.valid_examples = 0
        self.random = random or Random(getrandbits(128))
        self.database_key = database_key
        self.status_runtimes = {}

        self.all_drawtimes = []
        self.all_runtimes = []

        self.events_to_strings = WeakKeyDictionary()

        self.interesting_examples = {}
        # We use call_count because there may be few possible valid_examples.
        self.first_bug_found_at = None
        self.last_bug_found_at = None

        self.shrunk_examples = set()

        self.health_check_state = None

        self.tree = DataTree()

        self.best_observed_targets = defaultdict(lambda: NO_SCORE)
        self.best_examples_of_observed_targets = {}

        # We keep the pareto front in the example database if we have one. This
        # is only marginally useful at present, but speeds up local development
        # because it means that large targets will be quickly surfaced in your
        # testing.
        if self.database_key is not None and self.settings.database is not None:
            self.pareto_front = ParetoFront(self.random)
            self.pareto_front.on_evict(self.on_pareto_evict)
        else:
            self.pareto_front = None

        # We want to be able to get the ConjectureData object that results
        # from running a buffer without recalculating, especially during
        # shrinking where we need to know about the structure of the
        # executed test case.
        self.__data_cache = LRUReusedCache(CACHE_SIZE)

    @property
    def should_optimise(self):
        return Phase.target in self.settings.phases

    def __tree_is_exhausted(self):
        return self.tree.is_exhausted

    def __stoppable_test_function(self, data):
        """Run ``self._test_function``, but convert a ``StopTest`` exception
        into a normal return.
        """
        try:
            self._test_function(data)
        except StopTest as e:
            if e.testcounter == data.testcounter:
                # This StopTest has successfully stopped its test, and can now
                # be discarded.
                pass
            else:
                # This StopTest was raised by a different ConjectureData. We
                # need to re-raise it so that it will eventually reach the
                # correct engine.
                raise

    def test_function(self, data):
        assert isinstance(data.observer, TreeRecordingObserver)
        self.call_count += 1

        interrupted = False
        try:
            self.__stoppable_test_function(data)
        except KeyboardInterrupt:
            interrupted = True
            raise
        except BaseException:
            self.save_buffer(data.buffer)
            raise
        finally:
            # No branch, because if we're interrupted we always raise
            # the KeyboardInterrupt, never continue to the code below.
            if not interrupted:  # pragma: no branch
                data.freeze()
                self.note_details(data)

        self.debug_data(data)

        if self.pareto_front is not None and self.pareto_front.add(data.as_result()):
            self.save_buffer(data.buffer, sub_key=b"pareto")

        assert len(data.buffer) <= BUFFER_SIZE

        if data.status >= Status.VALID:
            for k, v in data.target_observations.items():
                self.best_observed_targets[k] = max(self.best_observed_targets[k], v)

                if k not in self.best_examples_of_observed_targets:
                    self.best_examples_of_observed_targets[k] = data.as_result()
                    continue

                existing_example = self.best_examples_of_observed_targets[k]
                existing_score = existing_example.target_observations[k]

                if v < existing_score:
                    continue

                if v > existing_score or sort_key(data.buffer) < sort_key(
                    existing_example.buffer
                ):
                    self.best_examples_of_observed_targets[k] = data.as_result()

        if data.status == Status.VALID:
            self.valid_examples += 1

        if data.status == Status.INTERESTING:
            key = data.interesting_origin
            changed = False
            try:
                existing = self.interesting_examples[key]
            except KeyError:
                changed = True
                self.last_bug_found_at = self.call_count
                if self.first_bug_found_at is None:
                    self.first_bug_found_at = self.call_count
            else:
                if sort_key(data.buffer) < sort_key(existing.buffer):
                    self.shrinks += 1
                    self.downgrade_buffer(existing.buffer)
                    self.__data_cache.unpin(existing.buffer)
                    changed = True

            if changed:
                self.save_buffer(data.buffer)
                self.interesting_examples[key] = data.as_result()
                self.__data_cache.pin(data.buffer)
                self.shrunk_examples.discard(key)

            if self.shrinks >= MAX_SHRINKS:
                self.exit_with(ExitReason.max_shrinks)

        if not self.interesting_examples:
            # Note that this logic is reproduced to end the generation phase when
            # we have interesting examples.  Update that too if you change this!
            # (The doubled implementation is because here we exit the engine entirely,
            #  while in the other case below we just want to move on to shrinking.)
            if self.valid_examples >= self.settings.max_examples:
                self.exit_with(ExitReason.max_examples)
            if self.call_count >= max(
                self.settings.max_examples * 10,
                # We have a high-ish default max iterations, so that tests
                # don't become flaky when max_examples is too low.
                1000,
            ):
                self.exit_with(ExitReason.max_iterations)

        if self.__tree_is_exhausted():
            self.exit_with(ExitReason.finished)

        self.record_for_health_check(data)

    def on_pareto_evict(self, data):
        self.settings.database.delete(self.pareto_key, data.buffer)

    def generate_novel_prefix(self):
        """Uses the tree to proactively generate a starting sequence of bytes
        that we haven't explored yet for this test.

        When this method is called, we assume that there must be at
        least one novel prefix left to find. If there were not, then the
        test run should have already stopped due to tree exhaustion.
        """
        return self.tree.generate_novel_prefix(self.random)

    def record_for_health_check(self, data):
        # Once we've actually found a bug, there's no point in trying to run
        # health checks - they'll just mask the actually important information.
        if data.status == Status.INTERESTING:
            self.health_check_state = None

        state = self.health_check_state

        if state is None:
            return

        state.draw_times.extend(data.draw_times)

        if data.status == Status.VALID:
            state.valid_examples += 1
        elif data.status == Status.INVALID:
            state.invalid_examples += 1
        else:
            assert data.status == Status.OVERRUN
            state.overrun_examples += 1

        max_valid_draws = 10
        max_invalid_draws = 50
        max_overrun_draws = 20

        assert state.valid_examples <= max_valid_draws

        if state.valid_examples == max_valid_draws:
            self.health_check_state = None
            return

        if state.overrun_examples == max_overrun_draws:
            fail_health_check(
                self.settings,
                (
                    "Examples routinely exceeded the max allowable size. "
                    "(%d examples overran while generating %d valid ones)"
                    ". Generating examples this large will usually lead to"
                    " bad results. You could try setting max_size parameters "
                    "on your collections and turning "
                    "max_leaves down on recursive() calls."
                )
                % (state.overrun_examples, state.valid_examples),
                HealthCheck.data_too_large,
            )
        if state.invalid_examples == max_invalid_draws:
            fail_health_check(
                self.settings,
                (
                    "It looks like your strategy is filtering out a lot "
                    "of data. Health check found %d filtered examples but "
                    "only %d good ones. This will make your tests much "
                    "slower, and also will probably distort the data "
                    "generation quite a lot. You should adapt your "
                    "strategy to filter less. This can also be caused by "
                    "a low max_leaves parameter in recursive() calls"
                )
                % (state.invalid_examples, state.valid_examples),
                HealthCheck.filter_too_much,
            )

        draw_time = sum(state.draw_times)

        if draw_time > 1.0:
            fail_health_check(
                self.settings,
                (
                    "Data generation is extremely slow: Only produced "
                    "%d valid examples in %.2f seconds (%d invalid ones "
                    "and %d exceeded maximum size). Try decreasing "
                    "size of the data you're generating (with e.g."
                    "max_size or max_leaves parameters)."
                )
                % (
                    state.valid_examples,
                    draw_time,
                    state.invalid_examples,
                    state.overrun_examples,
                ),
                HealthCheck.too_slow,
            )

    def save_buffer(self, buffer, sub_key=None):
        if self.settings.database is not None:
            key = self.sub_key(sub_key)
            if key is None:
                return
            self.settings.database.save(key, hbytes(buffer))

    def downgrade_buffer(self, buffer):
        if self.settings.database is not None and self.database_key is not None:
            self.settings.database.move(self.database_key, self.secondary_key, buffer)

    def sub_key(self, sub_key):
        if self.database_key is None:
            return None
        if sub_key is None:
            return self.database_key
        return b".".join((self.database_key, sub_key))

    @property
    def secondary_key(self):
        return self.sub_key(b"secondary")

    @property
    def pareto_key(self):
        return self.sub_key(b"pareto")

    def note_details(self, data):
        self.__data_cache[data.buffer] = data.as_result()
        runtime = max(data.finish_time - data.start_time, 0.0)
        self.all_runtimes.append(runtime)
        self.all_drawtimes.extend(data.draw_times)
        self.status_runtimes.setdefault(data.status, []).append(runtime)
        for event in set(map(self.event_to_string, data.events)):
            self.event_call_counts[event] += 1

    def debug(self, message):
        if self.settings.verbosity >= Verbosity.debug:
            base_report(message)

    @property
    def report_debug_info(self):
        return self.settings.verbosity >= Verbosity.debug

    def debug_data(self, data):
        if not self.report_debug_info:
            return

        stack = [[]]

        def go(ex):
            if ex.length == 0:
                return
            if len(ex.children) == 0:
                stack[-1].append(int_from_bytes(data.buffer[ex.start : ex.end]))
            else:
                node = []
                stack.append(node)

                for v in ex.children:
                    go(v)
                stack.pop()
                if len(node) == 1:
                    stack[-1].extend(node)
                else:
                    stack[-1].append(node)

        go(data.examples[0])
        assert len(stack) == 1

        status = repr(data.status)

        if data.status == Status.INTERESTING:
            status = "%s (%r)" % (status, data.interesting_origin)

        self.debug(
            "%d bytes %r -> %s, %s" % (data.index, stack[0], status, data.output)
        )

    def run(self):
        with local_settings(self.settings):
            try:
                self._run()
            except RunIsComplete:
                pass
            for v in self.interesting_examples.values():
                self.debug_data(v)
            self.debug(
                u"Run complete after %d examples (%d valid) and %d shrinks"
                % (self.call_count, self.valid_examples, self.shrinks)
            )

    @property
    def database(self):
        if self.database_key is None:
            return None
        return self.settings.database

    def has_existing_examples(self):
        return self.database is not None and Phase.reuse in self.settings.phases

    def reuse_existing_examples(self):
        """If appropriate (we have a database and have been told to use it),
        try to reload existing examples from the database.

        If there are a lot we don't try all of them. We always try the
        smallest example in the database (which is guaranteed to be the
        last failure) and the largest (which is usually the seed example
        which the last failure came from but we don't enforce that). We
        then take a random sampling of the remainder and try those. Any
        examples that are no longer interesting are cleared out.
        """
        if self.has_existing_examples():
            self.debug("Reusing examples from database")
            # We have to do some careful juggling here. We have two database
            # corpora: The primary and secondary. The primary corpus is a
            # small set of minimized examples each of which has at one point
            # demonstrated a distinct bug. We want to retry all of these.

            # We also have a secondary corpus of examples that have at some
            # point demonstrated interestingness (currently only ones that
            # were previously non-minimal examples of a bug, but this will
            # likely expand in future). These are a good source of potentially
            # interesting examples, but there are a lot of them, so we down
            # sample the secondary corpus to a more manageable size.

            corpus = sorted(
                self.settings.database.fetch(self.database_key), key=sort_key
            )
            desired_size = max(2, ceil(0.1 * self.settings.max_examples))

            if len(corpus) < desired_size:
                extra_corpus = list(self.settings.database.fetch(self.secondary_key))

                shortfall = desired_size - len(corpus)

                if len(extra_corpus) <= shortfall:
                    extra = extra_corpus
                else:
                    extra = self.random.sample(extra_corpus, shortfall)
                extra.sort(key=sort_key)
                corpus.extend(extra)

            for existing in corpus:
                data = self.cached_test_function(existing)
                if data.status != Status.INTERESTING:
                    self.settings.database.delete(self.database_key, existing)
                    self.settings.database.delete(self.secondary_key, existing)

            # If we've not found any interesting examples so far we try some of
            # the pareto front from the last run.
            if len(corpus) < desired_size and not self.interesting_examples:
                desired_extra = desired_size - len(corpus)
                pareto_corpus = list(self.settings.database.fetch(self.pareto_key))
                if len(pareto_corpus) > desired_extra:
                    pareto_corpus = self.random.sample(pareto_corpus, desired_extra)
                pareto_corpus.sort(key=sort_key)

                for existing in pareto_corpus:
                    data = self.cached_test_function(existing)
                    if data not in self.pareto_front:
                        self.settings.database.delete(self.pareto_key, existing)
                    if data.status == Status.INTERESTING:
                        break

    def exit_with(self, reason):
        self.debug("exit_with(%s)" % (reason.name,))
        self.exit_reason = reason
        raise RunIsComplete()

    def generate_new_examples(self):
        if Phase.generate not in self.settings.phases:
            return
        if self.interesting_examples:
            # The example database has failing examples from a previous run,
            # so we'd rather report that they're still failing ASAP than take
            # the time to look for additional failures.
            return

        self.debug("Generating new examples")

        zero_data = self.cached_test_function(hbytes(BUFFER_SIZE))
        if zero_data.status > Status.OVERRUN:
            self.__data_cache.pin(zero_data.buffer)

        if zero_data.status == Status.OVERRUN or (
            zero_data.status == Status.VALID and len(zero_data.buffer) * 2 > BUFFER_SIZE
        ):
            fail_health_check(
                self.settings,
                "The smallest natural example for your test is extremely "
                "large. This makes it difficult for Hypothesis to generate "
                "good examples, especially when trying to reduce failing ones "
                "at the end. Consider reducing the size of your data if it is "
                "of a fixed size. You could also fix this by improving how "
                "your data shrinks (see https://hypothesis.readthedocs.io/en/"
                "latest/data.html#shrinking for details), or by introducing "
                "default values inside your strategy. e.g. could you replace "
                "some arguments with their defaults by using "
                "one_of(none(), some_complex_strategy)?",
                HealthCheck.large_base_example,
            )

        self.health_check_state = HealthCheckState()

        def should_generate_more():
            # End the generation phase where we would have ended it if no bugs had
            # been found.  This reproduces the exit logic in `self.test_function`,
            # but with the important distinction that this clause will move on to
            # the shrinking phase having found one or more bugs, while the other
            # will exit having found zero bugs.
            if (
                self.valid_examples >= self.settings.max_examples
                or self.call_count >= max(self.settings.max_examples * 10, 1000)
            ):  # pragma: no cover
                return False

            # If we haven't found a bug, keep looking - if we hit any limits on
            # the number of tests to run that will raise an exception and stop
            # the run.
            if not self.interesting_examples:
                return True
            # If we've found a bug and won't report more than one, stop looking.
            elif not self.settings.report_multiple_bugs:
                return False
            assert self.first_bug_found_at <= self.last_bug_found_at <= self.call_count
            # Otherwise, keep searching for between ten and 'a heuristic' calls.
            # We cap 'calls after first bug' so errors are reported reasonably
            # soon even for tests that are allowed to run for a very long time,
            # or sooner if the latest half of our test effort has been fruitless.
            return self.call_count < MIN_TEST_CALLS or self.call_count < min(
                self.first_bug_found_at + 1000, self.last_bug_found_at * 2
            )

        # We attempt to use the size of the minimal generated test case starting
        # from a given novel prefix as a guideline to generate smaller test
        # cases for an initial period, by restriscting ourselves to test cases
        # that are not much larger than it.
        #
        # Calculating the actual minimal generated test case is hard, so we
        # take a best guess that zero extending a prefix produces the minimal
        # test case starting with that prefix (this is true for our built in
        # strategies). This is only a reasonable thing to do if the resulting
        # test case is valid. If we regularly run into situations where it is
        # not valid then this strategy is a waste of time, so we want to
        # abandon it early. In order to do this we track how many times in a
        # row it has failed to work, and abort small test case generation when
        # it has failed too many times in a row.
        consecutive_zero_extend_is_invalid = 0

        # We control growth during initial example generation, for two
        # reasons:
        #
        # * It gives us an opportunity to find small examples early, which
        #   gives us a fast path for easy to find bugs.
        # * It avoids low probability events where we might end up
        #   generating very large examples during health checks, which
        #   on slower machines can trigger HealthCheck.too_slow.
        #
        # The heuristic we use is that we attempt to estimate the smallest
        # extension of this prefix, and limit the size to no more than
        # an order of magnitude larger than that. If we fail to estimate
        # the size accurately, we skip over this prefix and try again.
        #
        # We need to tune the example size based on the initial prefix,
        # because any fixed size might be too small, and any size based
        # on the strategy in general can fall afoul of strategies that
        # have very different sizes for different prefixes.
        small_example_cap = clamp(10, self.settings.max_examples // 10, 50)

        optimise_at = max(self.settings.max_examples // 2, small_example_cap + 1)
        ran_optimisations = False

        while should_generate_more():
            prefix = self.generate_novel_prefix()
            assert len(prefix) <= BUFFER_SIZE
            if (
                self.valid_examples <= small_example_cap
                and self.call_count <= 5 * small_example_cap
                and not self.interesting_examples
                and consecutive_zero_extend_is_invalid < 5
            ):
                minimal_example = self.cached_test_function(
                    prefix + hbytes(BUFFER_SIZE - len(prefix))
                )

                if minimal_example.status < Status.VALID:
                    consecutive_zero_extend_is_invalid += 1
                    continue

                consecutive_zero_extend_is_invalid = 0

                minimal_extension = len(minimal_example.buffer) - len(prefix)

                max_length = min(len(prefix) + minimal_extension * 10, BUFFER_SIZE)

                # We could end up in a situation where even though the prefix was
                # novel when we generated it, because we've now tried zero extending
                # it not all possible continuations of it will be novel. In order to
                # avoid making redundant test calls, we rerun it in simulation mode
                # first. If this has a predictable result, then we don't bother
                # running the test function for real here. If however we encounter
                # some novel behaviour, we try again with the real test function,
                # starting from the new novel prefix that has discovered.
                try:
                    trial_data = self.new_conjecture_data(
                        prefix=prefix, max_length=max_length
                    )
                    self.tree.simulate_test_function(trial_data)
                    continue
                except PreviouslyUnseenBehaviour:
                    pass

                # If the simulation entered part of the tree that has been killed,
                # we don't want to run this.
                if trial_data.observer.killed:
                    continue

                # We might have hit the cap on number of examples we should
                # run when calculating the minimal example.
                if not should_generate_more():
                    break

                prefix = trial_data.buffer
            else:
                max_length = BUFFER_SIZE

            data = self.new_conjecture_data(prefix=prefix, max_length=max_length)

            self.test_function(data)

            # A thing that is often useful but rarely happens by accident is
            # to generate the same value at multiple different points in the
            # test case.
            #
            # Rather than make this the responsibility of individual strategies
            # we implement a small mutator that just takes parts of the test
            # case with the same label and tries replacing one of them with a
            # copy of the other and tries running it. If we've made a good
            # guess about what to put where, this will run a similar generated
            # test case with more duplication.
            if (
                # An OVERRUN doesn't have enough information about the test
                # case to mutate, so we just skip those.
                data.status >= Status.INVALID
                # This has a tendency to trigger some weird edge cases during
                # generation so we don't let it run until we're done with the
                # health checks.
                and self.health_check_state is None
            ):
                initial_calls = self.call_count
                failed_mutations = 0
                while (
                    should_generate_more()
                    # We implement fairly conservative checks for how long we
                    # we should run mutation for, as it's generally not obvious
                    # how helpful it is for any given test case.
                    and self.call_count <= initial_calls + 5
                    and failed_mutations <= 5
                ):
                    groups = defaultdict(list)
                    for ex in data.examples:
                        groups[ex.label, ex.depth].append(ex)

                    groups = [v for v in groups.values() if len(v) > 1]

                    if not groups:
                        break

                    group = self.random.choice(groups)

                    ex1, ex2 = sorted(
                        self.random.sample(group, 2), key=lambda i: i.index
                    )
                    assert ex1.end <= ex2.start

                    replacements = [data.buffer[e.start : e.end] for e in [ex1, ex2]]

                    replacement = self.random.choice(replacements)

                    try:
                        # We attempt to replace both the the examples with
                        # whichever choice we made. Note that this might end
                        # up messing up and getting the example boundaries
                        # wrong - labels matching are only a best guess as to
                        # whether the two are equivalent - but it doesn't
                        # really matter. It may not achieve the desired result
                        # but it's still a perfectly acceptable choice sequence.
                        # to try.
                        new_data = self.cached_test_function(
                            data.buffer[: ex1.start]
                            + replacement
                            + data.buffer[ex1.end : ex2.start]
                            + replacement
                            + data.buffer[ex2.end :],
                            # We set error_on_discard so that we don't end up
                            # entering parts of the tree we consider redundant
                            # and not worth exploring.
                            error_on_discard=True,
                            extend=BUFFER_SIZE,
                        )
                    except ContainsDiscard:
                        failed_mutations += 1
                        continue

                    if (
                        new_data.status >= data.status
                        and data.buffer != new_data.buffer
                        and all(
                            k in new_data.target_observations
                            and new_data.target_observations[k] >= v
                            for k, v in data.target_observations.items()
                        )
                    ):
                        data = new_data
                        failed_mutations = 0
                    else:
                        failed_mutations += 1

            # Although the optimisations are logically a distinct phase, we
            # actually normally run them as part of example generation. The
            # reason for this is that we cannot guarantee that optimisation
            # actually exhausts our budget: It might finish running and we
            # discover that actually we still could run a bunch more test cases
            # if we want.
            if (
                self.valid_examples >= max(small_example_cap, optimise_at)
                and not ran_optimisations
            ):
                ran_optimisations = True
                self.optimise_targets()

    def optimise_targets(self):
        """If any target observations have been made, attempt to optimise them
        all."""
        if not self.should_optimise:
            return
        from hypothesis.internal.conjecture.optimiser import Optimiser

        # We want to avoid running the optimiser for too long in case we hit
        # an unbounded target score. We start this off fairly conservatively
        # in case interesting examples are easy to find and then ramp it up
        # on an exponential schedule so we don't hamper the optimiser too much
        # if it needs a long time to find good enough improvements.
        max_improvements = 10
        while True:
            prev_calls = self.call_count

            any_improvements = False

            for target, data in list(self.best_examples_of_observed_targets.items()):
                optimiser = Optimiser(
                    self, data, target, max_improvements=max_improvements
                )
                optimiser.run()
                if optimiser.improvements > 0:
                    any_improvements = True

            if self.interesting_examples:
                break

            max_improvements *= 2

            if any_improvements:
                continue

            self.pareto_optimise()

            if prev_calls == self.call_count:
                break

    def pareto_optimise(self):
        if self.pareto_front is not None:
            ParetoOptimiser(self).run()

    def _run(self):
        self.reuse_existing_examples()
        self.generate_new_examples()

        # We normally run the targeting phase mixed in with the generate phase,
        # but if we've been asked to run it but not generation then we have to
        # run it explciitly on its own here.
        if Phase.generate not in self.settings.phases:
            self.optimise_targets()
        self.shrink_interesting_examples()
        self.exit_with(ExitReason.finished)

    def new_conjecture_data(self, prefix, max_length=BUFFER_SIZE, observer=None):
        return ConjectureData(
            prefix=prefix,
            max_length=max_length,
            random=self.random,
            observer=observer or self.tree.new_observer(),
        )

    def new_conjecture_data_for_buffer(self, buffer):
        return ConjectureData.for_buffer(buffer, observer=self.tree.new_observer())

    def shrink_interesting_examples(self):
        """If we've found interesting examples, try to replace each of them
        with a minimal interesting example with the same interesting_origin.

        We may find one or more examples with a new interesting_origin
        during the shrink process. If so we shrink these too.
        """
        if Phase.shrink not in self.settings.phases or not self.interesting_examples:
            return

        self.debug("Shrinking interesting examples")

        for prev_data in sorted(
            self.interesting_examples.values(), key=lambda d: sort_key(d.buffer)
        ):
            assert prev_data.status == Status.INTERESTING
            data = self.new_conjecture_data_for_buffer(prev_data.buffer)
            self.test_function(data)
            if data.status != Status.INTERESTING:
                self.exit_with(ExitReason.flaky)

        self.clear_secondary_key()

        while len(self.shrunk_examples) < len(self.interesting_examples):
            target, example = min(
                (
                    (k, v)
                    for k, v in self.interesting_examples.items()
                    if k not in self.shrunk_examples
                ),
                key=lambda kv: (sort_key(kv[1].buffer), sort_key(repr(kv[0]))),
            )
            self.debug("Shrinking %r" % (target,))

            if not self.settings.report_multiple_bugs:
                # If multi-bug reporting is disabled, we shrink our currently-minimal
                # failure, allowing 'slips' to any bug with a smaller minimal example.
                self.shrink(example, lambda d: d.status == Status.INTERESTING)
                return

            def predicate(d):
                if d.status < Status.INTERESTING:
                    return False
                return d.interesting_origin == target

            self.shrink(example, predicate)

            self.shrunk_examples.add(target)

    def clear_secondary_key(self):
        if self.has_existing_examples():
            # If we have any smaller examples in the secondary corpus, now is
            # a good time to try them to see if they work as shrinks. They
            # probably won't, but it's worth a shot and gives us a good
            # opportunity to clear out the database.

            # It's not worth trying the primary corpus because we already
            # tried all of those in the initial phase.
            corpus = sorted(
                self.settings.database.fetch(self.secondary_key), key=sort_key
            )
            for c in corpus:
                primary = {v.buffer for v in self.interesting_examples.values()}

                cap = max(map(sort_key, primary))

                if sort_key(c) > cap:
                    break
                else:
                    self.cached_test_function(c)
                    # We unconditionally remove c from the secondary key as it
                    # is either now primary or worse than our primary example
                    # of this reason for interestingness.
                    self.settings.database.delete(self.secondary_key, c)

    def shrink(self, example, predicate):
        s = self.new_shrinker(example, predicate)
        s.shrink()
        return s.shrink_target

    def new_shrinker(self, example, predicate):
        return Shrinker(self, example, predicate)

    def cached_test_function(self, buffer, error_on_discard=False, extend=0):
        """Checks the tree to see if we've tested this buffer, and returns the
        previous result if we have.

        Otherwise we call through to ``test_function``, and return a
        fresh result.

        If ``error_on_discard`` is set to True this will raise ``ContainsDiscard``
        in preference to running the actual test function. This is to allow us
        to skip test cases we expect to be redundant in some cases. Note that
        it may be the case that we don't raise ``ContainsDiscard`` even if the
        result has discards if we cannot determine from previous runs whether
        it will have a discard.
        """
        buffer = hbytes(buffer)[:BUFFER_SIZE]

        max_length = min(BUFFER_SIZE, len(buffer) + extend)

        def check_result(result):
            assert result is Overrun or (
                isinstance(result, ConjectureResult) and result.status != Status.OVERRUN
            )
            return result

        try:
            return check_result(self.__data_cache[buffer])
        except KeyError:
            pass

        if error_on_discard:

            class DiscardObserver(DataObserver):
                def kill_branch(self):
                    raise ContainsDiscard()

            observer = DiscardObserver()
        else:
            observer = DataObserver()

        dummy_data = self.new_conjecture_data(
            prefix=buffer, max_length=max_length, observer=observer
        )

        try:
            self.tree.simulate_test_function(dummy_data)
        except PreviouslyUnseenBehaviour:
            pass
        else:
            if dummy_data.status > Status.OVERRUN:
                dummy_data.freeze()
                try:
                    return self.__data_cache[dummy_data.buffer]
                except KeyError:
                    pass
            else:
                self.__data_cache[buffer] = Overrun
                return Overrun

        # We didn't find a match in the tree, so we need to run the test
        # function normally. Note that test_function will automatically
        # add this to the tree so we don't need to update the cache.

        result = None

        data = self.new_conjecture_data(
            prefix=max((buffer, dummy_data.buffer), key=len), max_length=max_length,
        )
        self.test_function(data)
        result = check_result(data.as_result())
        self.__data_cache[buffer] = result
        return result

    def event_to_string(self, event):
        if isinstance(event, str):
            return event
        try:
            return self.events_to_strings[event]
        except KeyError:
            pass
        result = str(event)
        self.events_to_strings[event] = result
        return result


class ContainsDiscard(Exception):
    pass
