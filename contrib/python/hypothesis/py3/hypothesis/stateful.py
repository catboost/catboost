# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""This module provides support for a stateful style of testing, where tests
attempt to find a sequence of operations that cause a breakage rather than just
a single value.

Notably, the set of steps available at any point may depend on the
execution to date.
"""

import inspect
from copy import copy
from functools import lru_cache
from io import StringIO
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Union,
    overload,
)
from unittest import TestCase

import attr

from hypothesis import strategies as st
from hypothesis._settings import (
    HealthCheck,
    Verbosity,
    note_deprecation,
    settings as Settings,
)
from hypothesis.control import _current_build_context, current_build_context
from hypothesis.core import TestFunc, given
from hypothesis.errors import InvalidArgument, InvalidDefinition
from hypothesis.internal.conjecture import utils as cu
from hypothesis.internal.healthcheck import fail_health_check
from hypothesis.internal.reflection import (
    function_digest,
    get_pretty_function_description,
    nicerepr,
    proxies,
)
from hypothesis.internal.validation import check_type
from hypothesis.reporting import current_verbosity, report
from hypothesis.strategies._internal.featureflags import FeatureStrategy
from hypothesis.strategies._internal.strategies import (
    Ex,
    Ex_Inv,
    OneOfStrategy,
    SearchStrategy,
    check_strategy,
)
from hypothesis.vendor.pretty import RepresentationPrinter

STATE_MACHINE_RUN_LABEL = cu.calc_label_from_name("another state machine step")
SHOULD_CONTINUE_LABEL = cu.calc_label_from_name("should we continue drawing")


class _OmittedArgument:
    """Sentinel class to prevent overlapping overloads in type hints. See comments
    above the overloads of @rule."""


class TestCaseProperty:  # pragma: no cover
    def __get__(self, obj, typ=None):
        if obj is not None:
            typ = type(obj)
        return typ._to_test_case()

    def __set__(self, obj, value):
        raise AttributeError("Cannot set TestCase")

    def __delete__(self, obj):
        raise AttributeError("Cannot delete TestCase")


def run_state_machine_as_test(state_machine_factory, *, settings=None, _min_steps=0):
    """Run a state machine definition as a test, either silently doing nothing
    or printing a minimal breaking program and raising an exception.

    state_machine_factory is anything which returns an instance of
    RuleBasedStateMachine when called with no arguments - it can be a class or a
    function. settings will be used to control the execution of the test.
    """
    if settings is None:
        try:
            settings = state_machine_factory.TestCase.settings
            check_type(Settings, settings, "state_machine_factory.TestCase.settings")
        except AttributeError:
            settings = Settings(deadline=None, suppress_health_check=list(HealthCheck))
    check_type(Settings, settings, "settings")
    check_type(int, _min_steps, "_min_steps")
    if _min_steps < 0:
        # Because settings can vary via e.g. profiles, settings.stateful_step_count
        # overrides this argument and we don't bother cross-validating.
        raise InvalidArgument(f"_min_steps={_min_steps} must be non-negative.")

    @settings
    @given(st.data())
    def run_state_machine(factory, data):
        cd = data.conjecture_data
        machine = factory()
        check_type(RuleBasedStateMachine, machine, "state_machine_factory()")
        cd.hypothesis_runner = machine

        print_steps = (
            current_build_context().is_final or current_verbosity() >= Verbosity.debug
        )
        try:
            if print_steps:
                report(f"state = {machine.__class__.__name__}()")
            machine.check_invariants(settings)
            max_steps = settings.stateful_step_count
            steps_run = 0

            while True:
                # We basically always want to run the maximum number of steps,
                # but need to leave a small probability of terminating early
                # in order to allow for reducing the number of steps once we
                # find a failing test case, so we stop with probability of
                # 2 ** -16 during normal operation but force a stop when we've
                # generated enough steps.
                cd.start_example(STATE_MACHINE_RUN_LABEL)
                must_stop = None
                if steps_run >= max_steps:
                    must_stop = True
                elif steps_run <= _min_steps:
                    must_stop = False
                if cu.biased_coin(cd, 2**-16, forced=must_stop):
                    break
                steps_run += 1

                # Choose a rule to run, preferring an initialize rule if there are
                # any which have not been run yet.
                if machine._initialize_rules_to_run:
                    init_rules = [
                        st.tuples(st.just(rule), st.fixed_dictionaries(rule.arguments))
                        for rule in machine._initialize_rules_to_run
                    ]
                    rule, data = cd.draw(st.one_of(init_rules))
                    machine._initialize_rules_to_run.remove(rule)
                else:
                    rule, data = cd.draw(machine._rules_strategy)

                # Pretty-print the values this rule was called with *before* calling
                # _add_result_to_targets, to avoid printing arguments which are also
                # a return value using the variable name they are assigned to.
                # See https://github.com/HypothesisWorks/hypothesis/issues/2341
                if print_steps:
                    data_to_print = {
                        k: machine._pretty_print(v) for k, v in data.items()
                    }

                # Assign 'result' here in case executing the rule fails below
                result = multiple()
                try:
                    data = dict(data)
                    for k, v in list(data.items()):
                        if isinstance(v, VarReference):
                            data[k] = machine.names_to_values[v.name]
                    result = rule.function(machine, **data)
                    if rule.targets:
                        if isinstance(result, MultipleResults):
                            for single_result in result.values:
                                machine._add_result_to_targets(
                                    rule.targets, single_result
                                )
                        else:
                            machine._add_result_to_targets(rule.targets, result)
                    elif result is not None:
                        fail_health_check(
                            settings,
                            "Rules should return None if they have no target bundle, "
                            f"but {rule.function.__qualname__} returned {result!r}",
                            HealthCheck.return_value,
                        )
                finally:
                    if print_steps:
                        # 'result' is only used if the step has target bundles.
                        # If it does, and the result is a 'MultipleResult',
                        # then 'print_step' prints a multi-variable assignment.
                        machine._print_step(rule, data_to_print, result)
                machine.check_invariants(settings)
                cd.stop_example()
        finally:
            if print_steps:
                report("state.teardown()")
            machine.teardown()

    # Use a machine digest to identify stateful tests in the example database
    run_state_machine.hypothesis.inner_test._hypothesis_internal_add_digest = (
        function_digest(state_machine_factory)
    )
    # Copy some attributes so @seed and @reproduce_failure "just work"
    run_state_machine._hypothesis_internal_use_seed = getattr(
        state_machine_factory, "_hypothesis_internal_use_seed", None
    )
    run_state_machine._hypothesis_internal_use_reproduce_failure = getattr(
        state_machine_factory, "_hypothesis_internal_use_reproduce_failure", None
    )
    run_state_machine._hypothesis_internal_print_given_args = False

    run_state_machine(state_machine_factory)


class StateMachineMeta(type):
    def __setattr__(cls, name, value):
        if name == "settings" and isinstance(value, Settings):
            raise AttributeError(
                (
                    "Assigning {cls}.settings = {value} does nothing. Assign "
                    "to {cls}.TestCase.settings, or use @{value} as a decorator "
                    "on the {cls} class."
                ).format(cls=cls.__name__, value=value)
            )
        return super().__setattr__(name, value)


class RuleBasedStateMachine(metaclass=StateMachineMeta):
    """A RuleBasedStateMachine gives you a structured way to define state machines.

    The idea is that a state machine carries a bunch of types of data
    divided into Bundles, and has a set of rules which may read data
    from bundles (or just from normal strategies) and push data onto
    bundles. At any given point a random applicable rule will be
    executed.
    """

    _rules_per_class: Dict[type, List[classmethod]] = {}
    _invariants_per_class: Dict[type, List[classmethod]] = {}
    _initializers_per_class: Dict[type, List[classmethod]] = {}

    def __init__(self) -> None:
        if not self.rules():
            raise InvalidDefinition(f"Type {type(self).__name__} defines no rules")
        self.bundles: Dict[str, list] = {}
        self.name_counter = 1
        self.names_to_values: Dict[str, Any] = {}
        self.__stream = StringIO()
        self.__printer = RepresentationPrinter(
            self.__stream, context=_current_build_context.value
        )
        self._initialize_rules_to_run = copy(self.initialize_rules())
        self._rules_strategy = RuleStrategy(self)

    def _pretty_print(self, value):
        if isinstance(value, VarReference):
            return value.name
        self.__stream.seek(0)
        self.__stream.truncate(0)
        self.__printer.output_width = 0
        self.__printer.buffer_width = 0
        self.__printer.buffer.clear()
        self.__printer.pretty(value)
        self.__printer.flush()
        return self.__stream.getvalue()

    def __repr__(self):
        return f"{type(self).__name__}({nicerepr(self.bundles)})"

    def _new_name(self):
        result = f"v{self.name_counter}"
        self.name_counter += 1
        return result

    def _last_names(self, n):
        assert self.name_counter > n
        count = self.name_counter
        return [f"v{i}" for i in range(count - n, count)]

    def bundle(self, name):
        return self.bundles.setdefault(name, [])

    @classmethod
    def initialize_rules(cls):
        try:
            return cls._initializers_per_class[cls]
        except KeyError:
            pass

        cls._initializers_per_class[cls] = []
        for _, v in inspect.getmembers(cls):
            r = getattr(v, INITIALIZE_RULE_MARKER, None)
            if r is not None:
                cls._initializers_per_class[cls].append(r)
        return cls._initializers_per_class[cls]

    @classmethod
    def rules(cls):
        try:
            return cls._rules_per_class[cls]
        except KeyError:
            pass

        cls._rules_per_class[cls] = []
        for _, v in inspect.getmembers(cls):
            r = getattr(v, RULE_MARKER, None)
            if r is not None:
                cls._rules_per_class[cls].append(r)
        return cls._rules_per_class[cls]

    @classmethod
    def invariants(cls):
        try:
            return cls._invariants_per_class[cls]
        except KeyError:
            pass

        target = []
        for _, v in inspect.getmembers(cls):
            i = getattr(v, INVARIANT_MARKER, None)
            if i is not None:
                target.append(i)
        cls._invariants_per_class[cls] = target
        return cls._invariants_per_class[cls]

    def _print_step(self, rule, data, result):
        self.step_count = getattr(self, "step_count", 0) + 1
        output_assignment = ""
        if rule.targets:
            if isinstance(result, MultipleResults):
                if len(result.values) == 1:
                    output_assignment = f"({self._last_names(1)[0]},) = "
                elif result.values:
                    output_names = self._last_names(len(result.values))
                    output_assignment = ", ".join(output_names) + " = "
            else:
                output_assignment = self._last_names(1)[0] + " = "
        report(
            "{}state.{}({})".format(
                output_assignment,
                rule.function.__name__,
                ", ".join("%s=%s" % kv for kv in data.items()),
            )
        )

    def _add_result_to_targets(self, targets, result):
        name = self._new_name()
        self.__printer.singleton_pprinters.setdefault(
            id(result), lambda obj, p, cycle: p.text(name)
        )
        self.names_to_values[name] = result
        for target in targets:
            self.bundles.setdefault(target, []).append(VarReference(name))

    def check_invariants(self, settings):
        for invar in self.invariants():
            if self._initialize_rules_to_run and not invar.check_during_init:
                continue
            if not all(precond(self) for precond in invar.preconditions):
                continue
            if (
                current_build_context().is_final
                or settings.verbosity >= Verbosity.debug
            ):
                report(f"state.{invar.function.__name__}()")
            result = invar.function(self)
            if result is not None:
                fail_health_check(
                    settings,
                    "The return value of an @invariant is always ignored, but "
                    f"{invar.function.__qualname__} returned {result!r} "
                    "instead of None",
                    HealthCheck.return_value,
                )

    def teardown(self):
        """Called after a run has finished executing to clean up any necessary
        state.

        Does nothing by default.
        """

    TestCase = TestCaseProperty()

    @classmethod
    @lru_cache()
    def _to_test_case(cls):
        class StateMachineTestCase(TestCase):
            settings = Settings(deadline=None, suppress_health_check=list(HealthCheck))

            def runTest(self):
                run_state_machine_as_test(cls)

            runTest.is_hypothesis_test = True

        StateMachineTestCase.__name__ = cls.__name__ + ".TestCase"
        StateMachineTestCase.__qualname__ = cls.__qualname__ + ".TestCase"
        return StateMachineTestCase


@attr.s()
class Rule:
    targets = attr.ib()
    function = attr.ib(repr=get_pretty_function_description)
    arguments = attr.ib()
    preconditions = attr.ib()
    bundles = attr.ib(init=False)

    def __attrs_post_init__(self):
        arguments = {}
        bundles = []
        for k, v in sorted(self.arguments.items()):
            assert not isinstance(v, BundleReferenceStrategy)
            if isinstance(v, Bundle):
                bundles.append(v)
                consume = isinstance(v, BundleConsumer)
                arguments[k] = BundleReferenceStrategy(v.name, consume)
            else:
                arguments[k] = v
        self.bundles = tuple(bundles)
        self.arguments_strategy = st.fixed_dictionaries(arguments)


self_strategy = st.runner()


class BundleReferenceStrategy(SearchStrategy):
    def __init__(self, name, consume=False):
        self.name = name
        self.consume = consume

    def do_draw(self, data):
        machine = data.draw(self_strategy)
        bundle = machine.bundle(self.name)
        if not bundle:
            data.mark_invalid()
        # Shrink towards the right rather than the left. This makes it easier
        # to delete data generated earlier, as when the error is towards the
        # end there can be a lot of hard to remove padding.
        position = cu.integer_range(data, 0, len(bundle) - 1, center=len(bundle))
        if self.consume:
            return bundle.pop(position)
        else:
            return bundle[position]


class Bundle(SearchStrategy[Ex]):
    def __init__(self, name: str, consume: bool = False) -> None:
        self.name = name
        self.__reference_strategy = BundleReferenceStrategy(name, consume)

    def do_draw(self, data):
        machine = data.draw(self_strategy)
        reference = data.draw(self.__reference_strategy)
        return machine.names_to_values[reference.name]

    def __repr__(self):
        consume = self.__reference_strategy.consume
        if consume is False:
            return f"Bundle(name={self.name!r})"
        return f"Bundle(name={self.name!r}, consume={consume!r})"

    def calc_is_empty(self, recur):
        # We assume that a bundle will grow over time
        return False

    def available(self, data):
        # ``self_strategy`` is an instance of the ``st.runner()`` strategy.
        # Hence drawing from it only returns the current state machine without
        # modifying the underlying buffer.
        machine = data.draw(self_strategy)
        return bool(machine.bundle(self.name))


class BundleConsumer(Bundle[Ex]):
    def __init__(self, bundle: Bundle[Ex]) -> None:
        super().__init__(bundle.name, consume=True)


def consumes(bundle: Bundle[Ex]) -> SearchStrategy[Ex]:
    """When introducing a rule in a RuleBasedStateMachine, this function can
    be used to mark bundles from which each value used in a step with the
    given rule should be removed. This function returns a strategy object
    that can be manipulated and combined like any other.

    For example, a rule declared with

    ``@rule(value1=b1, value2=consumes(b2), value3=lists(consumes(b3)))``

    will consume a value from Bundle ``b2`` and several values from Bundle
    ``b3`` to populate ``value2`` and ``value3`` each time it is executed.
    """
    if not isinstance(bundle, Bundle):
        raise TypeError("Argument to be consumed must be a bundle.")
    return BundleConsumer(bundle)


@attr.s()
class MultipleResults(Iterable[Ex]):
    values = attr.ib()

    def __iter__(self):
        return iter(self.values)


# We need to use an invariant typevar here to avoid a mypy error, as covariant
# typevars cannot be used as parameters.
def multiple(*args: Ex_Inv) -> MultipleResults[Ex_Inv]:
    """This function can be used to pass multiple results to the target(s) of
    a rule. Just use ``return multiple(result1, result2, ...)`` in your rule.

    It is also possible to use ``return multiple()`` with no arguments in
    order to end a rule without passing any result.
    """
    return MultipleResults(args)


def _convert_targets(targets, target):
    """Single validator and converter for target arguments."""
    if target is not None:
        if targets:
            raise InvalidArgument(
                "Passing both targets=%r and target=%r is redundant - pass "
                "targets=%r instead." % (targets, target, tuple(targets) + (target,))
            )
        targets = (target,)

    converted_targets = []
    for t in targets:
        if not isinstance(t, Bundle):
            msg = "Got invalid target %r of type %r, but all targets must be Bundles."
            if isinstance(t, OneOfStrategy):
                msg += (
                    "\nIt looks like you passed `one_of(a, b)` or `a | b` as "
                    "a target.  You should instead pass `targets=(a, b)` to "
                    "add the return value of this rule to both the `a` and "
                    "`b` bundles, or define a rule for each target if it "
                    "should be added to exactly one."
                )
            raise InvalidArgument(msg % (t, type(t)))
        while isinstance(t, Bundle):
            if isinstance(t, BundleConsumer):
                note_deprecation(
                    f"Using consumes({t.name}) doesn't makes sense in this context.  "
                    "This will be an error in a future version of Hypothesis.",
                    since="2021-09-08",
                    has_codemod=False,
                )
            t = t.name
        converted_targets.append(t)
    return tuple(converted_targets)


RULE_MARKER = "hypothesis_stateful_rule"
INITIALIZE_RULE_MARKER = "hypothesis_stateful_initialize_rule"
PRECONDITIONS_MARKER = "hypothesis_stateful_preconditions"
INVARIANT_MARKER = "hypothesis_stateful_invariant"


_RuleType = Callable[..., Union[MultipleResults[Ex], Ex]]
_RuleWrapper = Callable[[_RuleType[Ex]], _RuleType[Ex]]


# We cannot exclude `target` or `targets` from any of these signatures because
# otherwise they would be matched against the `kwargs`, either leading to
# overlapping overloads of incompatible return types, or a concrete
# implementation that does not accept all overloaded variant signatures.
# Although it is possible to reorder the variants to fix the former, it will
# always lead to the latter, as then the omitted parameter could be typed as
# a `SearchStrategy`, which the concrete implementation does not accept.
#
# Omitted `targets` parameters, where the default value is used, are typed with
# a special `_OmittedArgument` type. We cannot type them as `Tuple[()]`, because
# `Tuple[()]` is a subtype of `Sequence[Bundle[Ex]]`, leading to signature
# overlaps with incompatible return types. The `_OmittedArgument` type will never be
# encountered at runtime, and exists solely to annotate the default of `targets`.
# PEP 661 (Sentinel Values) might provide a more elegant alternative in the future.
#
# We could've also annotated `targets` as `Tuple[_OmittedArgument]`, but then when
# both `target` and `targets` are provided, mypy describes the type error as an
# invalid argument type for `targets` (expected `Tuple[_OmittedArgument]`, got ...).
# By annotating it as a bare `_OmittedArgument` type, mypy's error will warn that
# there is no overloaded signature matching the call, which is more descriptive.
#
# When `target` xor `targets` is provided, the function to decorate must return
# a value whose type matches the one stored in the bundle. When neither are
# provided, the function to decorate must return nothing. There is no variant
# for providing `target` and `targets`, as these parameters are mutually exclusive.
@overload
def rule(
    *,
    targets: Sequence[Bundle[Ex]],
    target: None = ...,
    **kwargs: SearchStrategy,
) -> _RuleWrapper[Ex]:  # pragma: no cover
    ...


@overload
def rule(
    *, target: Bundle[Ex], targets: _OmittedArgument = ..., **kwargs: SearchStrategy
) -> _RuleWrapper[Ex]:  # pragma: no cover
    ...


@overload
def rule(
    *,
    target: None = ...,
    targets: _OmittedArgument = ...,
    **kwargs: SearchStrategy,
) -> Callable[[Callable[..., None]], Callable[..., None]]:  # pragma: no cover
    ...


def rule(
    *,
    targets: Union[Sequence[Bundle[Ex]], _OmittedArgument] = (),
    target: Optional[Bundle[Ex]] = None,
    **kwargs: SearchStrategy,
) -> Union[_RuleWrapper[Ex], Callable[[Callable[..., None]], Callable[..., None]]]:
    """Decorator for RuleBasedStateMachine. Any Bundle present in ``target`` or
    ``targets`` will define where the end result of this function should go. If
    both are empty then the end result will be discarded.

    ``target`` must be a Bundle, or if the result should go to multiple
    bundles you can pass a tuple of them as the ``targets`` argument.
    It is invalid to use both arguments for a single rule.  If the result
    should go to exactly one of several bundles, define a separate rule for
    each case.

    kwargs then define the arguments that will be passed to the function
    invocation. If their value is a Bundle, or if it is ``consumes(b)``
    where ``b`` is a Bundle, then values that have previously been produced
    for that bundle will be provided. If ``consumes`` is used, the value
    will also be removed from the bundle.

    Any other kwargs should be strategies and values from them will be
    provided.
    """
    converted_targets = _convert_targets(targets, target)
    for k, v in kwargs.items():
        check_strategy(v, name=k)

    def accept(f):
        if getattr(f, INVARIANT_MARKER, None):
            raise InvalidDefinition(
                "A function cannot be used for both a rule and an invariant.",
                Settings.default,
            )
        existing_rule = getattr(f, RULE_MARKER, None)
        existing_initialize_rule = getattr(f, INITIALIZE_RULE_MARKER, None)
        if existing_rule is not None or existing_initialize_rule is not None:
            raise InvalidDefinition(
                "A function cannot be used for two distinct rules. ", Settings.default
            )
        preconditions = getattr(f, PRECONDITIONS_MARKER, ())
        rule = Rule(
            targets=converted_targets,
            arguments=kwargs,
            function=f,
            preconditions=preconditions,
        )

        @proxies(f)
        def rule_wrapper(*args, **kwargs):
            return f(*args, **kwargs)

        setattr(rule_wrapper, RULE_MARKER, rule)
        return rule_wrapper

    return accept


# See also comments of `rule`'s overloads.
@overload
def initialize(
    *,
    targets: Sequence[Bundle[Ex]],
    target: None = ...,
    **kwargs: SearchStrategy,
) -> _RuleWrapper[Ex]:  # pragma: no cover
    ...


@overload
def initialize(
    *, target: Bundle[Ex], targets: _OmittedArgument = ..., **kwargs: SearchStrategy
) -> _RuleWrapper[Ex]:  # pragma: no cover
    ...


@overload
def initialize(
    *,
    target: None = ...,
    targets: _OmittedArgument = ...,
    **kwargs: SearchStrategy,
) -> Callable[[Callable[..., None]], Callable[..., None]]:  # pragma: no cover
    ...


def initialize(
    *,
    targets: Union[Sequence[Bundle[Ex]], _OmittedArgument] = (),
    target: Optional[Bundle[Ex]] = None,
    **kwargs: SearchStrategy,
) -> Union[_RuleWrapper[Ex], Callable[[Callable[..., None]], Callable[..., None]]]:
    """Decorator for RuleBasedStateMachine.

    An initialize decorator behaves like a rule, but all ``@initialize()`` decorated
    methods will be called before any ``@rule()`` decorated methods, in an arbitrary
    order.  Each ``@initialize()`` method will be called exactly once per run, unless
    one raises an exception - after which only the ``.teardown()`` method will be run.
    ``@initialize()`` methods may not have preconditions.
    """
    converted_targets = _convert_targets(targets, target)
    for k, v in kwargs.items():
        check_strategy(v, name=k)

    def accept(f):
        if getattr(f, INVARIANT_MARKER, None):
            raise InvalidDefinition(
                "A function cannot be used for both a rule and an invariant.",
                Settings.default,
            )
        existing_rule = getattr(f, RULE_MARKER, None)
        existing_initialize_rule = getattr(f, INITIALIZE_RULE_MARKER, None)
        if existing_rule is not None or existing_initialize_rule is not None:
            raise InvalidDefinition(
                "A function cannot be used for two distinct rules. ", Settings.default
            )
        preconditions = getattr(f, PRECONDITIONS_MARKER, ())
        if preconditions:
            raise InvalidDefinition(
                "An initialization rule cannot have a precondition. ", Settings.default
            )
        rule = Rule(
            targets=converted_targets,
            arguments=kwargs,
            function=f,
            preconditions=preconditions,
        )

        @proxies(f)
        def rule_wrapper(*args, **kwargs):
            return f(*args, **kwargs)

        setattr(rule_wrapper, INITIALIZE_RULE_MARKER, rule)
        return rule_wrapper

    return accept


@attr.s()
class VarReference:
    name = attr.ib()


# There are multiple alternatives for annotating the `precond` type, all of them
# have drawbacks. See https://github.com/HypothesisWorks/hypothesis/pull/3068#issuecomment-906642371
def precondition(precond: Callable[[Any], bool]) -> Callable[[TestFunc], TestFunc]:
    """Decorator to apply a precondition for rules in a RuleBasedStateMachine.
    Specifies a precondition for a rule to be considered as a valid step in the
    state machine, which is more efficient than using :func:`~hypothesis.assume`
    within the rule.  The ``precond`` function will be called with the instance of
    RuleBasedStateMachine and should return True or False. Usually it will need
    to look at attributes on that instance.

    For example::

        class MyTestMachine(RuleBasedStateMachine):
            state = 1

            @precondition(lambda self: self.state != 0)
            @rule(numerator=integers())
            def divide_with(self, numerator):
                self.state = numerator / self.state

    If multiple preconditions are applied to a single rule, it is only considered
    a valid step when all of them return True.  Preconditions may be applied to
    invariants as well as rules.
    """

    def decorator(f):
        @proxies(f)
        def precondition_wrapper(*args, **kwargs):
            return f(*args, **kwargs)

        existing_initialize_rule = getattr(f, INITIALIZE_RULE_MARKER, None)
        if existing_initialize_rule is not None:
            raise InvalidDefinition(
                "An initialization rule cannot have a precondition. ", Settings.default
            )

        rule = getattr(f, RULE_MARKER, None)
        invariant = getattr(f, INVARIANT_MARKER, None)
        if rule is not None:
            assert invariant is None
            new_rule = attr.evolve(rule, preconditions=rule.preconditions + (precond,))
            setattr(precondition_wrapper, RULE_MARKER, new_rule)
        elif invariant is not None:
            assert rule is None
            new_invariant = attr.evolve(
                invariant, preconditions=invariant.preconditions + (precond,)
            )
            setattr(precondition_wrapper, INVARIANT_MARKER, new_invariant)
        else:
            setattr(
                precondition_wrapper,
                PRECONDITIONS_MARKER,
                getattr(f, PRECONDITIONS_MARKER, ()) + (precond,),
            )

        return precondition_wrapper

    return decorator


@attr.s()
class Invariant:
    function = attr.ib(repr=get_pretty_function_description)
    preconditions = attr.ib()
    check_during_init = attr.ib()


def invariant(*, check_during_init: bool = False) -> Callable[[TestFunc], TestFunc]:
    """Decorator to apply an invariant for rules in a RuleBasedStateMachine.
    The decorated function will be run after every rule and can raise an
    exception to indicate failed invariants.

    For example::

        class MyTestMachine(RuleBasedStateMachine):
            state = 1

            @invariant()
            def is_nonzero(self):
                assert self.state != 0

    By default, invariants are only checked after all
    :func:`@initialize() <hypothesis.stateful.initialize>` rules have been run.
    Pass ``check_during_init=True`` for invariants which can also be checked
    during initialization.
    """
    check_type(bool, check_during_init, "check_during_init")

    def accept(f):
        if getattr(f, RULE_MARKER, None) or getattr(f, INITIALIZE_RULE_MARKER, None):
            raise InvalidDefinition(
                "A function cannot be used for both a rule and an invariant.",
                Settings.default,
            )
        existing_invariant = getattr(f, INVARIANT_MARKER, None)
        if existing_invariant is not None:
            raise InvalidDefinition(
                "A function cannot be used for two distinct invariants.",
                Settings.default,
            )
        preconditions = getattr(f, PRECONDITIONS_MARKER, ())
        invar = Invariant(
            function=f,
            preconditions=preconditions,
            check_during_init=check_during_init,
        )

        @proxies(f)
        def invariant_wrapper(*args, **kwargs):
            return f(*args, **kwargs)

        setattr(invariant_wrapper, INVARIANT_MARKER, invar)
        return invariant_wrapper

    return accept


LOOP_LABEL = cu.calc_label_from_name("RuleStrategy loop iteration")


class RuleStrategy(SearchStrategy):
    def __init__(self, machine):
        super().__init__()
        self.machine = machine
        self.rules = list(machine.rules())

        self.enabled_rules_strategy = st.shared(
            FeatureStrategy(), key=("enabled rules", machine)
        )

        # The order is a bit arbitrary. Primarily we're trying to group rules
        # that write to the same location together, and to put rules with no
        # target first as they have less effect on the structure. We order from
        # fewer to more arguments on grounds that it will plausibly need less
        # data. This probably won't work especially well and we could be
        # smarter about it, but it's better than just doing it in definition
        # order.
        self.rules.sort(
            key=lambda rule: (
                sorted(rule.targets),
                len(rule.arguments),
                rule.function.__name__,
            )
        )

    def __repr__(self):
        return "{}(machine={}({{...}}))".format(
            self.__class__.__name__,
            self.machine.__class__.__name__,
        )

    def do_draw(self, data):
        if not any(self.is_valid(rule) for rule in self.rules):
            msg = f"No progress can be made from state {self.machine!r}"
            raise InvalidDefinition(msg) from None

        feature_flags = data.draw(self.enabled_rules_strategy)

        # Note: The order of the filters here is actually quite important,
        # because checking is_enabled makes choices, so increases the size of
        # the choice sequence. This means that if we are in a case where many
        # rules are invalid we will make a lot more choices if we ask if they
        # are enabled before we ask if they are valid, so our test cases will
        # be artificially large.
        rule = data.draw(
            st.sampled_from(self.rules)
            .filter(self.is_valid)
            .filter(lambda r: feature_flags.is_enabled(r.function.__name__))
        )

        return (rule, data.draw(rule.arguments_strategy))

    def is_valid(self, rule):
        if not all(precond(self.machine) for precond in rule.preconditions):
            return False

        for b in rule.bundles:
            bundle = self.machine.bundle(b.name)
            if not bundle:
                return False
        return True
