# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import math
import time
from collections import defaultdict
from enum import IntEnum
from random import Random
from sys import float_info
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    DefaultDict,
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    List,
    Literal,
    NoReturn,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypedDict,
    TypeVar,
    Union,
)

import attr

from hypothesis.errors import Frozen, InvalidArgument, StopTest
from hypothesis.internal.cache import LRUReusedCache
from hypothesis.internal.compat import floor, int_from_bytes, int_to_bytes
from hypothesis.internal.conjecture.floats import float_to_lex, lex_to_float
from hypothesis.internal.conjecture.junkdrawer import IntList, uniform
from hypothesis.internal.conjecture.utils import (
    INT_SIZES,
    INT_SIZES_SAMPLER,
    Sampler,
    calc_label_from_name,
    many,
)
from hypothesis.internal.floats import (
    SIGNALING_NAN,
    SMALLEST_SUBNORMAL,
    float_to_int,
    make_float_clamper,
    next_down,
    next_up,
    sign_aware_lte,
)
from hypothesis.internal.intervalsets import IntervalSet

if TYPE_CHECKING:
    from typing_extensions import dataclass_transform

    from hypothesis.strategies import SearchStrategy
    from hypothesis.strategies._internal.strategies import Ex
else:

    def dataclass_transform():
        def wrapper(tp):
            return tp

        return wrapper


ONE_BOUND_INTEGERS_LABEL = calc_label_from_name("trying a one-bound int allowing 0")
INTEGER_RANGE_DRAW_LABEL = calc_label_from_name("another draw in integer_range()")
BIASED_COIN_LABEL = calc_label_from_name("biased_coin()")

TOP_LABEL = calc_label_from_name("top")
DRAW_BYTES_LABEL = calc_label_from_name("draw_bytes() in ConjectureData")
DRAW_FLOAT_LABEL = calc_label_from_name("drawing a float")
FLOAT_STRATEGY_DO_DRAW_LABEL = calc_label_from_name(
    "getting another float in FloatStrategy"
)

InterestingOrigin = Tuple[
    Type[BaseException], str, int, Tuple[Any, ...], Tuple[Tuple[Any, ...], ...]
]
TargetObservations = Dict[Optional[str], Union[int, float]]

T = TypeVar("T")


class ExtraInformation:
    """A class for holding shared state on a ``ConjectureData`` that should
    be added to the final ``ConjectureResult``."""

    def __repr__(self) -> str:
        return "ExtraInformation({})".format(
            ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items()),
        )

    def has_information(self) -> bool:
        return bool(self.__dict__)


class Status(IntEnum):
    OVERRUN = 0
    INVALID = 1
    VALID = 2
    INTERESTING = 3

    def __repr__(self) -> str:
        return f"Status.{self.name}"


@dataclass_transform()
@attr.s(frozen=True, slots=True, auto_attribs=True)
class StructuralCoverageTag:
    label: int


STRUCTURAL_COVERAGE_CACHE: Dict[int, StructuralCoverageTag] = {}


def structural_coverage(label: int) -> StructuralCoverageTag:
    try:
        return STRUCTURAL_COVERAGE_CACHE[label]
    except KeyError:
        return STRUCTURAL_COVERAGE_CACHE.setdefault(label, StructuralCoverageTag(label))


NASTY_FLOATS = sorted(
    [
        0.0,
        0.5,
        1.1,
        1.5,
        1.9,
        1.0 / 3,
        10e6,
        10e-6,
        1.175494351e-38,
        next_up(0.0),
        float_info.min,
        float_info.max,
        3.402823466e38,
        9007199254740992,
        1 - 10e-6,
        2 + 10e-6,
        1.192092896e-07,
        2.2204460492503131e-016,
    ]
    + [2.0**-n for n in (24, 14, 149, 126)]  # minimum (sub)normals for float16,32
    + [float_info.min / n for n in (2, 10, 1000, 100_000)]  # subnormal in float64
    + [math.inf, math.nan] * 5
    + [SIGNALING_NAN],
    key=float_to_lex,
)
NASTY_FLOATS = list(map(float, NASTY_FLOATS))
NASTY_FLOATS.extend([-x for x in NASTY_FLOATS])

FLOAT_INIT_LOGIC_CACHE = LRUReusedCache(4096)

DRAW_STRING_DEFAULT_MAX_SIZE = 10**10  # "arbitrarily large"


class Example:
    """Examples track the hierarchical structure of draws from the byte stream,
    within a single test run.

    Examples are created to mark regions of the byte stream that might be
    useful to the shrinker, such as:
    - The bytes used by a single draw from a strategy.
    - Useful groupings within a strategy, such as individual list elements.
    - Strategy-like helper functions that aren't first-class strategies.
    - Each lowest-level draw of bits or bytes from the byte stream.
    - A single top-level example that spans the entire input.

    Example-tracking allows the shrinker to try "high-level" transformations,
    such as rearranging or deleting the elements of a list, without having
    to understand their exact representation in the byte stream.

    Rather than store each ``Example`` as a rich object, it is actually
    just an index into the ``Examples`` class defined below. This has two
    purposes: Firstly, for most properties of examples we will never need
    to allocate storage at all, because most properties are not used on
    most examples. Secondly, by storing the properties as compact lists
    of integers, we save a considerable amount of space compared to
    Python's normal object size.

    This does have the downside that it increases the amount of allocation
    we do, and slows things down as a result, in some usage patterns because
    we repeatedly allocate the same Example or int objects, but it will
    often dramatically reduce our memory usage, so is worth it.
    """

    __slots__ = ("owner", "index")

    def __init__(self, owner: "Examples", index: int) -> None:
        self.owner = owner
        self.index = index

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if not isinstance(other, Example):
            return NotImplemented
        return (self.owner is other.owner) and (self.index == other.index)

    def __ne__(self, other: object) -> bool:
        if self is other:
            return False
        if not isinstance(other, Example):
            return NotImplemented
        return (self.owner is not other.owner) or (self.index != other.index)

    def __repr__(self) -> str:
        return f"examples[{self.index}]"

    @property
    def label(self) -> int:
        """A label is an opaque value that associates each example with its
        approximate origin, such as a particular strategy class or a particular
        kind of draw."""
        return self.owner.labels[self.owner.label_indices[self.index]]

    @property
    def parent(self):
        """The index of the example that this one is nested directly within."""
        if self.index == 0:
            return None
        return self.owner.parentage[self.index]

    @property
    def start(self) -> int:
        """The position of the start of this example in the byte stream."""
        return self.owner.starts[self.index]

    @property
    def end(self) -> int:
        """The position directly after the last byte in this byte stream.
        i.e. the example corresponds to the half open region [start, end).
        """
        return self.owner.ends[self.index]

    @property
    def depth(self):
        """Depth of this example in the example tree. The top-level example has a
        depth of 0."""
        return self.owner.depths[self.index]

    @property
    def trivial(self):
        """An example is "trivial" if it only contains forced bytes and zero bytes.
        All examples start out as trivial, and then get marked non-trivial when
        we see a byte that is neither forced nor zero."""
        return self.index in self.owner.trivial

    @property
    def discarded(self) -> bool:
        """True if this is example's ``stop_example`` call had ``discard`` set to
        ``True``. This means we believe that the shrinker should be able to delete
        this example completely, without affecting the value produced by its enclosing
        strategy. Typically set when a rejection sampler decides to reject a
        generated value and try again."""
        return self.index in self.owner.discarded

    @property
    def length(self) -> int:
        """The number of bytes in this example."""
        return self.end - self.start

    @property
    def children(self) -> "List[Example]":
        """The list of all examples with this as a parent, in increasing index
        order."""
        return [self.owner[i] for i in self.owner.children[self.index]]


class ExampleProperty:
    """There are many properties of examples that we calculate by
    essentially rerunning the test case multiple times based on the
    calls which we record in ExampleRecord.

    This class defines a visitor, subclasses of which can be used
    to calculate these properties.
    """

    def __init__(self, examples: "Examples"):
        self.example_stack: "List[int]" = []
        self.examples = examples
        self.bytes_read = 0
        self.example_count = 0
        self.block_count = 0

    def run(self) -> Any:
        """Rerun the test case with this visitor and return the
        results of ``self.finish()``."""
        self.begin()
        blocks = self.examples.blocks
        for record in self.examples.trail:
            if record == DRAW_BITS_RECORD:
                self.__push(0)
                self.bytes_read = blocks.endpoints[self.block_count]
                self.block(self.block_count)
                self.block_count += 1
                self.__pop(discarded=False)
            elif record >= START_EXAMPLE_RECORD:
                self.__push(record - START_EXAMPLE_RECORD)
            else:
                assert record in (
                    STOP_EXAMPLE_DISCARD_RECORD,
                    STOP_EXAMPLE_NO_DISCARD_RECORD,
                )
                self.__pop(discarded=record == STOP_EXAMPLE_DISCARD_RECORD)
        return self.finish()

    def __push(self, label_index: int) -> None:
        i = self.example_count
        assert i < len(self.examples)
        self.start_example(i, label_index=label_index)
        self.example_count += 1
        self.example_stack.append(i)

    def __pop(self, *, discarded: bool) -> None:
        i = self.example_stack.pop()
        self.stop_example(i, discarded=discarded)

    def begin(self) -> None:
        """Called at the beginning of the run to initialise any
        relevant state."""
        self.result = IntList.of_length(len(self.examples))

    def start_example(self, i: int, label_index: int) -> None:
        """Called at the start of each example, with ``i`` the
        index of the example and ``label_index`` the index of
        its label in ``self.examples.labels``."""

    def block(self, i: int) -> None:
        """Called with each ``draw_bits`` call, with ``i`` the index of the
        corresponding block in ``self.examples.blocks``"""

    def stop_example(self, i: int, *, discarded: bool) -> None:
        """Called at the end of each example, with ``i`` the
        index of the example and ``discarded`` being ``True`` if ``stop_example``
        was called with ``discard=True``."""

    def finish(self) -> Any:
        return self.result


def calculated_example_property(cls: Type[ExampleProperty]) -> Any:
    """Given an ``ExampleProperty`` as above we use this decorator
    to transform it into a lazy property on the ``Examples`` class,
    which has as its value the result of calling ``cls.run()``,
    computed the first time the property is accessed.

    This has the slightly weird result that we are defining nested
    classes which get turned into properties."""
    name = cls.__name__
    cache_name = "__" + name

    def lazy_calculate(self: "Examples") -> IntList:
        result = getattr(self, cache_name, None)
        if result is None:
            result = cls(self).run()
            setattr(self, cache_name, result)
        return result

    lazy_calculate.__name__ = cls.__name__
    lazy_calculate.__qualname__ = cls.__qualname__
    return property(lazy_calculate)


DRAW_BITS_RECORD = 0
STOP_EXAMPLE_DISCARD_RECORD = 1
STOP_EXAMPLE_NO_DISCARD_RECORD = 2
START_EXAMPLE_RECORD = 3


class ExampleRecord:
    """Records the series of ``start_example``, ``stop_example``, and
    ``draw_bits`` calls so that these may be stored in ``Examples`` and
    replayed when we need to know about the structure of individual
    ``Example`` objects.

    Note that there is significant similarity between this class and
    ``DataObserver``, and the plan is to eventually unify them, but
    they currently have slightly different functions and implementations.
    """

    def __init__(self) -> None:
        self.labels = [DRAW_BYTES_LABEL]
        self.__index_of_labels: "Optional[Dict[int, int]]" = {DRAW_BYTES_LABEL: 0}
        self.trail = IntList()

    def freeze(self) -> None:
        self.__index_of_labels = None

    def start_example(self, label: int) -> None:
        assert self.__index_of_labels is not None
        try:
            i = self.__index_of_labels[label]
        except KeyError:
            i = self.__index_of_labels.setdefault(label, len(self.labels))
            self.labels.append(label)
        self.trail.append(START_EXAMPLE_RECORD + i)

    def stop_example(self, *, discard: bool) -> None:
        if discard:
            self.trail.append(STOP_EXAMPLE_DISCARD_RECORD)
        else:
            self.trail.append(STOP_EXAMPLE_NO_DISCARD_RECORD)

    def draw_bits(self, n: int, forced: Optional[int]) -> None:
        self.trail.append(DRAW_BITS_RECORD)


class Examples:
    """A lazy collection of ``Example`` objects, derived from
    the record of recorded behaviour in ``ExampleRecord``.

    Behaves logically as if it were a list of ``Example`` objects,
    but actually mostly exists as a compact store of information
    for them to reference into. All properties on here are best
    understood as the backing storage for ``Example`` and are
    described there.
    """

    def __init__(self, record: ExampleRecord, blocks: "Blocks") -> None:
        self.trail = record.trail
        self.labels = record.labels
        self.__length = (
            self.trail.count(STOP_EXAMPLE_DISCARD_RECORD)
            + record.trail.count(STOP_EXAMPLE_NO_DISCARD_RECORD)
            + record.trail.count(DRAW_BITS_RECORD)
        )
        self.blocks = blocks
        self.__children: "Optional[List[Sequence[int]]]" = None

    class _starts_and_ends(ExampleProperty):
        def begin(self):
            self.starts = IntList.of_length(len(self.examples))
            self.ends = IntList.of_length(len(self.examples))

        def start_example(self, i: int, label_index: int) -> None:
            self.starts[i] = self.bytes_read

        def stop_example(self, i: int, *, discarded: bool) -> None:
            self.ends[i] = self.bytes_read

        def finish(self) -> Tuple[IntList, IntList]:
            return (self.starts, self.ends)

    starts_and_ends: "Tuple[IntList, IntList]" = calculated_example_property(
        _starts_and_ends
    )

    @property
    def starts(self) -> IntList:
        return self.starts_and_ends[0]

    @property
    def ends(self) -> IntList:
        return self.starts_and_ends[1]

    class _discarded(ExampleProperty):
        def begin(self) -> None:
            self.result: "Set[int]" = set()  # type: ignore  # IntList in parent class

        def finish(self) -> FrozenSet[int]:
            return frozenset(self.result)

        def stop_example(self, i: int, *, discarded: bool) -> None:
            if discarded:
                self.result.add(i)

    discarded: FrozenSet[int] = calculated_example_property(_discarded)

    class _trivial(ExampleProperty):
        def begin(self) -> None:
            self.nontrivial = IntList.of_length(len(self.examples))
            self.result: "Set[int]" = set()  # type: ignore  # IntList in parent class

        def block(self, i: int) -> None:
            if not self.examples.blocks.trivial(i):
                self.nontrivial[self.example_stack[-1]] = 1

        def stop_example(self, i: int, *, discarded: bool) -> None:
            if self.nontrivial[i]:
                if self.example_stack:
                    self.nontrivial[self.example_stack[-1]] = 1
            else:
                self.result.add(i)

        def finish(self) -> FrozenSet[int]:
            return frozenset(self.result)

    trivial: FrozenSet[int] = calculated_example_property(_trivial)

    class _parentage(ExampleProperty):
        def stop_example(self, i: int, *, discarded: bool) -> None:
            if i > 0:
                self.result[i] = self.example_stack[-1]

    parentage: IntList = calculated_example_property(_parentage)

    class _depths(ExampleProperty):
        def begin(self):
            self.result = IntList.of_length(len(self.examples))

        def start_example(self, i: int, label_index: int) -> None:
            self.result[i] = len(self.example_stack)

    depths: IntList = calculated_example_property(_depths)

    class _label_indices(ExampleProperty):
        def start_example(self, i: int, label_index: int) -> None:
            self.result[i] = label_index

    label_indices: IntList = calculated_example_property(_label_indices)

    class _mutator_groups(ExampleProperty):
        def begin(self) -> None:
            self.groups: "Dict[Tuple[int, int], List[int]]" = defaultdict(list)

        def start_example(self, i: int, label_index: int) -> None:
            depth = len(self.example_stack)
            self.groups[label_index, depth].append(i)

        def finish(self) -> Iterable[Iterable[int]]:
            # Discard groups with only one example, since the mutator can't
            # do anything useful with them.
            return [g for g in self.groups.values() if len(g) >= 2]

    mutator_groups: List[List[int]] = calculated_example_property(_mutator_groups)

    @property
    def children(self) -> List[Sequence[int]]:
        if self.__children is None:
            children = [IntList() for _ in range(len(self))]
            for i, p in enumerate(self.parentage):
                if i > 0:
                    children[p].append(i)
            # Replace empty children lists with a tuple to reduce
            # memory usage.
            for i, c in enumerate(children):
                if not c:
                    children[i] = ()  # type: ignore
            self.__children = children  # type: ignore
        return self.__children  # type: ignore

    def __len__(self) -> int:
        return self.__length

    def __getitem__(self, i: int) -> Example:
        assert isinstance(i, int)
        n = len(self)
        if i < -n or i >= n:
            raise IndexError(f"Index {i} out of range [-{n}, {n})")
        if i < 0:
            i += n
        return Example(self, i)


@dataclass_transform()
@attr.s(slots=True, frozen=True)
class Block:
    """Blocks track the flat list of lowest-level draws from the byte stream,
    within a single test run.

    Block-tracking allows the shrinker to try "low-level"
    transformations, such as minimizing the numeric value of an
    individual call to ``draw_bits``.
    """

    start: int = attr.ib()
    end: int = attr.ib()

    # Index of this block inside the overall list of blocks.
    index: int = attr.ib()

    # True if this block's byte values were forced by a write operation.
    # As long as the bytes before this block remain the same, modifying this
    # block's bytes will have no effect.
    forced: bool = attr.ib(repr=False)

    # True if this block's byte values are all 0. Reading this flag can be
    # more convenient than explicitly checking a slice for non-zero bytes.
    all_zero: bool = attr.ib(repr=False)

    @property
    def bounds(self) -> Tuple[int, int]:
        return (self.start, self.end)

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def trivial(self) -> bool:
        return self.forced or self.all_zero


class Blocks:
    """A lazily calculated list of blocks for a particular ``ConjectureResult``
    or ``ConjectureData`` object.

    Pretends to be a list containing ``Block`` objects but actually only
    contains their endpoints right up until the point where you want to
    access the actual block, at which point it is constructed.

    This is designed to be as space efficient as possible, so will at
    various points silently transform its representation into one
    that is better suited for the current access pattern.

    In addition, it has a number of convenience methods for accessing
    properties of the block object at index ``i`` that should generally
    be preferred to using the Block objects directly, as it will not
    have to allocate the actual object."""

    __slots__ = ("endpoints", "owner", "__blocks", "__count", "__sparse")
    owner: "Union[ConjectureData, ConjectureResult, None]"
    __blocks: Union[Dict[int, Block], List[Optional[Block]]]

    def __init__(self, owner: "ConjectureData") -> None:
        self.owner = owner
        self.endpoints = IntList()
        self.__blocks = {}
        self.__count = 0
        self.__sparse = True

    def add_endpoint(self, n: int) -> None:
        """Add n to the list of endpoints."""
        assert isinstance(self.owner, ConjectureData)
        self.endpoints.append(n)

    def transfer_ownership(self, new_owner: "ConjectureResult") -> None:
        """Used to move ``Blocks`` over to a ``ConjectureResult`` object
        when that is read to be used and we no longer want to keep the
        whole ``ConjectureData`` around."""
        assert isinstance(new_owner, ConjectureResult)
        self.owner = new_owner
        self.__check_completion()

    def start(self, i: int) -> int:
        """Equivalent to self[i].start."""
        i = self._check_index(i)

        if i == 0:
            return 0
        else:
            return self.end(i - 1)

    def end(self, i: int) -> int:
        """Equivalent to self[i].end."""
        return self.endpoints[i]

    def bounds(self, i: int) -> Tuple[int, int]:
        """Equivalent to self[i].bounds."""
        return (self.start(i), self.end(i))

    def all_bounds(self) -> Iterable[Tuple[int, int]]:
        """Equivalent to [(b.start, b.end) for b in self]."""
        prev = 0
        for e in self.endpoints:
            yield (prev, e)
            prev = e

    @property
    def last_block_length(self):
        return self.end(-1) - self.start(-1)

    def __len__(self) -> int:
        return len(self.endpoints)

    def __known_block(self, i: int) -> Optional[Block]:
        try:
            return self.__blocks[i]
        except (KeyError, IndexError):
            return None

    def trivial(self, i: int) -> Any:
        """Equivalent to self.blocks[i].trivial."""
        if self.owner is not None:
            return self.start(i) in self.owner.forced_indices or not any(
                self.owner.buffer[self.start(i) : self.end(i)]
            )
        else:
            return self[i].trivial

    def _check_index(self, i: int) -> int:
        n = len(self)
        if i < -n or i >= n:
            raise IndexError(f"Index {i} out of range [-{n}, {n})")
        if i < 0:
            i += n
        return i

    def __getitem__(self, i: int) -> Block:
        i = self._check_index(i)
        assert i >= 0
        result = self.__known_block(i)
        if result is not None:
            return result

        # We store the blocks as a sparse dict mapping indices to the
        # actual result, but this isn't the best representation once we
        # stop being sparse and want to use most of the blocks. Switch
        # over to a list at that point.
        if self.__sparse and len(self.__blocks) * 2 >= len(self):
            new_blocks: "List[Optional[Block]]" = [None] * len(self)
            assert isinstance(self.__blocks, dict)
            for k, v in self.__blocks.items():
                new_blocks[k] = v
            self.__sparse = False
            self.__blocks = new_blocks
            assert self.__blocks[i] is None

        start = self.start(i)
        end = self.end(i)

        # We keep track of the number of blocks that have actually been
        # instantiated so that when every block that could be instantiated
        # has been we know that the list is complete and can throw away
        # some data that we no longer need.
        self.__count += 1

        # Integrity check: We can't have allocated more blocks than we have
        # positions for blocks.
        assert self.__count <= len(self)
        assert self.owner is not None
        result = Block(
            start=start,
            end=end,
            index=i,
            forced=start in self.owner.forced_indices,
            all_zero=not any(self.owner.buffer[start:end]),
        )
        try:
            self.__blocks[i] = result
        except IndexError:
            assert isinstance(self.__blocks, list)
            assert len(self.__blocks) < len(self)
            self.__blocks.extend([None] * (len(self) - len(self.__blocks)))
            self.__blocks[i] = result

        self.__check_completion()

        return result

    def __check_completion(self):
        """The list of blocks is complete if we have created every ``Block``
        object that we currently good and know that no more will be created.

        If this happens then we don't need to keep the reference to the
        owner around, and delete it so that there is no circular reference.
        The main benefit of this is that the gc doesn't need to run to collect
        this because normal reference counting is enough.
        """
        if self.__count == len(self) and isinstance(self.owner, ConjectureResult):
            self.owner = None

    def __iter__(self) -> Iterator[Block]:
        for i in range(len(self)):
            yield self[i]

    def __repr__(self) -> str:
        parts: "List[str]" = []
        for i in range(len(self)):
            b = self.__known_block(i)
            if b is None:
                parts.append("...")
            else:
                parts.append(repr(b))
        return "Block([{}])".format(", ".join(parts))


class _Overrun:
    status = Status.OVERRUN

    def __repr__(self):
        return "Overrun"

    def as_result(self) -> "_Overrun":
        return self


Overrun = _Overrun()

global_test_counter = 0


MAX_DEPTH = 100


class IntegerKWargs(TypedDict):
    min_value: Optional[int]
    max_value: Optional[int]
    weights: Optional[Sequence[float]]
    shrink_towards: int


class FloatKWargs(TypedDict):
    min_value: float
    max_value: float
    allow_nan: bool
    smallest_nonzero_magnitude: float


class StringKWargs(TypedDict):
    intervals: IntervalSet
    min_size: int
    max_size: Optional[int]


class BytesKWargs(TypedDict):
    size: int


class BooleanKWargs(TypedDict):
    p: float


class DataObserver:
    """Observer class for recording the behaviour of a
    ConjectureData object, primarily used for tracking
    the behaviour in the tree cache."""

    def conclude_test(
        self,
        status: Status,
        interesting_origin: Optional[InterestingOrigin],
    ) -> None:
        """Called when ``conclude_test`` is called on the
        observed ``ConjectureData``, with the same arguments.

        Note that this is called after ``freeze`` has completed.
        """

    def kill_branch(self) -> None:
        """Mark this part of the tree as not worth re-exploring."""

    def draw_integer(
        self, value: int, *, was_forced: bool, kwargs: IntegerKWargs
    ) -> None:
        pass

    def draw_float(
        self, value: float, *, was_forced: bool, kwargs: FloatKWargs
    ) -> None:
        pass

    def draw_string(
        self, value: str, *, was_forced: bool, kwargs: StringKWargs
    ) -> None:
        pass

    def draw_bytes(
        self, value: bytes, *, was_forced: bool, kwargs: BytesKWargs
    ) -> None:
        pass

    def draw_boolean(
        self, value: bool, *, was_forced: bool, kwargs: BooleanKWargs
    ) -> None:
        pass


@dataclass_transform()
@attr.s(slots=True)
class ConjectureResult:
    """Result class storing the parts of ConjectureData that we
    will care about after the original ConjectureData has outlived its
    usefulness."""

    status: Status = attr.ib()
    interesting_origin: Optional[InterestingOrigin] = attr.ib()
    buffer: bytes = attr.ib()
    blocks: Blocks = attr.ib()
    output: str = attr.ib()
    extra_information: Optional[ExtraInformation] = attr.ib()
    has_discards: bool = attr.ib()
    target_observations: TargetObservations = attr.ib()
    tags: FrozenSet[StructuralCoverageTag] = attr.ib()
    forced_indices: FrozenSet[int] = attr.ib(repr=False)
    examples: Examples = attr.ib(repr=False)
    arg_slices: Set[Tuple[int, int]] = attr.ib(repr=False)
    slice_comments: Dict[Tuple[int, int], str] = attr.ib(repr=False)

    index: int = attr.ib(init=False)

    def __attrs_post_init__(self) -> None:
        self.index = len(self.buffer)
        self.forced_indices = frozenset(self.forced_indices)

    def as_result(self) -> "ConjectureResult":
        return self


# Masks for masking off the first byte of an n-bit buffer.
# The appropriate mask is stored at position n % 8.
BYTE_MASKS = [(1 << n) - 1 for n in range(8)]
BYTE_MASKS[0] = 255


class PrimitiveProvider:
    # This is the low-level interface which would also be implemented
    # by e.g. CrossHair, by an Atheris-hypothesis integration, etc.
    # We'd then build the structured tree handling, database and replay
    # support, etc. on top of this - so all backends get those for free.
    #
    # See https://github.com/HypothesisWorks/hypothesis/issues/3086

    def __init__(self, conjecturedata: "ConjectureData", /) -> None:
        self._cd = conjecturedata

    def draw_boolean(self, p: float = 0.5, *, forced: Optional[bool] = None) -> bool:
        """Return True with probability p (assuming a uniform generator),
        shrinking towards False. If ``forced`` is set to a non-None value, this
        will always return that value but will write choices appropriate to having
        drawn that value randomly."""
        # Note that this could also be implemented in terms of draw_integer().

        # NB this function is vastly more complicated than it may seem reasonable
        # for it to be. This is because it is used in a lot of places and it's
        # important for it to shrink well, so it's worth the engineering effort.

        if p <= 0 or p >= 1:
            bits = 1
        else:
            # When there is a meaningful draw, in order to shrink well we will
            # set things up so that 0 and 1 always correspond to False and True
            # respectively. This means we want enough bits available that in a
            # draw we will always have at least one truthy value and one falsey
            # value.
            bits = math.ceil(-math.log(min(p, 1 - p), 2))
        # In order to avoid stupidly large draws where the probability is
        # effectively zero or one, we treat probabilities of under 2^-64 to be
        # effectively zero.
        if bits > 64:
            # There isn't enough precision near one for this to occur for values
            # far from 0.
            p = 0.0
            bits = 1

        size = 2**bits

        self._cd.start_example(BIASED_COIN_LABEL)
        while True:
            # The logic here is a bit complicated and special cased to make it
            # play better with the shrinker.

            # We imagine partitioning the real interval [0, 1] into 2**n equal parts
            # and looking at each part and whether its interior is wholly <= p
            # or wholly >= p. At most one part can be neither.

            # We then pick a random part. If it's wholly on one side or the other
            # of p then we use that as the answer. If p is contained in the
            # interval then we start again with a new probability that is given
            # by the fraction of that interval that was <= our previous p.

            # We then take advantage of the fact that we have control of the
            # labelling to make this shrink better, using the following tricks:

            # If p is <= 0 or >= 1 the result of this coin is certain. We make sure
            # to write a byte to the data stream anyway so that these don't cause
            # difficulties when shrinking.
            if p <= 0:
                self._cd.draw_bits(1, forced=0)
                result = False
            elif p >= 1:
                self._cd.draw_bits(1, forced=1)
                result = True
            else:
                falsey = floor(size * (1 - p))
                truthy = floor(size * p)
                remainder = size * p - truthy

                if falsey + truthy == size:
                    partial = False
                else:
                    partial = True

                i = self._cd.draw_bits(
                    bits, forced=None if forced is None else int(forced)
                )

                # We always choose the region that causes us to repeat the loop as
                # the maximum value, so that shrinking the drawn bits never causes
                # us to need to draw more self._cd.
                if partial and i == size - 1:
                    p = remainder
                    continue
                if falsey == 0:
                    # Every other partition is truthy, so the result is true
                    result = True
                elif truthy == 0:
                    # Every other partition is falsey, so the result is false
                    result = False
                elif i <= 1:
                    # We special case so that zero is always false and 1 is always
                    # true which makes shrinking easier because we can always
                    # replace a truthy block with 1. This has the slightly weird
                    # property that shrinking from 2 to 1 can cause the result to
                    # grow, but the shrinker always tries 0 and 1 first anyway, so
                    # this will usually be fine.
                    result = bool(i)
                else:
                    # Originally everything in the region 0 <= i < falsey was false
                    # and everything above was true. We swapped one truthy element
                    # into this region, so the region becomes 0 <= i <= falsey
                    # except for i = 1. We know i > 1 here, so the test for truth
                    # becomes i > falsey.
                    result = i > falsey

            break
        self._cd.stop_example()
        return result

    def draw_integer(
        self,
        min_value: Optional[int] = None,
        max_value: Optional[int] = None,
        *,
        # weights are for choosing an element index from a bounded range
        weights: Optional[Sequence[float]] = None,
        shrink_towards: int = 0,
        forced: Optional[int] = None,
    ) -> int:
        if min_value is not None:
            shrink_towards = max(min_value, shrink_towards)
        if max_value is not None:
            shrink_towards = min(max_value, shrink_towards)

        # This is easy to build on top of our existing conjecture utils,
        # and it's easy to build sampled_from and weighted_coin on this.
        if weights is not None:
            assert min_value is not None
            assert max_value is not None

            sampler = Sampler(weights, observe=False)
            gap = max_value - shrink_towards

            forced_idx = None
            if forced is not None:
                if forced >= shrink_towards:
                    forced_idx = forced - shrink_towards
                else:
                    forced_idx = shrink_towards + gap - forced
            idx = sampler.sample(self._cd, forced=forced_idx)

            # For range -2..2, interpret idx = 0..4 as [0, 1, 2, -1, -2]
            if idx <= gap:
                return shrink_towards + idx
            else:
                return shrink_towards - (idx - gap)

        if min_value is None and max_value is None:
            return self._draw_unbounded_integer(forced=forced)

        if min_value is None:
            assert max_value is not None  # make mypy happy
            probe = max_value + 1
            while max_value < probe:
                self._cd.start_example(ONE_BOUND_INTEGERS_LABEL)
                probe = shrink_towards + self._draw_unbounded_integer(
                    forced=None if forced is None else forced - shrink_towards
                )
                self._cd.stop_example()
            return probe

        if max_value is None:
            assert min_value is not None
            probe = min_value - 1
            while probe < min_value:
                self._cd.start_example(ONE_BOUND_INTEGERS_LABEL)
                probe = shrink_towards + self._draw_unbounded_integer(
                    forced=None if forced is None else forced - shrink_towards
                )
                self._cd.stop_example()
            return probe

        return self._draw_bounded_integer(
            min_value,
            max_value,
            center=shrink_towards,
            forced=forced,
        )

    def draw_float(
        self,
        *,
        min_value: float = -math.inf,
        max_value: float = math.inf,
        allow_nan: bool = True,
        smallest_nonzero_magnitude: float,
        # TODO: consider supporting these float widths at the IR level in the
        # future.
        # width: Literal[16, 32, 64] = 64,
        # exclude_min and exclude_max handled higher up,
        forced: Optional[float] = None,
    ) -> float:
        (
            sampler,
            forced_sign_bit,
            neg_clamper,
            pos_clamper,
            nasty_floats,
        ) = self._draw_float_init_logic(
            min_value=min_value,
            max_value=max_value,
            allow_nan=allow_nan,
            smallest_nonzero_magnitude=smallest_nonzero_magnitude,
        )

        while True:
            self._cd.start_example(FLOAT_STRATEGY_DO_DRAW_LABEL)
            # If `forced in nasty_floats`, then `forced` was *probably*
            # generated by drawing a nonzero index from the sampler. However, we
            # have no obligation to generate it that way when forcing. In particular,
            # i == 0 is able to produce all possible floats, and the forcing
            # logic is simpler if we assume this choice.
            forced_i = None if forced is None else 0
            i = sampler.sample(self._cd, forced=forced_i) if sampler else 0
            self._cd.start_example(DRAW_FLOAT_LABEL)
            if i == 0:
                result = self._draw_float(
                    forced_sign_bit=forced_sign_bit, forced=forced
                )
                if math.copysign(1.0, result) == -1:
                    assert neg_clamper is not None
                    clamped = -neg_clamper(-result)
                else:
                    assert pos_clamper is not None
                    clamped = pos_clamper(result)
                if clamped != result and not (math.isnan(result) and allow_nan):
                    self._cd.stop_example()
                    self._cd.start_example(DRAW_FLOAT_LABEL)
                    self._draw_float(forced=clamped)
                    result = clamped
            else:
                result = nasty_floats[i - 1]

                self._draw_float(forced=result)

            self._cd.stop_example()  # (DRAW_FLOAT_LABEL)
            self._cd.stop_example()  # (FLOAT_STRATEGY_DO_DRAW_LABEL)
            return result

    def draw_string(
        self,
        intervals: IntervalSet,
        *,
        min_size: int = 0,
        max_size: Optional[int] = None,
        forced: Optional[str] = None,
    ) -> str:
        if max_size is None:
            max_size = DRAW_STRING_DEFAULT_MAX_SIZE

        assert forced is None or min_size <= len(forced) <= max_size

        average_size = min(
            max(min_size * 2, min_size + 5),
            0.5 * (min_size + max_size),
        )

        chars = []
        elements = many(
            self._cd,
            min_size=min_size,
            max_size=max_size,
            average_size=average_size,
            forced=None if forced is None else len(forced),
            observe=False,
        )
        while elements.more():
            forced_i: Optional[int] = None
            if forced is not None:
                c = forced[elements.count - 1]
                forced_i = intervals.index_from_char_in_shrink_order(c)

            if len(intervals) > 256:
                if self.draw_boolean(
                    0.2, forced=None if forced_i is None else forced_i > 255
                ):
                    i = self._draw_bounded_integer(
                        256, len(intervals) - 1, forced=forced_i
                    )
                else:
                    i = self._draw_bounded_integer(0, 255, forced=forced_i)
            else:
                i = self._draw_bounded_integer(0, len(intervals) - 1, forced=forced_i)

            chars.append(intervals.char_in_shrink_order(i))

        return "".join(chars)

    def draw_bytes(self, size: int, *, forced: Optional[bytes] = None) -> bytes:
        forced_i = None
        if forced is not None:
            forced_i = int_from_bytes(forced)
            size = len(forced)

        return self._cd.draw_bits(8 * size, forced=forced_i).to_bytes(size, "big")

    def _draw_float(
        self, forced_sign_bit: Optional[int] = None, *, forced: Optional[float] = None
    ) -> float:
        """
        Helper for draw_float which draws a random 64-bit float.
        """
        if forced is not None:
            # sign_aware_lte(forced, -0.0) does not correctly handle the
            # math.nan case here.
            forced_sign_bit = math.copysign(1, forced) == -1
        is_negative = self._cd.draw_bits(1, forced=forced_sign_bit)
        f = lex_to_float(
            self._cd.draw_bits(
                64, forced=None if forced is None else float_to_lex(abs(forced))
            )
        )
        return -f if is_negative else f

    def _draw_unbounded_integer(self, *, forced: Optional[int] = None) -> int:
        forced_i = None
        if forced is not None:
            # Using any bucket large enough to contain this integer would be a
            # valid way to force it. This is because an n bit integer could have
            # been drawn from a bucket of size n, or from any bucket of size
            # m > n.
            # We'll always choose the smallest eligible bucket here.

            # We need an extra bit to handle forced signed integers. INT_SIZES
            # is interpreted as unsigned sizes.
            bit_size = forced.bit_length() + 1
            size = min(size for size in INT_SIZES if bit_size <= size)
            forced_i = INT_SIZES.index(size)

        size = INT_SIZES[INT_SIZES_SAMPLER.sample(self._cd, forced=forced_i)]

        forced_r = None
        if forced is not None:
            forced_r = forced
            forced_r <<= 1
            if forced < 0:
                forced_r = -forced_r
                forced_r |= 1

        r = self._cd.draw_bits(size, forced=forced_r)
        sign = r & 1
        r >>= 1
        if sign:
            r = -r
        return r

    def _draw_bounded_integer(
        self,
        lower: int,
        upper: int,
        *,
        center: Optional[int] = None,
        forced: Optional[int] = None,
    ) -> int:
        assert lower <= upper
        assert forced is None or lower <= forced <= upper
        if lower == upper:
            # Write a value even when this is trivial so that when a bound depends
            # on other values we don't suddenly disappear when the gap shrinks to
            # zero - if that happens then often the data stream becomes misaligned
            # and we fail to shrink in cases where we really should be able to.
            self._cd.draw_bits(1, forced=0)
            return int(lower)

        if center is None:
            center = lower
        center = min(max(center, lower), upper)

        if center == upper:
            above = False
        elif center == lower:
            above = True
        else:
            force_above = None if forced is None else forced < center
            above = not self._cd.draw_bits(1, forced=force_above)

        if above:
            gap = upper - center
        else:
            gap = center - lower

        assert gap > 0

        bits = gap.bit_length()
        probe = gap + 1

        if bits > 24 and self.draw_boolean(
            7 / 8, forced=None if forced is None else False
        ):
            # For large ranges, we combine the uniform random distribution from draw_bits
            # with a weighting scheme with moderate chance.  Cutoff at 2 ** 24 so that our
            # choice of unicode characters is uniform but the 32bit distribution is not.
            idx = INT_SIZES_SAMPLER.sample(self._cd)
            bits = min(bits, INT_SIZES[idx])

        while probe > gap:
            self._cd.start_example(INTEGER_RANGE_DRAW_LABEL)
            probe = self._cd.draw_bits(
                bits, forced=None if forced is None else abs(forced - center)
            )
            self._cd.stop_example()

        if above:
            result = center + probe
        else:
            result = center - probe

        assert lower <= result <= upper
        assert forced is None or result == forced, (result, forced, center, above)
        return result

    @classmethod
    def _draw_float_init_logic(
        cls,
        *,
        min_value: float,
        max_value: float,
        allow_nan: bool,
        smallest_nonzero_magnitude: float,
    ) -> Tuple[
        Optional[Sampler],
        Optional[Literal[0, 1]],
        Optional[Callable[[float], float]],
        Optional[Callable[[float], float]],
        List[float],
    ]:
        """
        Caches initialization logic for draw_float, as an alternative to
        computing this for *every* float draw.
        """
        # float_to_int allows us to distinguish between e.g. -0.0 and 0.0,
        # even in light of hash(-0.0) == hash(0.0) and -0.0 == 0.0.
        key = (
            float_to_int(min_value),
            float_to_int(max_value),
            allow_nan,
            float_to_int(smallest_nonzero_magnitude),
        )
        if key in FLOAT_INIT_LOGIC_CACHE:
            return FLOAT_INIT_LOGIC_CACHE[key]

        result = cls._compute_draw_float_init_logic(
            min_value=min_value,
            max_value=max_value,
            allow_nan=allow_nan,
            smallest_nonzero_magnitude=smallest_nonzero_magnitude,
        )
        FLOAT_INIT_LOGIC_CACHE[key] = result
        return result

    @staticmethod
    def _compute_draw_float_init_logic(
        *,
        min_value: float,
        max_value: float,
        allow_nan: bool,
        smallest_nonzero_magnitude: float,
    ) -> Tuple[
        Optional[Sampler],
        Optional[Literal[0, 1]],
        Optional[Callable[[float], float]],
        Optional[Callable[[float], float]],
        List[float],
    ]:
        if smallest_nonzero_magnitude == 0.0:  # pragma: no cover
            raise FloatingPointError(
                "Got allow_subnormal=True, but we can't represent subnormal floats "
                "right now, in violation of the IEEE-754 floating-point "
                "specification.  This is usually because something was compiled with "
                "-ffast-math or a similar option, which sets global processor state.  "
                "See https://simonbyrne.github.io/notes/fastmath/ for a more detailed "
                "writeup - and good luck!"
            )

        def permitted(f):
            assert isinstance(f, float)
            if math.isnan(f):
                return allow_nan
            if 0 < abs(f) < smallest_nonzero_magnitude:
                return False
            return sign_aware_lte(min_value, f) and sign_aware_lte(f, max_value)

        boundary_values = [
            min_value,
            next_up(min_value),
            min_value + 1,
            max_value - 1,
            next_down(max_value),
            max_value,
        ]
        nasty_floats = [f for f in NASTY_FLOATS + boundary_values if permitted(f)]
        weights = [0.2 * len(nasty_floats)] + [0.8] * len(nasty_floats)
        sampler = Sampler(weights, observe=False) if nasty_floats else None

        pos_clamper = neg_clamper = None
        if sign_aware_lte(0.0, max_value):
            pos_min = max(min_value, smallest_nonzero_magnitude)
            allow_zero = sign_aware_lte(min_value, 0.0)
            pos_clamper = make_float_clamper(pos_min, max_value, allow_zero=allow_zero)
        if sign_aware_lte(min_value, -0.0):
            neg_max = min(max_value, -smallest_nonzero_magnitude)
            allow_zero = sign_aware_lte(-0.0, max_value)
            neg_clamper = make_float_clamper(
                -neg_max, -min_value, allow_zero=allow_zero
            )

        forced_sign_bit: Optional[Literal[0, 1]] = None
        if (pos_clamper is None) != (neg_clamper is None):
            forced_sign_bit = 1 if neg_clamper else 0

        return (sampler, forced_sign_bit, neg_clamper, pos_clamper, nasty_floats)


class ConjectureData:
    @classmethod
    def for_buffer(
        cls,
        buffer: Union[List[int], bytes],
        observer: Optional[DataObserver] = None,
    ) -> "ConjectureData":
        return cls(len(buffer), buffer, random=None, observer=observer)

    def __init__(
        self,
        max_length: int,
        prefix: Union[List[int], bytes, bytearray],
        random: Optional[Random],
        observer: Optional[DataObserver] = None,
    ) -> None:
        if observer is None:
            observer = DataObserver()
        assert isinstance(observer, DataObserver)
        self.__bytes_drawn = 0
        self.observer = observer
        self.max_length = max_length
        self.is_find = False
        self.overdraw = 0
        self.__prefix = bytes(prefix)
        self.__random = random

        assert random is not None or max_length <= len(prefix)

        self.blocks = Blocks(self)
        self.buffer: "Union[bytes, bytearray]" = bytearray()
        self.index = 0
        self.output = ""
        self.status = Status.VALID
        self.frozen = False
        global global_test_counter
        self.testcounter = global_test_counter
        global_test_counter += 1
        self.start_time = time.perf_counter()
        self.events: Dict[str, Union[str, int, float]] = {}
        self.forced_indices: "Set[int]" = set()
        self.interesting_origin: Optional[InterestingOrigin] = None
        self.draw_times: "Dict[str, float]" = {}
        self._stateful_run_times: "DefaultDict[str, float]" = defaultdict(float)
        self.max_depth = 0
        self.has_discards = False
        self.provider = PrimitiveProvider(self)

        self.__result: "Optional[ConjectureResult]" = None

        # Observations used for targeted search.  They'll be aggregated in
        # ConjectureRunner.generate_new_examples and fed to TargetSelector.
        self.target_observations: TargetObservations = {}

        # Tags which indicate something about which part of the search space
        # this example is in. These are used to guide generation.
        self.tags: "Set[StructuralCoverageTag]" = set()
        self.labels_for_structure_stack: "List[Set[int]]" = []

        # Normally unpopulated but we need this in the niche case
        # that self.as_result() is Overrun but we still want the
        # examples for reporting purposes.
        self.__examples: "Optional[Examples]" = None

        # We want the top level example to have depth 0, so we start
        # at -1.
        self.depth = -1
        self.__example_record = ExampleRecord()

        # Slice indices for discrete reportable parts that which-parts-matter can
        # try varying, to report if the minimal example always fails anyway.
        self.arg_slices: Set[Tuple[int, int]] = set()
        self.slice_comments: Dict[Tuple[int, int], str] = {}
        self._observability_args: Dict[str, Any] = {}
        self._observability_predicates: defaultdict = defaultdict(
            lambda: {"satisfied": 0, "unsatisfied": 0}
        )

        self.extra_information = ExtraInformation()

        self.start_example(TOP_LABEL)

    def __repr__(self):
        return "ConjectureData(%s, %d bytes%s)" % (
            self.status.name,
            len(self.buffer),
            ", frozen" if self.frozen else "",
        )

    # A bit of explanation of the `observe` argument in our draw_* functions.
    #
    # There are two types of draws: sub-ir and super-ir. For instance, some ir
    # nodes use `many`, which in turn calls draw_boolean. But some strategies
    # also use many, at the super-ir level. We don't want to write sub-ir draws
    # to the DataTree (and consequently use them when computing novel prefixes),
    # since they are fully recorded by writing the ir node itself.
    # But super-ir draws are not included in the ir node, so we do want to write
    # these to the tree.
    #
    # `observe` formalizes this distinction. The draw will only be written to
    # the DataTree if observe is True.

    def draw_integer(
        self,
        min_value: Optional[int] = None,
        max_value: Optional[int] = None,
        *,
        # weights are for choosing an element index from a bounded range
        weights: Optional[Sequence[float]] = None,
        shrink_towards: int = 0,
        forced: Optional[int] = None,
        observe: bool = True,
    ) -> int:
        # Validate arguments
        if weights is not None:
            assert min_value is not None
            assert max_value is not None
            width = max_value - min_value + 1
            assert width <= 255  # arbitrary practical limit
            assert len(weights) == width

        if forced is not None and (min_value is None or max_value is None):
            # We draw `forced=forced - shrink_towards` here internally. If that
            # grows larger than a 128 bit signed integer, we can't represent it.
            # Disallow this combination for now.
            # Note that bit_length() = 128 -> signed bit size = 129.
            assert (forced - shrink_towards).bit_length() < 128
        if forced is not None and min_value is not None:
            assert min_value <= forced
        if forced is not None and max_value is not None:
            assert forced <= max_value

        kwargs: IntegerKWargs = {
            "min_value": min_value,
            "max_value": max_value,
            "weights": weights,
            "shrink_towards": shrink_towards,
        }
        value = self.provider.draw_integer(**kwargs, forced=forced)
        if observe:
            self.observer.draw_integer(
                value, was_forced=forced is not None, kwargs=kwargs
            )
        return value

    def draw_float(
        self,
        min_value: float = -math.inf,
        max_value: float = math.inf,
        *,
        allow_nan: bool = True,
        smallest_nonzero_magnitude: float = SMALLEST_SUBNORMAL,
        # TODO: consider supporting these float widths at the IR level in the
        # future.
        # width: Literal[16, 32, 64] = 64,
        # exclude_min and exclude_max handled higher up,
        forced: Optional[float] = None,
        observe: bool = True,
    ) -> float:
        assert smallest_nonzero_magnitude > 0
        assert not math.isnan(min_value)
        assert not math.isnan(max_value)

        if forced is not None:
            assert allow_nan or not math.isnan(forced)
            assert math.isnan(forced) or (
                sign_aware_lte(min_value, forced) and sign_aware_lte(forced, max_value)
            )

        kwargs: FloatKWargs = {
            "min_value": min_value,
            "max_value": max_value,
            "allow_nan": allow_nan,
            "smallest_nonzero_magnitude": smallest_nonzero_magnitude,
        }
        value = self.provider.draw_float(**kwargs, forced=forced)
        if observe:
            self.observer.draw_float(
                value, kwargs=kwargs, was_forced=forced is not None
            )
        return value

    def draw_string(
        self,
        intervals: IntervalSet,
        *,
        min_size: int = 0,
        max_size: Optional[int] = None,
        forced: Optional[str] = None,
        observe: bool = True,
    ) -> str:
        assert forced is None or min_size <= len(forced)

        kwargs: StringKWargs = {
            "intervals": intervals,
            "min_size": min_size,
            "max_size": max_size,
        }
        value = self.provider.draw_string(**kwargs, forced=forced)
        if observe:
            self.observer.draw_string(
                value, kwargs=kwargs, was_forced=forced is not None
            )
        return value

    def draw_bytes(
        self,
        # TODO move to min_size and max_size here.
        size: int,
        *,
        forced: Optional[bytes] = None,
        observe: bool = True,
    ) -> bytes:
        assert forced is None or len(forced) == size
        assert size >= 0

        kwargs: BytesKWargs = {"size": size}
        value = self.provider.draw_bytes(**kwargs, forced=forced)
        if observe:
            self.observer.draw_bytes(
                value, kwargs=kwargs, was_forced=forced is not None
            )
        return value

    def draw_boolean(
        self, p: float = 0.5, *, forced: Optional[bool] = None, observe: bool = True
    ) -> bool:
        # Internally, we treat probabilities lower than 1 / 2**64 as
        # unconditionally false.
        #
        # Note that even if we lift this 64 bit restriction in the future, p
        # cannot be 0 (1) when forced is True (False).
        if forced is True:
            assert p > 2 ** (-64)
        if forced is False:
            assert p < (1 - 2 ** (-64))

        kwargs: BooleanKWargs = {"p": p}
        value = self.provider.draw_boolean(**kwargs, forced=forced)
        if observe:
            self.observer.draw_boolean(
                value, kwargs=kwargs, was_forced=forced is not None
            )
        return value

    def as_result(self) -> Union[ConjectureResult, _Overrun]:
        """Convert the result of running this test into
        either an Overrun object or a ConjectureResult."""

        assert self.frozen
        if self.status == Status.OVERRUN:
            return Overrun
        if self.__result is None:
            self.__result = ConjectureResult(
                status=self.status,
                interesting_origin=self.interesting_origin,
                buffer=self.buffer,
                examples=self.examples,
                blocks=self.blocks,
                output=self.output,
                extra_information=(
                    self.extra_information
                    if self.extra_information.has_information()
                    else None
                ),
                has_discards=self.has_discards,
                target_observations=self.target_observations,
                tags=frozenset(self.tags),
                forced_indices=frozenset(self.forced_indices),
                arg_slices=self.arg_slices,
                slice_comments=self.slice_comments,
            )
            assert self.__result is not None
            self.blocks.transfer_ownership(self.__result)
        return self.__result

    def __assert_not_frozen(self, name: str) -> None:
        if self.frozen:
            raise Frozen(f"Cannot call {name} on frozen ConjectureData")

    def note(self, value: Any) -> None:
        self.__assert_not_frozen("note")
        if not isinstance(value, str):
            value = repr(value)
        self.output += value

    def draw(
        self,
        strategy: "SearchStrategy[Ex]",
        label: Optional[int] = None,
        observe_as: Optional[str] = None,
    ) -> "Ex":
        if self.is_find and not strategy.supports_find:
            raise InvalidArgument(
                f"Cannot use strategy {strategy!r} within a call to find "
                "(presumably because it would be invalid after the call had ended)."
            )

        at_top_level = self.depth == 0
        start_time = None
        if at_top_level:
            # We start this timer early, because accessing attributes on a LazyStrategy
            # can be almost arbitrarily slow.  In cases like characters() and text()
            # where we cache something expensive, this led to Flaky deadline errors!
            # See https://github.com/HypothesisWorks/hypothesis/issues/2108
            start_time = time.perf_counter()

        strategy.validate()

        if strategy.is_empty:
            self.mark_invalid("strategy is empty")

        if self.depth >= MAX_DEPTH:
            self.mark_invalid("max depth exceeded")

        if label is None:
            assert isinstance(strategy.label, int)
            label = strategy.label
        self.start_example(label=label)
        try:
            if not at_top_level:
                return strategy.do_draw(self)
            else:
                assert start_time is not None
                strategy.validate()
                try:
                    return strategy.do_draw(self)
                finally:
                    key = observe_as or f"generate:unlabeled_{len(self.draw_times)}"
                    self.draw_times[key] = time.perf_counter() - start_time
        finally:
            self.stop_example()

    def start_example(self, label: int) -> None:
        self.__assert_not_frozen("start_example")
        self.depth += 1
        # Logically it would make sense for this to just be
        # ``self.depth = max(self.depth, self.max_depth)``, which is what it used to
        # be until we ran the code under tracemalloc and found a rather significant
        # chunk of allocation was happening here. This was presumably due to varargs
        # or the like, but we didn't investigate further given that it was easy
        # to fix with this check.
        if self.depth > self.max_depth:
            self.max_depth = self.depth
        self.__example_record.start_example(label)
        self.labels_for_structure_stack.append({label})

    def stop_example(self, *, discard: bool = False) -> None:
        if self.frozen:
            return
        if discard:
            self.has_discards = True
        self.depth -= 1
        assert self.depth >= -1
        self.__example_record.stop_example(discard=discard)

        labels_for_structure = self.labels_for_structure_stack.pop()

        if not discard:
            if self.labels_for_structure_stack:
                self.labels_for_structure_stack[-1].update(labels_for_structure)
            else:
                self.tags.update([structural_coverage(l) for l in labels_for_structure])

        if discard:
            # Once we've discarded an example, every test case starting with
            # this prefix contains discards. We prune the tree at that point so
            # as to avoid future test cases bothering with this region, on the
            # assumption that some example that you could have used instead
            # there would *not* trigger the discard. This greatly speeds up
            # test case generation in some cases, because it allows us to
            # ignore large swathes of the search space that are effectively
            # redundant.
            #
            # A scenario that can cause us problems but which we deliberately
            # have decided not to support is that if there are side effects
            # during data generation then you may end up with a scenario where
            # every good test case generates a discard because the discarded
            # section sets up important things for later. This is not terribly
            # likely and all that you see in this case is some degradation in
            # quality of testing, so we don't worry about it.
            #
            # Note that killing the branch does *not* mean we will never
            # explore below this point, and in particular we may do so during
            # shrinking. Any explicit request for a data object that starts
            # with the branch here will work just fine, but novel prefix
            # generation will avoid it, and we can use it to detect when we
            # have explored the entire tree (up to redundancy).

            self.observer.kill_branch()

    @property
    def examples(self) -> Examples:
        assert self.frozen
        if self.__examples is None:
            self.__examples = Examples(record=self.__example_record, blocks=self.blocks)
        return self.__examples

    def freeze(self) -> None:
        if self.frozen:
            assert isinstance(self.buffer, bytes)
            return
        self.finish_time = time.perf_counter()
        assert len(self.buffer) == self.index

        # Always finish by closing all remaining examples so that we have a
        # valid tree.
        while self.depth >= 0:
            self.stop_example()

        self.__example_record.freeze()

        self.frozen = True

        self.buffer = bytes(self.buffer)
        self.observer.conclude_test(self.status, self.interesting_origin)

    def choice(
        self,
        values: Sequence[T],
        *,
        forced: Optional[T] = None,
        observe: bool = True,
    ) -> T:
        forced_i = None if forced is None else values.index(forced)
        i = self.draw_integer(0, len(values) - 1, forced=forced_i, observe=observe)
        return values[i]

    def draw_bits(self, n: int, *, forced: Optional[int] = None) -> int:
        """Return an ``n``-bit integer from the underlying source of
        bytes. If ``forced`` is set to an integer will instead
        ignore the underlying source and simulate a draw as if it had
        returned that integer."""
        self.__assert_not_frozen("draw_bits")
        if n == 0:
            return 0
        assert n > 0
        n_bytes = bits_to_bytes(n)
        self.__check_capacity(n_bytes)

        if forced is not None:
            buf = int_to_bytes(forced, n_bytes)
        elif self.__bytes_drawn < len(self.__prefix):
            index = self.__bytes_drawn
            buf = self.__prefix[index : index + n_bytes]
            if len(buf) < n_bytes:
                assert self.__random is not None
                buf += uniform(self.__random, n_bytes - len(buf))
        else:
            assert self.__random is not None
            buf = uniform(self.__random, n_bytes)
        buf = bytearray(buf)
        self.__bytes_drawn += n_bytes

        assert len(buf) == n_bytes

        # If we have a number of bits that is not a multiple of 8
        # we have to mask off the high bits.
        buf[0] &= BYTE_MASKS[n % 8]
        buf = bytes(buf)
        result = int_from_bytes(buf)

        self.__example_record.draw_bits(n, forced)

        initial = self.index

        assert isinstance(self.buffer, bytearray)
        self.buffer.extend(buf)
        self.index = len(self.buffer)

        if forced is not None:
            self.forced_indices.update(range(initial, self.index))

        self.blocks.add_endpoint(self.index)

        assert result.bit_length() <= n
        return result

    def __check_capacity(self, n: int) -> None:
        if self.index + n > self.max_length:
            self.mark_overrun()

    def conclude_test(
        self,
        status: Status,
        interesting_origin: Optional[InterestingOrigin] = None,
    ) -> NoReturn:
        assert (interesting_origin is None) or (status == Status.INTERESTING)
        self.__assert_not_frozen("conclude_test")
        self.interesting_origin = interesting_origin
        self.status = status
        self.freeze()
        raise StopTest(self.testcounter)

    def mark_interesting(
        self, interesting_origin: Optional[InterestingOrigin] = None
    ) -> NoReturn:
        self.conclude_test(Status.INTERESTING, interesting_origin)

    def mark_invalid(self, why: Optional[str] = None) -> NoReturn:
        if why is not None:
            self.events["invalid because"] = why
        self.conclude_test(Status.INVALID)

    def mark_overrun(self) -> NoReturn:
        self.conclude_test(Status.OVERRUN)


def bits_to_bytes(n: int) -> int:
    """The number of bytes required to represent an n-bit number.
    Equivalent to (n + 7) // 8, but slightly faster. This really is
    called enough times that that matters."""
    return (n + 7) >> 3
