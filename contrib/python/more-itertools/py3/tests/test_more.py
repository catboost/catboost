import cmath
import warnings

from collections import Counter, abc
from collections.abc import Set
from datetime import datetime, timedelta
from decimal import Decimal
from doctest import DocTestSuite
from fractions import Fraction
from functools import partial, reduce
from io import StringIO
from itertools import (
    accumulate,
    chain,
    combinations,
    combinations_with_replacement,
    count,
    cycle,
    groupby,
    islice,
    permutations,
    product,
    repeat,
)
from operator import add, mul, itemgetter
from pickle import loads, dumps
from random import Random, random, randrange, seed
from statistics import mean
from string import ascii_letters
from sys import version_info
from time import sleep
from traceback import format_exc
from unittest import skipIf, TestCase

import more_itertools as mi


def load_tests(loader, tests, ignore):
    # Add the doctests
    tests.addTests(DocTestSuite('more_itertools.more'))
    return tests


class ChunkedTests(TestCase):
    """Tests for ``chunked()``"""

    def test_even(self):
        """Test when ``n`` divides evenly into the length of the iterable."""
        self.assertEqual(
            list(mi.chunked('ABCDEF', 3)), [['A', 'B', 'C'], ['D', 'E', 'F']]
        )

    def test_odd(self):
        """Test when ``n`` does not divide evenly into the length of the
        iterable.

        """
        self.assertEqual(
            list(mi.chunked('ABCDE', 3)), [['A', 'B', 'C'], ['D', 'E']]
        )

    def test_none(self):
        """Test when ``n`` has the value ``None``."""
        self.assertEqual(
            list(mi.chunked('ABCDE', None)), [['A', 'B', 'C', 'D', 'E']]
        )

    def test_strict_false(self):
        """Test when ``n`` does not divide evenly into the length of the
        iterable and strict is false.

        """
        self.assertEqual(
            list(mi.chunked('ABCDE', 3, strict=False)),
            [['A', 'B', 'C'], ['D', 'E']],
        )

    def test_strict_being_true(self):
        """Test when ``n`` does not divide evenly into the length of the
        iterable and strict is True (raising an exception).

        """

        def f():
            return list(mi.chunked('ABCDE', 3, strict=True))

        self.assertRaisesRegex(ValueError, "iterable is not divisible by n", f)
        self.assertEqual(
            list(mi.chunked('ABCDEF', 3, strict=True)),
            [['A', 'B', 'C'], ['D', 'E', 'F']],
        )

    def test_strict_being_true_with_size_none(self):
        """Test when ``n`` has value ``None`` and the keyword strict is True
        (raising an exception).

        """

        def f():
            return list(mi.chunked('ABCDE', None, strict=True))

        self.assertRaisesRegex(
            ValueError, "n must not be None when using strict mode.", f
        )


class FirstTests(TestCase):
    def test_many(self):
        # Also try it on a generator expression to make sure it works on
        # whatever those return, across Python versions.
        self.assertEqual(mi.first(x for x in range(4)), 0)

    def test_one(self):
        self.assertEqual(mi.first([3]), 3)

    def test_empty(self):
        with self.assertRaises(ValueError):
            mi.first([])

    def test_default(self):
        self.assertEqual(mi.first([], 'boo'), 'boo')


class IterOnlyRange:
    """User-defined iterable class which only support __iter__.

    >>> r = IterOnlyRange(5)
    >>> r[0]  # doctest: +SKIP
    AttributeError: IterOnlyRange instance has no attribute '__getitem__'

    Note: In Python 3, ``TypeError`` will be raised because ``object`` is
    inherited implicitly by default.

    >>> r[0]  # doctest: +SKIP
    TypeError: 'IterOnlyRange' object does not support indexing
    """

    def __init__(self, n):
        """Set the length of the range."""
        self.n = n

    def __iter__(self):
        """Works same as range()."""
        return iter(range(self.n))


class LastTests(TestCase):
    def test_basic(self):
        cases = [
            (range(4), 3),
            (iter(range(4)), 3),
            (range(1), 0),
            (iter(range(1)), 0),
            (IterOnlyRange(5), 4),
            ({n: str(n) for n in range(5)}, 4),
            ({0: '0', -1: '-1', 2: '-2'}, 2),
        ]

        for iterable, expected in cases:
            with self.subTest(iterable=iterable):
                self.assertEqual(mi.last(iterable), expected)

    def test_default(self):
        for iterable, default, expected in [
            (range(1), None, 0),
            ([], None, None),
            ({}, None, None),
            (iter([]), None, None),
        ]:
            with self.subTest(args=(iterable, default)):
                self.assertEqual(mi.last(iterable, default=default), expected)

    def test_empty(self):
        for iterable in ([], iter(range(0))):
            with self.subTest(iterable=iterable):
                with self.assertRaises(ValueError):
                    mi.last(iterable)


class NthOrLastTests(TestCase):
    """Tests for ``nth_or_last()``"""

    def test_basic(self):
        self.assertEqual(mi.nth_or_last(range(3), 1), 1)
        self.assertEqual(mi.nth_or_last(range(3), 3), 2)

    def test_default_value(self):
        default = 42
        self.assertEqual(mi.nth_or_last(range(0), 3, default), default)

    def test_empty_iterable_no_default(self):
        self.assertRaises(ValueError, lambda: mi.nth_or_last(range(0), 0))


class PeekableMixinTests:
    """Common tests for ``peekable()`` and ``seekable()`` behavior"""

    cls = None

    def test_passthrough(self):
        """Iterating a peekable without using ``peek()`` or ``prepend()``
        should just give the underlying iterable's elements (a trivial test but
        useful to set a baseline in case something goes wrong)"""
        expected = [1, 2, 3, 4, 5]
        actual = list(self.cls(expected))
        self.assertEqual(actual, expected)

    def test_peek_default(self):
        """Make sure passing a default into ``peek()`` works."""
        p = self.cls([])
        self.assertEqual(p.peek(7), 7)

    def test_truthiness(self):
        """Make sure a ``peekable`` tests true iff there are items remaining in
        the iterable.

        """
        p = self.cls([])
        self.assertFalse(p)

        p = self.cls(range(3))
        self.assertTrue(p)

    def test_simple_peeking(self):
        """Make sure ``next`` and ``peek`` advance and don't advance the
        iterator, respectively.

        """
        p = self.cls(range(10))
        self.assertEqual(next(p), 0)
        self.assertEqual(p.peek(), 1)
        self.assertEqual(p.peek(), 1)
        self.assertEqual(next(p), 1)


class PeekableTests(PeekableMixinTests, TestCase):
    cls = mi.peekable

    def test_indexing(self):
        """
        Indexing into the peekable shouldn't advance the iterator.
        """
        p = mi.peekable('abcdefghijkl')

        # The 0th index is what ``next()`` will return
        self.assertEqual(p[0], 'a')
        self.assertEqual(next(p), 'a')

        # Indexing further into the peekable shouldn't advance the iterator
        self.assertEqual(p[2], 'd')
        self.assertEqual(next(p), 'b')

        # The 0th index moves up with the iterator; the last index follows
        self.assertEqual(p[0], 'c')
        self.assertEqual(p[9], 'l')

        self.assertEqual(next(p), 'c')
        self.assertEqual(p[8], 'l')

        # Negative indexing should work too
        self.assertEqual(p[-2], 'k')
        self.assertEqual(p[-9], 'd')
        self.assertRaises(IndexError, lambda: p[-10])

    def test_slicing(self):
        """Slicing the peekable shouldn't advance the iterator."""
        seq = list('abcdefghijkl')
        p = mi.peekable(seq)

        # Slicing the peekable should just be like slicing a re-iterable
        self.assertEqual(p[1:4], seq[1:4])

        # Advancing the iterator moves the slices up also
        self.assertEqual(next(p), 'a')
        self.assertEqual(p[1:4], seq[1:][1:4])

        # Implicit starts and stop should work
        self.assertEqual(p[:5], seq[1:][:5])
        self.assertEqual(p[:], seq[1:][:])

        # Indexing past the end should work
        self.assertEqual(p[:100], seq[1:][:100])

        # Steps should work, including negative
        self.assertEqual(p[::2], seq[1:][::2])
        self.assertEqual(p[::-1], seq[1:][::-1])

    def test_slicing_reset(self):
        """Test slicing on a fresh iterable each time"""
        iterable = ['0', '1', '2', '3', '4', '5']
        indexes = list(range(-4, len(iterable) + 4)) + [None]
        steps = [1, 2, 3, 4, -1, -2, -3, 4]
        for slice_args in product(indexes, indexes, steps):
            it = iter(iterable)
            p = mi.peekable(it)
            next(p)
            index = slice(*slice_args)
            actual = p[index]
            expected = iterable[1:][index]
            self.assertEqual(actual, expected, slice_args)

    def test_slicing_error(self):
        iterable = '01234567'
        p = mi.peekable(iter(iterable))

        # Prime the cache
        p.peek()
        old_cache = list(p._cache)

        # Illegal slice
        with self.assertRaises(ValueError):
            p[1:-1:0]

        # Neither the cache nor the iteration should be affected
        self.assertEqual(old_cache, list(p._cache))
        self.assertEqual(list(p), list(iterable))

    # prepend() behavior tests

    def test_prepend(self):
        """Tests interspersed ``prepend()`` and ``next()`` calls"""
        it = mi.peekable(range(2))
        actual = []

        # Test prepend() before next()
        it.prepend(10)
        actual += [next(it), next(it)]

        # Test prepend() between next()s
        it.prepend(11)
        actual += [next(it), next(it)]

        # Test prepend() after source iterable is consumed
        it.prepend(12)
        actual += [next(it)]

        expected = [10, 0, 11, 1, 12]
        self.assertEqual(actual, expected)

    def test_multi_prepend(self):
        """Tests prepending multiple items and getting them in proper order"""
        it = mi.peekable(range(5))
        actual = [next(it), next(it)]
        it.prepend(10, 11, 12)
        it.prepend(20, 21)
        actual += list(it)
        expected = [0, 1, 20, 21, 10, 11, 12, 2, 3, 4]
        self.assertEqual(actual, expected)

    def test_empty(self):
        """Tests prepending in front of an empty iterable"""
        it = mi.peekable([])
        it.prepend(10)
        actual = list(it)
        expected = [10]
        self.assertEqual(actual, expected)

    def test_prepend_truthiness(self):
        """Tests that ``__bool__()`` or ``__nonzero__()`` works properly
        with ``prepend()``"""
        it = mi.peekable(range(5))
        self.assertTrue(it)
        actual = list(it)
        self.assertFalse(it)
        it.prepend(10)
        self.assertTrue(it)
        actual += [next(it)]
        self.assertFalse(it)
        expected = [0, 1, 2, 3, 4, 10]
        self.assertEqual(actual, expected)

    def test_multi_prepend_peek(self):
        """Tests prepending multiple elements and getting them in reverse order
        while peeking"""
        it = mi.peekable(range(5))
        actual = [next(it), next(it)]
        self.assertEqual(it.peek(), 2)
        it.prepend(10, 11, 12)
        self.assertEqual(it.peek(), 10)
        it.prepend(20, 21)
        self.assertEqual(it.peek(), 20)
        actual += list(it)
        self.assertFalse(it)
        expected = [0, 1, 20, 21, 10, 11, 12, 2, 3, 4]
        self.assertEqual(actual, expected)

    def test_prepend_after_stop(self):
        """Test resuming iteration after a previous exhaustion"""
        it = mi.peekable(range(3))
        self.assertEqual(list(it), [0, 1, 2])
        self.assertRaises(StopIteration, lambda: next(it))
        it.prepend(10)
        self.assertEqual(next(it), 10)
        self.assertRaises(StopIteration, lambda: next(it))

    def test_prepend_slicing(self):
        """Tests interaction between prepending and slicing"""
        seq = list(range(20))
        p = mi.peekable(seq)

        p.prepend(30, 40, 50)
        pseq = [30, 40, 50] + seq  # pseq for prepended_seq

        # adapt the specific tests from test_slicing
        self.assertEqual(p[0], 30)
        self.assertEqual(p[1:8], pseq[1:8])
        self.assertEqual(p[1:], pseq[1:])
        self.assertEqual(p[:5], pseq[:5])
        self.assertEqual(p[:], pseq[:])
        self.assertEqual(p[:100], pseq[:100])
        self.assertEqual(p[::2], pseq[::2])
        self.assertEqual(p[::-1], pseq[::-1])

    def test_prepend_indexing(self):
        """Tests interaction between prepending and indexing"""
        seq = list(range(20))
        p = mi.peekable(seq)

        p.prepend(30, 40, 50)

        self.assertEqual(p[0], 30)
        self.assertEqual(next(p), 30)
        self.assertEqual(p[2], 0)
        self.assertEqual(next(p), 40)
        self.assertEqual(p[0], 50)
        self.assertEqual(p[9], 8)
        self.assertEqual(next(p), 50)
        self.assertEqual(p[8], 8)
        self.assertEqual(p[-2], 18)
        self.assertEqual(p[-9], 11)
        self.assertRaises(IndexError, lambda: p[-21])

    def test_prepend_iterable(self):
        """Tests prepending from an iterable"""
        it = mi.peekable(range(5))
        # Don't directly use the range() object to avoid any range-specific
        # optimizations
        it.prepend(*(x for x in range(5)))
        actual = list(it)
        expected = list(chain(range(5), range(5)))
        self.assertEqual(actual, expected)

    def test_prepend_many(self):
        """Tests that prepending a huge number of elements works"""
        it = mi.peekable(range(5))
        # Don't directly use the range() object to avoid any range-specific
        # optimizations
        it.prepend(*(x for x in range(20000)))
        actual = list(it)
        expected = list(chain(range(20000), range(5)))
        self.assertEqual(actual, expected)

    def test_prepend_reversed(self):
        """Tests prepending from a reversed iterable"""
        it = mi.peekable(range(3))
        it.prepend(*reversed((10, 11, 12)))
        actual = list(it)
        expected = [12, 11, 10, 0, 1, 2]
        self.assertEqual(actual, expected)


class ConsumerTests(TestCase):
    """Tests for ``consumer()``"""

    def test_consumer(self):
        @mi.consumer
        def eater():
            while True:
                x = yield  # noqa

        e = eater()
        e.send('hi')  # without @consumer, would raise TypeError


class DistinctPermutationsTests(TestCase):
    def test_distinct_permutations(self):
        """Make sure the output for ``distinct_permutations()`` is the same as
        set(permutations(it)).

        """
        iterable = ['z', 'a', 'a', 'q', 'q', 'q', 'y']
        test_output = sorted(mi.distinct_permutations(iterable))
        ref_output = sorted(set(permutations(iterable)))
        self.assertEqual(test_output, ref_output)

    def test_other_iterables(self):
        """Make sure ``distinct_permutations()`` accepts a different type of
        iterables.

        """
        # a generator
        iterable = (c for c in ['z', 'a', 'a', 'q', 'q', 'q', 'y'])
        test_output = sorted(mi.distinct_permutations(iterable))
        # "reload" it
        iterable = (c for c in ['z', 'a', 'a', 'q', 'q', 'q', 'y'])
        ref_output = sorted(set(permutations(iterable)))
        self.assertEqual(test_output, ref_output)

        # an iterator
        iterable = iter(['z', 'a', 'a', 'q', 'q', 'q', 'y'])
        test_output = sorted(mi.distinct_permutations(iterable))
        # "reload" it
        iterable = iter(['z', 'a', 'a', 'q', 'q', 'q', 'y'])
        ref_output = sorted(set(permutations(iterable)))
        self.assertEqual(test_output, ref_output)

    def test_r(self):
        for iterable, r in (
            ('mississippi', 0),
            ('mississippi', 1),
            ('mississippi', 6),
            ('mississippi', 7),
            ('mississippi', 12),
            ([0, 1, 1, 0], 0),
            ([0, 1, 1, 0], 1),
            ([0, 1, 1, 0], 2),
            ([0, 1, 1, 0], 3),
            ([0, 1, 1, 0], 4),
            (['a'], 0),
            (['a'], 1),
            (['a'], 5),
            ([], 0),
            ([], 1),
            ([], 4),
        ):
            with self.subTest(iterable=iterable, r=r):
                expected = sorted(set(permutations(iterable, r)))
                actual = sorted(mi.distinct_permutations(iter(iterable), r))
                self.assertEqual(actual, expected)


class IlenTests(TestCase):
    def test_ilen(self):
        """Sanity-checks for ``ilen()``."""
        # Non-empty
        self.assertEqual(
            mi.ilen(filter(lambda x: x % 10 == 0, range(101))), 11
        )

        # Empty
        self.assertEqual(mi.ilen(x for x in range(0)), 0)

        # Iterable with __len__
        self.assertEqual(mi.ilen(list(range(6))), 6)


class MinMaxTests(TestCase):
    def test_basic(self):
        for iterable, expected in (
            # easy case
            ([0, 1, 2, 3], (0, 3)),
            # min and max are not in the extremes + we have `int`s and `float`s
            ([3, 5.5, -1, 2], (-1, 5.5)),
            # unordered collection
            ({3, 5.5, -1, 2}, (-1, 5.5)),
            # with repetitions
            ([3, 5.5, float('-Inf'), 5.5], (float('-Inf'), 5.5)),
            # other collections
            ('banana', ('a', 'n')),
            ({0: 1, 2: 100, 1: 10}, (0, 2)),
            (range(3, 14), (3, 13)),
        ):
            with self.subTest(iterable=iterable, expected=expected):
                # check for expected results
                self.assertTupleEqual(mi.minmax(iterable), expected)
                # check for equality with built-in `min` and `max`
                self.assertTupleEqual(
                    mi.minmax(iterable), (min(iterable), max(iterable))
                )

    def test_unpacked(self):
        self.assertTupleEqual(mi.minmax(2, 3, 1), (1, 3))
        self.assertTupleEqual(mi.minmax(12, 3, 4, key=str), (12, 4))

    def test_iterables(self):
        self.assertTupleEqual(mi.minmax(x for x in [0, 1, 2, 3]), (0, 3))
        self.assertTupleEqual(
            mi.minmax(map(str, [3, 5.5, 'a', 2])), ('2', 'a')
        )
        self.assertTupleEqual(
            mi.minmax(filter(None, [0, 3, '', None, 10])), (3, 10)
        )

    def test_key(self):
        self.assertTupleEqual(
            mi.minmax({(), (1, 4, 2), 'abcde', range(4)}, key=len),
            ((), 'abcde'),
        )
        self.assertTupleEqual(
            mi.minmax((x for x in [10, 3, 25]), key=str), (10, 3)
        )

    def test_default(self):
        with self.assertRaises(ValueError):
            mi.minmax([])

        self.assertIs(mi.minmax([], default=None), None)
        self.assertListEqual(mi.minmax([], default=[1, 'a']), [1, 'a'])


class WithIterTests(TestCase):
    def test_with_iter(self):
        s = StringIO('One fish\nTwo fish')
        initial_words = [line.split()[0] for line in mi.with_iter(s)]

        # Iterable's items should be faithfully represented
        self.assertEqual(initial_words, ['One', 'Two'])
        # The file object should be closed
        self.assertTrue(s.closed)


class OneTests(TestCase):
    def test_basic(self):
        it = iter(['item'])
        self.assertEqual(mi.one(it), 'item')

    def test_too_short(self):
        it = iter([])
        for too_short, exc_type in [
            (None, ValueError),
            (IndexError, IndexError),
        ]:
            with self.subTest(too_short=too_short):
                try:
                    mi.one(it, too_short=too_short)
                except exc_type:
                    formatted_exc = format_exc()
                    self.assertIn('StopIteration', formatted_exc)
                    self.assertIn(
                        'The above exception was the direct cause',
                        formatted_exc,
                    )
                else:
                    self.fail()

    def test_too_long(self):
        it = count()
        self.assertRaises(ValueError, lambda: mi.one(it))  # burn 0 and 1
        self.assertEqual(next(it), 2)
        self.assertRaises(
            OverflowError, lambda: mi.one(it, too_long=OverflowError)
        )

    def test_too_long_default_message(self):
        it = count()
        self.assertRaisesRegex(
            ValueError,
            "Expected exactly one item in "
            "iterable, but got 0, 1, and "
            "perhaps more.",
            lambda: mi.one(it),
        )


class IntersperseTest(TestCase):
    """Tests for intersperse()"""

    def test_even(self):
        iterable = (x for x in '01')
        self.assertEqual(
            list(mi.intersperse(None, iterable)), ['0', None, '1']
        )

    def test_odd(self):
        iterable = (x for x in '012')
        self.assertEqual(
            list(mi.intersperse(None, iterable)), ['0', None, '1', None, '2']
        )

    def test_nested(self):
        element = ('a', 'b')
        iterable = (x for x in '012')
        actual = list(mi.intersperse(element, iterable))
        expected = ['0', ('a', 'b'), '1', ('a', 'b'), '2']
        self.assertEqual(actual, expected)

    def test_not_iterable(self):
        self.assertRaises(TypeError, lambda: mi.intersperse('x', 1))

    def test_n(self):
        for n, element, expected in [
            (1, '_', ['0', '_', '1', '_', '2', '_', '3', '_', '4', '_', '5']),
            (2, '_', ['0', '1', '_', '2', '3', '_', '4', '5']),
            (3, '_', ['0', '1', '2', '_', '3', '4', '5']),
            (4, '_', ['0', '1', '2', '3', '_', '4', '5']),
            (5, '_', ['0', '1', '2', '3', '4', '_', '5']),
            (6, '_', ['0', '1', '2', '3', '4', '5']),
            (7, '_', ['0', '1', '2', '3', '4', '5']),
            (3, ['a', 'b'], ['0', '1', '2', ['a', 'b'], '3', '4', '5']),
        ]:
            iterable = (x for x in '012345')
            actual = list(mi.intersperse(element, iterable, n=n))
            self.assertEqual(actual, expected)

    def test_n_zero(self):
        self.assertRaises(
            ValueError, lambda: list(mi.intersperse('x', '012', n=0))
        )


class UniqueToEachTests(TestCase):
    """Tests for ``unique_to_each()``"""

    def test_all_unique(self):
        """When all the input iterables are unique the output should match
        the input."""
        iterables = [[1, 2], [3, 4, 5], [6, 7, 8]]
        self.assertEqual(mi.unique_to_each(*iterables), iterables)

    def test_duplicates(self):
        """When there are duplicates in any of the input iterables that aren't
        in the rest, those duplicates should be emitted."""
        iterables = ["mississippi", "missouri"]
        self.assertEqual(
            mi.unique_to_each(*iterables), [['p', 'p'], ['o', 'u', 'r']]
        )

    def test_mixed(self):
        """When the input iterables contain different types the function should
        still behave properly"""
        iterables = ['x', (i for i in range(3)), [1, 2, 3], tuple()]
        self.assertEqual(mi.unique_to_each(*iterables), [['x'], [0], [3], []])


class WindowedTests(TestCase):
    def test_basic(self):
        iterable = [1, 2, 3, 4, 5]

        for n, expected in (
            (6, [(1, 2, 3, 4, 5, None)]),
            (5, [(1, 2, 3, 4, 5)]),
            (4, [(1, 2, 3, 4), (2, 3, 4, 5)]),
            (3, [(1, 2, 3), (2, 3, 4), (3, 4, 5)]),
            (2, [(1, 2), (2, 3), (3, 4), (4, 5)]),
            (1, [(1,), (2,), (3,), (4,), (5,)]),
            (0, [()]),
        ):
            with self.subTest(n=n):
                actual = list(mi.windowed(iterable, n))
                self.assertEqual(actual, expected)

    def test_fillvalue(self):
        actual = list(mi.windowed([1, 2, 3, 4, 5], 6, fillvalue='!'))
        expected = [(1, 2, 3, 4, 5, '!')]
        self.assertEqual(actual, expected)

    def test_step(self):
        iterable = [1, 2, 3, 4, 5, 6, 7]
        for n, step, expected in [
            (3, 2, [(1, 2, 3), (3, 4, 5), (5, 6, 7)]),  # n > step
            (3, 3, [(1, 2, 3), (4, 5, 6), (7, None, None)]),  # n == step
            (3, 4, [(1, 2, 3), (5, 6, 7)]),  # lines up nicely
            (3, 5, [(1, 2, 3), (6, 7, None)]),  # off by one
            (3, 6, [(1, 2, 3), (7, None, None)]),  # off by two
            (3, 7, [(1, 2, 3)]),  # step past the end
            (7, 8, [(1, 2, 3, 4, 5, 6, 7)]),  # step > len(iterable)
        ]:
            with self.subTest(n=n, step=step):
                actual = list(mi.windowed(iterable, n, step=step))
                self.assertEqual(actual, expected)

    def test_invalid_step(self):
        # Step must be greater than or equal to 1
        with self.assertRaises(ValueError):
            list(mi.windowed([1, 2, 3, 4, 5], 3, step=0))

    def test_fillvalue_step(self):
        actual = list(mi.windowed([1, 2, 3, 4, 5], 3, fillvalue='!', step=3))
        expected = [(1, 2, 3), (4, 5, '!')]
        self.assertEqual(actual, expected)

    def test_negative(self):
        with self.assertRaises(ValueError):
            list(mi.windowed([1, 2, 3, 4, 5], -1))

    def test_empty_seq(self):
        actual = list(mi.windowed([], 3))
        expected = []
        self.assertEqual(actual, expected)


class SubstringsTests(TestCase):
    def test_basic(self):
        iterable = (x for x in range(4))
        actual = list(mi.substrings(iterable))
        expected = [
            (0,),
            (1,),
            (2,),
            (3,),
            (0, 1),
            (1, 2),
            (2, 3),
            (0, 1, 2),
            (1, 2, 3),
            (0, 1, 2, 3),
        ]
        self.assertEqual(actual, expected)

    def test_strings(self):
        iterable = 'abc'
        actual = list(mi.substrings(iterable))
        expected = [
            ('a',),
            ('b',),
            ('c',),
            ('a', 'b'),
            ('b', 'c'),
            ('a', 'b', 'c'),
        ]
        self.assertEqual(actual, expected)

    def test_empty(self):
        iterable = iter([])
        actual = list(mi.substrings(iterable))
        expected = []
        self.assertEqual(actual, expected)

    def test_order(self):
        iterable = [2, 0, 1]
        actual = list(mi.substrings(iterable))
        expected = [(2,), (0,), (1,), (2, 0), (0, 1), (2, 0, 1)]
        self.assertEqual(actual, expected)


class SubstringsIndexesTests(TestCase):
    def test_basic(self):
        sequence = [x for x in range(4)]
        actual = list(mi.substrings_indexes(sequence))
        expected = [
            ([0], 0, 1),
            ([1], 1, 2),
            ([2], 2, 3),
            ([3], 3, 4),
            ([0, 1], 0, 2),
            ([1, 2], 1, 3),
            ([2, 3], 2, 4),
            ([0, 1, 2], 0, 3),
            ([1, 2, 3], 1, 4),
            ([0, 1, 2, 3], 0, 4),
        ]
        self.assertEqual(actual, expected)

    def test_strings(self):
        sequence = 'abc'
        actual = list(mi.substrings_indexes(sequence))
        expected = [
            ('a', 0, 1),
            ('b', 1, 2),
            ('c', 2, 3),
            ('ab', 0, 2),
            ('bc', 1, 3),
            ('abc', 0, 3),
        ]
        self.assertEqual(actual, expected)

    def test_empty(self):
        sequence = []
        actual = list(mi.substrings_indexes(sequence))
        expected = []
        self.assertEqual(actual, expected)

    def test_order(self):
        sequence = [2, 0, 1]
        actual = list(mi.substrings_indexes(sequence))
        expected = [
            ([2], 0, 1),
            ([0], 1, 2),
            ([1], 2, 3),
            ([2, 0], 0, 2),
            ([0, 1], 1, 3),
            ([2, 0, 1], 0, 3),
        ]
        self.assertEqual(actual, expected)

    def test_reverse(self):
        sequence = [2, 0, 1]
        actual = list(mi.substrings_indexes(sequence, reverse=True))
        expected = [
            ([2, 0, 1], 0, 3),
            ([2, 0], 0, 2),
            ([0, 1], 1, 3),
            ([2], 0, 1),
            ([0], 1, 2),
            ([1], 2, 3),
        ]
        self.assertEqual(actual, expected)


class BucketTests(TestCase):
    def test_basic(self):
        iterable = [10, 20, 30, 11, 21, 31, 12, 22, 23, 33]
        D = mi.bucket(iterable, key=lambda x: 10 * (x // 10))

        # In-order access
        self.assertEqual(list(D[10]), [10, 11, 12])

        # Out of order access
        self.assertEqual(list(D[30]), [30, 31, 33])
        self.assertEqual(list(D[20]), [20, 21, 22, 23])

        self.assertEqual(list(D[40]), [])  # Nothing in here!

    def test_in(self):
        iterable = [10, 20, 30, 11, 21, 31, 12, 22, 23, 33]
        D = mi.bucket(iterable, key=lambda x: 10 * (x // 10))

        self.assertIn(10, D)
        self.assertNotIn(40, D)
        self.assertIn(20, D)
        self.assertNotIn(21, D)

        # Checking in-ness shouldn't advance the iterator
        self.assertEqual(next(D[10]), 10)

    def test_validator(self):
        iterable = count(0)
        key = lambda x: int(str(x)[0])  # First digit of each number
        validator = lambda x: 0 < x < 10  # No leading zeros
        D = mi.bucket(iterable, key, validator=validator)
        self.assertEqual(mi.take(3, D[1]), [1, 10, 11])
        self.assertNotIn(0, D)  # Non-valid entries don't return True
        self.assertNotIn(0, D._cache)  # Don't store non-valid entries
        self.assertEqual(list(D[0]), [])

    def test_list(self):
        iterable = [10, 20, 30, 11, 21, 31, 12, 22, 23, 33]
        D = mi.bucket(iterable, key=lambda x: 10 * (x // 10))
        self.assertEqual(list(D[10]), [10, 11, 12])
        self.assertEqual(list(D[20]), [20, 21, 22, 23])
        self.assertEqual(list(D[30]), [30, 31, 33])
        self.assertEqual(set(D), {10, 20, 30})

    def test_list_validator(self):
        iterable = [10, 20, 30, 11, 21, 31, 12, 22, 23, 33]
        key = lambda x: 10 * (x // 10)
        validator = lambda x: x != 20
        D = mi.bucket(iterable, key, validator=validator)
        self.assertEqual(set(D), {10, 30})
        self.assertEqual(list(D[10]), [10, 11, 12])
        self.assertEqual(list(D[20]), [])
        self.assertEqual(list(D[30]), [30, 31, 33])


class SpyTests(TestCase):
    """Tests for ``spy()``"""

    def test_basic(self):
        original_iterable = iter('abcdefg')
        head, new_iterable = mi.spy(original_iterable)
        self.assertEqual(head, ['a'])
        self.assertEqual(
            list(new_iterable), ['a', 'b', 'c', 'd', 'e', 'f', 'g']
        )

    def test_unpacking(self):
        original_iterable = iter('abcdefg')
        (first, second, third), new_iterable = mi.spy(original_iterable, 3)
        self.assertEqual(first, 'a')
        self.assertEqual(second, 'b')
        self.assertEqual(third, 'c')
        self.assertEqual(
            list(new_iterable), ['a', 'b', 'c', 'd', 'e', 'f', 'g']
        )

    def test_too_many(self):
        original_iterable = iter('abc')
        head, new_iterable = mi.spy(original_iterable, 4)
        self.assertEqual(head, ['a', 'b', 'c'])
        self.assertEqual(list(new_iterable), ['a', 'b', 'c'])

    def test_zero(self):
        original_iterable = iter('abc')
        head, new_iterable = mi.spy(original_iterable, 0)
        self.assertEqual(head, [])
        self.assertEqual(list(new_iterable), ['a', 'b', 'c'])

    def test_immutable(self):
        original_iterable = iter('abcdefg')
        head, new_iterable = mi.spy(original_iterable, 3)
        head[0] = 'A'
        self.assertEqual(head, ['A', 'b', 'c'])
        self.assertEqual(
            list(new_iterable), ['a', 'b', 'c', 'd', 'e', 'f', 'g']
        )


class InterleaveTests(TestCase):
    def test_even(self):
        actual = list(mi.interleave([1, 4, 7], [2, 5, 8], [3, 6, 9]))
        expected = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.assertEqual(actual, expected)

    def test_short(self):
        actual = list(mi.interleave([1, 4], [2, 5, 7], [3, 6, 8]))
        expected = [1, 2, 3, 4, 5, 6]
        self.assertEqual(actual, expected)

    def test_mixed_types(self):
        it_list = ['a', 'b', 'c', 'd']
        it_str = '12345'
        it_inf = count()
        actual = list(mi.interleave(it_list, it_str, it_inf))
        expected = ['a', '1', 0, 'b', '2', 1, 'c', '3', 2, 'd', '4', 3]
        self.assertEqual(actual, expected)


class InterleaveLongestTests(TestCase):
    def test_even(self):
        actual = list(mi.interleave_longest([1, 4, 7], [2, 5, 8], [3, 6, 9]))
        expected = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.assertEqual(actual, expected)

    def test_short(self):
        actual = list(mi.interleave_longest([1, 4], [2, 5, 7], [3, 6, 8]))
        expected = [1, 2, 3, 4, 5, 6, 7, 8]
        self.assertEqual(actual, expected)

    def test_mixed_types(self):
        it_list = ['a', 'b', 'c', 'd']
        it_str = '12345'
        it_gen = (x for x in range(3))
        actual = list(mi.interleave_longest(it_list, it_str, it_gen))
        expected = ['a', '1', 0, 'b', '2', 1, 'c', '3', 2, 'd', '4', '5']
        self.assertEqual(actual, expected)


class InterleaveEvenlyTests(TestCase):
    def test_equal_lengths(self):
        # when lengths are equal, the relative order shouldn't change
        a = [1, 2, 3]
        b = [5, 6, 7]
        actual = list(mi.interleave_evenly([a, b]))
        expected = [1, 5, 2, 6, 3, 7]
        self.assertEqual(actual, expected)

    def test_proportional(self):
        # easy case where the iterables have proportional length
        a = [1, 2, 3, 4]
        b = [5, 6]
        actual = list(mi.interleave_evenly([a, b]))
        expected = [1, 2, 5, 3, 4, 6]
        self.assertEqual(actual, expected)

        # swapping a and b should yield the same result
        actual_swapped = list(mi.interleave_evenly([b, a]))
        self.assertEqual(actual_swapped, expected)

    def test_not_proportional(self):
        a = [1, 2, 3, 4, 5, 6, 7]
        b = [8, 9, 10]
        expected = [1, 2, 8, 3, 4, 9, 5, 6, 10, 7]
        actual = list(mi.interleave_evenly([a, b]))
        self.assertEqual(actual, expected)

    def test_degenerate_one(self):
        a = [0, 1, 2, 3, 4]
        b = [5]
        expected = [0, 1, 2, 5, 3, 4]
        actual = list(mi.interleave_evenly([a, b]))
        self.assertEqual(actual, expected)

    def test_degenerate_empty(self):
        a = [1, 2, 3]
        b = []
        expected = [1, 2, 3]
        actual = list(mi.interleave_evenly([a, b]))
        self.assertEqual(actual, expected)

    def test_three_iters(self):
        a = ["a1", "a2", "a3", "a4", "a5"]
        b = ["b1", "b2", "b3"]
        c = ["c1"]
        actual = list(mi.interleave_evenly([a, b, c]))
        expected = ["a1", "b1", "a2", "c1", "a3", "b2", "a4", "b3", "a5"]
        self.assertEqual(actual, expected)

    def test_many_iters(self):
        # smoke test with many iterables: create iterables with a random
        # number of elements starting with a character ("a0", "a1", ...)
        rng = Random(0)
        iterables = []
        for ch in ascii_letters:
            length = rng.randint(0, 100)
            iterable = [f"{ch}{i}" for i in range(length)]
            iterables.append(iterable)

        interleaved = list(mi.interleave_evenly(iterables))

        # for each iterable, check that the result contains all its items
        for iterable, ch_expect in zip(iterables, ascii_letters):
            interleaved_actual = [
                e for e in interleaved if e.startswith(ch_expect)
            ]
            assert len(set(interleaved_actual)) == len(iterable)

    def test_manual_lengths(self):
        a = combinations(range(4), 2)
        len_a = 4 * (4 - 1) // 2  # == 6
        b = combinations(range(4), 3)
        len_b = 4

        expected = [
            (0, 1),
            (0, 1, 2),
            (0, 2),
            (0, 3),
            (0, 1, 3),
            (1, 2),
            (0, 2, 3),
            (1, 3),
            (2, 3),
            (1, 2, 3),
        ]
        actual = list(mi.interleave_evenly([a, b], lengths=[len_a, len_b]))
        self.assertEqual(expected, actual)

    def test_no_length_raises(self):
        # combinations doesn't have __len__, should trigger ValueError
        iterables = [range(5), combinations(range(5), 2)]
        with self.assertRaises(ValueError):
            list(mi.interleave_evenly(iterables))

    def test_argument_mismatch_raises(self):
        # pass mismatching number of iterables and lengths
        iterables = [range(3)]
        lengths = [3, 4]
        with self.assertRaises(ValueError):
            list(mi.interleave_evenly(iterables, lengths=lengths))


class TestCollapse(TestCase):
    """Tests for ``collapse()``"""

    def test_collapse(self):
        l = [[1], 2, [[3], 4], [[[5]]]]
        self.assertEqual(list(mi.collapse(l)), [1, 2, 3, 4, 5])

    def test_collapse_to_string(self):
        l = [["s1"], "s2", [["s3"], "s4"], [[["s5"]]]]
        self.assertEqual(list(mi.collapse(l)), ["s1", "s2", "s3", "s4", "s5"])

    def test_collapse_to_bytes(self):
        l = [[b"s1"], b"s2", [[b"s3"], b"s4"], [[[b"s5"]]]]
        self.assertEqual(
            list(mi.collapse(l)), [b"s1", b"s2", b"s3", b"s4", b"s5"]
        )

    def test_collapse_flatten(self):
        l = [[1], [2], [[3], 4], [[[5]]]]
        self.assertEqual(list(mi.collapse(l, levels=1)), list(mi.flatten(l)))

    def test_collapse_to_level(self):
        l = [[1], 2, [[3], 4], [[[5]]]]
        self.assertEqual(list(mi.collapse(l, levels=2)), [1, 2, 3, 4, [5]])
        self.assertEqual(
            list(mi.collapse(mi.collapse(l, levels=1), levels=1)),
            list(mi.collapse(l, levels=2)),
        )

    def test_collapse_to_list(self):
        l = (1, [2], (3, [4, (5,)], 'ab'))
        actual = list(mi.collapse(l, base_type=list))
        expected = [1, [2], 3, [4, (5,)], 'ab']
        self.assertEqual(actual, expected)


class SideEffectTests(TestCase):
    """Tests for ``side_effect()``"""

    def test_individual(self):
        # The function increments the counter for each call
        counter = [0]

        def func(arg):
            counter[0] += 1

        result = list(mi.side_effect(func, range(10)))
        self.assertEqual(result, list(range(10)))
        self.assertEqual(counter[0], 10)

    def test_chunked(self):
        # The function increments the counter for each call
        counter = [0]

        def func(arg):
            counter[0] += 1

        result = list(mi.side_effect(func, range(10), 2))
        self.assertEqual(result, list(range(10)))
        self.assertEqual(counter[0], 5)

    def test_before_after(self):
        f = StringIO()
        collector = []

        def func(item):
            print(item, file=f)
            collector.append(f.getvalue())

        def it():
            yield 'a'
            yield 'b'
            raise RuntimeError('kaboom')

        before = lambda: print('HEADER', file=f)
        after = f.close

        try:
            mi.consume(mi.side_effect(func, it(), before=before, after=after))
        except RuntimeError:
            pass

        # The iterable should have been written to the file
        self.assertEqual(collector, ['HEADER\na\n', 'HEADER\na\nb\n'])

        # The file should be closed even though something bad happened
        self.assertTrue(f.closed)

    def test_before_fails(self):
        f = StringIO()
        func = lambda x: print(x, file=f)

        def before():
            raise RuntimeError('ouch')

        try:
            mi.consume(
                mi.side_effect(func, 'abc', before=before, after=f.close)
            )
        except RuntimeError:
            pass

        # The file should be closed even though something bad happened in the
        # before function
        self.assertTrue(f.closed)


class SlicedTests(TestCase):
    """Tests for ``sliced()``"""

    def test_even(self):
        """Test when the length of the sequence is divisible by *n*"""
        seq = 'ABCDEFGHI'
        self.assertEqual(list(mi.sliced(seq, 3)), ['ABC', 'DEF', 'GHI'])

    def test_odd(self):
        """Test when the length of the sequence is not divisible by *n*"""
        seq = 'ABCDEFGHI'
        self.assertEqual(list(mi.sliced(seq, 4)), ['ABCD', 'EFGH', 'I'])

    def test_not_sliceable(self):
        seq = (x for x in 'ABCDEFGHI')

        with self.assertRaises(TypeError):
            list(mi.sliced(seq, 3))

    def test_odd_and_strict(self):
        seq = [x for x in 'ABCDEFGHI']

        with self.assertRaises(ValueError):
            list(mi.sliced(seq, 4, strict=True))

    def test_numpy_like_array(self):
        # Numpy arrays don't behave like Python lists - calling bool()
        # on them doesn't return False for empty lists and True for non-empty
        # ones. Emulate that behavior.
        class FalseList(list):
            def __getitem__(self, key):
                ret = super().__getitem__(key)
                if isinstance(key, slice):
                    return FalseList(ret)

                return ret

            def __bool__(self):
                return False

        seq = FalseList(range(9))
        actual = list(mi.sliced(seq, 3))
        expected = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        self.assertEqual(actual, expected)


class SplitAtTests(TestCase):
    def test_basic(self):
        for iterable, separator in [
            ('a,bb,ccc,dddd', ','),
            (',a,bb,ccc,dddd', ','),
            ('a,bb,ccc,dddd,', ','),
            ('a,bb,ccc,,dddd', ','),
            ('', ','),
            (',', ','),
            ('a,bb,ccc,dddd', ';'),
        ]:
            with self.subTest(iterable=iterable, separator=separator):
                it = iter(iterable)
                pred = lambda x: x == separator
                actual = [''.join(x) for x in mi.split_at(it, pred)]
                expected = iterable.split(separator)
                self.assertEqual(actual, expected)

    def test_maxsplit(self):
        iterable = 'a,bb,ccc,dddd'
        separator = ','
        pred = lambda x: x == separator

        for maxsplit in range(-1, 4):
            with self.subTest(maxsplit=maxsplit):
                it = iter(iterable)
                result = mi.split_at(it, pred, maxsplit=maxsplit)
                actual = [''.join(x) for x in result]
                expected = iterable.split(separator, maxsplit)
                self.assertEqual(actual, expected)

    def test_keep_separator(self):
        separator = ','
        pred = lambda x: x == separator

        for iterable, expected in [
            ('a,bb,ccc', ['a', ',', 'bb', ',', 'ccc']),
            (',a,bb,ccc', ['', ',', 'a', ',', 'bb', ',', 'ccc']),
            ('a,bb,ccc,', ['a', ',', 'bb', ',', 'ccc', ',', '']),
        ]:
            with self.subTest(iterable=iterable):
                it = iter(iterable)
                result = mi.split_at(it, pred, keep_separator=True)
                actual = [''.join(x) for x in result]
                self.assertEqual(actual, expected)

    def test_combination(self):
        iterable = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        pred = lambda x: x % 3 == 0
        actual = list(
            mi.split_at(iterable, pred, maxsplit=2, keep_separator=True)
        )
        expected = [[1, 2], [3], [4, 5], [6], [7, 8, 9, 10]]
        self.assertEqual(actual, expected)


class SplitBeforeTest(TestCase):
    """Tests for ``split_before()``"""

    def test_starts_with_sep(self):
        actual = list(mi.split_before('xooxoo', lambda c: c == 'x'))
        expected = [['x', 'o', 'o'], ['x', 'o', 'o']]
        self.assertEqual(actual, expected)

    def test_ends_with_sep(self):
        actual = list(mi.split_before('ooxoox', lambda c: c == 'x'))
        expected = [['o', 'o'], ['x', 'o', 'o'], ['x']]
        self.assertEqual(actual, expected)

    def test_no_sep(self):
        actual = list(mi.split_before('ooo', lambda c: c == 'x'))
        expected = [['o', 'o', 'o']]
        self.assertEqual(actual, expected)

    def test_empty_collection(self):
        actual = list(mi.split_before([], lambda c: bool(c)))
        expected = []
        self.assertEqual(actual, expected)

    def test_max_split(self):
        for args, expected in [
            (
                ('a,b,c,d', lambda c: c == ',', -1),
                [['a'], [',', 'b'], [',', 'c'], [',', 'd']],
            ),
            (
                ('a,b,c,d', lambda c: c == ',', 0),
                [['a', ',', 'b', ',', 'c', ',', 'd']],
            ),
            (
                ('a,b,c,d', lambda c: c == ',', 1),
                [['a'], [',', 'b', ',', 'c', ',', 'd']],
            ),
            (
                ('a,b,c,d', lambda c: c == ',', 2),
                [['a'], [',', 'b'], [',', 'c', ',', 'd']],
            ),
            (
                ('a,b,c,d', lambda c: c == ',', 10),
                [['a'], [',', 'b'], [',', 'c'], [',', 'd']],
            ),
            (
                ('a,b,c,d', lambda c: c == '@', 2),
                [['a', ',', 'b', ',', 'c', ',', 'd']],
            ),
            (
                ('a,b,c,d', lambda c: c != ',', 2),
                [['a', ','], ['b', ','], ['c', ',', 'd']],
            ),
        ]:
            actual = list(mi.split_before(*args))
            self.assertEqual(actual, expected)


class SplitAfterTest(TestCase):
    """Tests for ``split_after()``"""

    def test_starts_with_sep(self):
        actual = list(mi.split_after('xooxoo', lambda c: c == 'x'))
        expected = [['x'], ['o', 'o', 'x'], ['o', 'o']]
        self.assertEqual(actual, expected)

    def test_ends_with_sep(self):
        actual = list(mi.split_after('ooxoox', lambda c: c == 'x'))
        expected = [['o', 'o', 'x'], ['o', 'o', 'x']]
        self.assertEqual(actual, expected)

    def test_no_sep(self):
        actual = list(mi.split_after('ooo', lambda c: c == 'x'))
        expected = [['o', 'o', 'o']]
        self.assertEqual(actual, expected)

    def test_max_split(self):
        for args, expected in [
            (
                ('a,b,c,d', lambda c: c == ',', -1),
                [['a', ','], ['b', ','], ['c', ','], ['d']],
            ),
            (
                ('a,b,c,d', lambda c: c == ',', 0),
                [['a', ',', 'b', ',', 'c', ',', 'd']],
            ),
            (
                ('a,b,c,d', lambda c: c == ',', 1),
                [['a', ','], ['b', ',', 'c', ',', 'd']],
            ),
            (
                ('a,b,c,d', lambda c: c == ',', 2),
                [['a', ','], ['b', ','], ['c', ',', 'd']],
            ),
            (
                ('a,b,c,d', lambda c: c == ',', 10),
                [['a', ','], ['b', ','], ['c', ','], ['d']],
            ),
            (
                ('a,b,c,d', lambda c: c == '@', 2),
                [['a', ',', 'b', ',', 'c', ',', 'd']],
            ),
            (
                ('a,b,c,d', lambda c: c != ',', 2),
                [['a'], [',', 'b'], [',', 'c', ',', 'd']],
            ),
            (
                ([1], lambda x: x == 1, 1),
                [[1]],
            ),
        ]:
            actual = list(mi.split_after(*args))
            self.assertEqual(actual, expected)


class SplitWhenTests(TestCase):
    """Tests for ``split_when()``"""

    @staticmethod
    def _split_when_before(iterable, pred):
        return mi.split_when(iterable, lambda _, c: pred(c))

    @staticmethod
    def _split_when_after(iterable, pred):
        return mi.split_when(iterable, lambda c, _: pred(c))

    # split_before emulation
    def test_before_emulation_starts_with_sep(self):
        actual = list(self._split_when_before('xooxoo', lambda c: c == 'x'))
        expected = [['x', 'o', 'o'], ['x', 'o', 'o']]
        self.assertEqual(actual, expected)

    def test_before_emulation_ends_with_sep(self):
        actual = list(self._split_when_before('ooxoox', lambda c: c == 'x'))
        expected = [['o', 'o'], ['x', 'o', 'o'], ['x']]
        self.assertEqual(actual, expected)

    def test_before_emulation_no_sep(self):
        actual = list(self._split_when_before('ooo', lambda c: c == 'x'))
        expected = [['o', 'o', 'o']]
        self.assertEqual(actual, expected)

    # split_after emulation
    def test_after_emulation_starts_with_sep(self):
        actual = list(self._split_when_after('xooxoo', lambda c: c == 'x'))
        expected = [['x'], ['o', 'o', 'x'], ['o', 'o']]
        self.assertEqual(actual, expected)

    def test_after_emulation_ends_with_sep(self):
        actual = list(self._split_when_after('ooxoox', lambda c: c == 'x'))
        expected = [['o', 'o', 'x'], ['o', 'o', 'x']]
        self.assertEqual(actual, expected)

    def test_after_emulation_no_sep(self):
        actual = list(self._split_when_after('ooo', lambda c: c == 'x'))
        expected = [['o', 'o', 'o']]
        self.assertEqual(actual, expected)

    # edge cases
    def test_empty_iterable(self):
        actual = list(mi.split_when('', lambda a, b: a != b))
        expected = []
        self.assertEqual(actual, expected)

    def test_one_element(self):
        actual = list(mi.split_when('o', lambda a, b: a == b))
        expected = [['o']]
        self.assertEqual(actual, expected)

    def test_one_element_is_second_item(self):
        actual = list(self._split_when_before('x', lambda c: c == 'x'))
        expected = [['x']]
        self.assertEqual(actual, expected)

    def test_one_element_is_first_item(self):
        actual = list(self._split_when_after('x', lambda c: c == 'x'))
        expected = [['x']]
        self.assertEqual(actual, expected)

    def test_max_split(self):
        for args, expected in [
            (
                ('a,b,c,d', lambda a, _: a == ',', -1),
                [['a', ','], ['b', ','], ['c', ','], ['d']],
            ),
            (
                ('a,b,c,d', lambda a, _: a == ',', 0),
                [['a', ',', 'b', ',', 'c', ',', 'd']],
            ),
            (
                ('a,b,c,d', lambda _, b: b == ',', 1),
                [['a'], [',', 'b', ',', 'c', ',', 'd']],
            ),
            (
                ('a,b,c,d', lambda a, _: a == ',', 2),
                [['a', ','], ['b', ','], ['c', ',', 'd']],
            ),
            (
                ('0124376', lambda a, b: a > b, -1),
                [['0', '1', '2', '4'], ['3', '7'], ['6']],
            ),
            (
                ('0124376', lambda a, b: a > b, 0),
                [['0', '1', '2', '4', '3', '7', '6']],
            ),
            (
                ('0124376', lambda a, b: a > b, 1),
                [['0', '1', '2', '4'], ['3', '7', '6']],
            ),
            (
                ('0124376', lambda a, b: a > b, 2),
                [['0', '1', '2', '4'], ['3', '7'], ['6']],
            ),
        ]:
            actual = list(mi.split_when(*args))
            self.assertEqual(actual, expected, str(args))


class SplitIntoTests(TestCase):
    """Tests for ``split_into()``"""

    def test_iterable_just_right(self):
        """Size of ``iterable`` equals the sum of ``sizes``."""
        iterable = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        sizes = [2, 3, 4]
        expected = [[1, 2], [3, 4, 5], [6, 7, 8, 9]]
        actual = list(mi.split_into(iterable, sizes))
        self.assertEqual(actual, expected)

    def test_iterable_too_small(self):
        """Size of ``iterable`` is smaller than sum of ``sizes``. Last return
        list is shorter as a result."""
        iterable = [1, 2, 3, 4, 5, 6, 7]
        sizes = [2, 3, 4]
        expected = [[1, 2], [3, 4, 5], [6, 7]]
        actual = list(mi.split_into(iterable, sizes))
        self.assertEqual(actual, expected)

    def test_iterable_too_small_extra(self):
        """Size of ``iterable`` is smaller than sum of ``sizes``. Second last
        return list is shorter and last return list is empty as a result."""
        iterable = [1, 2, 3, 4, 5, 6, 7]
        sizes = [2, 3, 4, 5]
        expected = [[1, 2], [3, 4, 5], [6, 7], []]
        actual = list(mi.split_into(iterable, sizes))
        self.assertEqual(actual, expected)

    def test_iterable_too_large(self):
        """Size of ``iterable`` is larger than sum of ``sizes``. Not all
        items of iterable are returned."""
        iterable = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        sizes = [2, 3, 2]
        expected = [[1, 2], [3, 4, 5], [6, 7]]
        actual = list(mi.split_into(iterable, sizes))
        self.assertEqual(actual, expected)

    def test_using_none_with_leftover(self):
        """Last item of ``sizes`` is None when items still remain in
        ``iterable``. Last list returned stretches to fit all remaining items
        of ``iterable``."""
        iterable = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        sizes = [2, 3, None]
        expected = [[1, 2], [3, 4, 5], [6, 7, 8, 9]]
        actual = list(mi.split_into(iterable, sizes))
        self.assertEqual(actual, expected)

    def test_using_none_without_leftover(self):
        """Last item of ``sizes`` is None when no items remain in
        ``iterable``. Last list returned is empty."""
        iterable = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        sizes = [2, 3, 4, None]
        expected = [[1, 2], [3, 4, 5], [6, 7, 8, 9], []]
        actual = list(mi.split_into(iterable, sizes))
        self.assertEqual(actual, expected)

    def test_using_none_mid_sizes(self):
        """None is present in ``sizes`` but is not the last item. Last list
        returned stretches to fit all remaining items of ``iterable`` but
        all items in ``sizes`` after None are ignored."""
        iterable = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        sizes = [2, 3, None, 4]
        expected = [[1, 2], [3, 4, 5], [6, 7, 8, 9]]
        actual = list(mi.split_into(iterable, sizes))
        self.assertEqual(actual, expected)

    def test_iterable_empty(self):
        """``iterable`` argument is empty but ``sizes`` is not. An empty
        list is returned for each item in ``sizes``."""
        iterable = []
        sizes = [2, 4, 2]
        expected = [[], [], []]
        actual = list(mi.split_into(iterable, sizes))
        self.assertEqual(actual, expected)

    def test_iterable_empty_using_none(self):
        """``iterable`` argument is empty but ``sizes`` is not. An empty
        list is returned for each item in ``sizes`` that is not after a
        None item."""
        iterable = []
        sizes = [2, 4, None, 2]
        expected = [[], [], []]
        actual = list(mi.split_into(iterable, sizes))
        self.assertEqual(actual, expected)

    def test_sizes_empty(self):
        """``sizes`` argument is empty but ``iterable`` is not. An empty
        generator is returned."""
        iterable = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        sizes = []
        expected = []
        actual = list(mi.split_into(iterable, sizes))
        self.assertEqual(actual, expected)

    def test_both_empty(self):
        """Both ``sizes`` and ``iterable`` arguments are empty. An empty
        generator is returned."""
        iterable = []
        sizes = []
        expected = []
        actual = list(mi.split_into(iterable, sizes))
        self.assertEqual(actual, expected)

    def test_bool_in_sizes(self):
        """A bool object is present in ``sizes`` is treated as a 1 or 0 for
        ``True`` or ``False`` due to bool being an instance of int."""
        iterable = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        sizes = [3, True, 2, False]
        expected = [[1, 2, 3], [4], [5, 6], []]
        actual = list(mi.split_into(iterable, sizes))
        self.assertEqual(actual, expected)

    def test_invalid_in_sizes(self):
        """A ValueError is raised if an object in ``sizes`` is neither ``None``
        or an integer."""
        iterable = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        sizes = [1, [], 3]
        with self.assertRaises(ValueError):
            list(mi.split_into(iterable, sizes))

    def test_invalid_in_sizes_after_none(self):
        """A item in ``sizes`` that is invalid will not raise a TypeError if it
        comes after a ``None`` item."""
        iterable = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        sizes = [3, 4, None, []]
        expected = [[1, 2, 3], [4, 5, 6, 7], [8, 9]]
        actual = list(mi.split_into(iterable, sizes))
        self.assertEqual(actual, expected)

    def test_generator_iterable_integrity(self):
        """Check that if ``iterable`` is an iterator, it is consumed only by as
        many items as the sum of ``sizes``."""
        iterable = (i for i in range(10))
        sizes = [2, 3]

        expected = [[0, 1], [2, 3, 4]]
        actual = list(mi.split_into(iterable, sizes))
        self.assertEqual(actual, expected)

        iterable_expected = [5, 6, 7, 8, 9]
        iterable_actual = list(iterable)
        self.assertEqual(iterable_actual, iterable_expected)

    def test_generator_sizes_integrity(self):
        """Check that if ``sizes`` is an iterator, it is consumed only until a
        ``None`` item is reached"""
        iterable = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        sizes = (i for i in [1, 2, None, 3, 4])

        expected = [[1], [2, 3], [4, 5, 6, 7, 8, 9]]
        actual = list(mi.split_into(iterable, sizes))
        self.assertEqual(actual, expected)

        sizes_expected = [3, 4]
        sizes_actual = list(sizes)
        self.assertEqual(sizes_actual, sizes_expected)


class PaddedTest(TestCase):
    """Tests for ``padded()``"""

    def test_no_n(self):
        seq = [1, 2, 3]

        # No fillvalue
        self.assertEqual(mi.take(5, mi.padded(seq)), [1, 2, 3, None, None])

        # With fillvalue
        self.assertEqual(
            mi.take(5, mi.padded(seq, fillvalue='')), [1, 2, 3, '', '']
        )

    def test_invalid_n(self):
        self.assertRaises(ValueError, lambda: list(mi.padded([1, 2, 3], n=-1)))
        self.assertRaises(ValueError, lambda: list(mi.padded([1, 2, 3], n=0)))

    def test_valid_n(self):
        seq = [1, 2, 3, 4, 5]

        # No need for padding: len(seq) <= n
        self.assertEqual(list(mi.padded(seq, n=4)), [1, 2, 3, 4, 5])
        self.assertEqual(list(mi.padded(seq, n=5)), [1, 2, 3, 4, 5])

        # No fillvalue
        self.assertEqual(
            list(mi.padded(seq, n=7)), [1, 2, 3, 4, 5, None, None]
        )

        # With fillvalue
        self.assertEqual(
            list(mi.padded(seq, fillvalue='', n=7)), [1, 2, 3, 4, 5, '', '']
        )

    def test_next_multiple(self):
        seq = [1, 2, 3, 4, 5, 6]

        # No need for padding: len(seq) % n == 0
        self.assertEqual(
            list(mi.padded(seq, n=3, next_multiple=True)), [1, 2, 3, 4, 5, 6]
        )

        # Padding needed: len(seq) < n
        self.assertEqual(
            list(mi.padded(seq, n=8, next_multiple=True)),
            [1, 2, 3, 4, 5, 6, None, None],
        )

        # No padding needed: len(seq) == n
        self.assertEqual(
            list(mi.padded(seq, n=6, next_multiple=True)), [1, 2, 3, 4, 5, 6]
        )

        # Padding needed: len(seq) > n
        self.assertEqual(
            list(mi.padded(seq, n=4, next_multiple=True)),
            [1, 2, 3, 4, 5, 6, None, None],
        )

        # With fillvalue
        self.assertEqual(
            list(mi.padded(seq, fillvalue='', n=4, next_multiple=True)),
            [1, 2, 3, 4, 5, 6, '', ''],
        )


class RepeatEachTests(TestCase):
    """Tests for repeat_each()"""

    def test_default(self):
        actual = list(mi.repeat_each('ABC'))
        expected = ['A', 'A', 'B', 'B', 'C', 'C']
        self.assertEqual(actual, expected)

    def test_basic(self):
        actual = list(mi.repeat_each('ABC', 3))
        expected = ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C']
        self.assertEqual(actual, expected)

    def test_empty(self):
        actual = list(mi.repeat_each(''))
        expected = []
        self.assertEqual(actual, expected)

    def test_no_repeat(self):
        actual = list(mi.repeat_each('ABC', 0))
        expected = []
        self.assertEqual(actual, expected)

    def test_negative_repeat(self):
        actual = list(mi.repeat_each('ABC', -1))
        expected = []
        self.assertEqual(actual, expected)

    def test_infinite_input(self):
        repeater = mi.repeat_each(cycle('AB'))
        actual = mi.take(6, repeater)
        expected = ['A', 'A', 'B', 'B', 'A', 'A']
        self.assertEqual(actual, expected)


class RepeatLastTests(TestCase):
    def test_empty_iterable(self):
        slice_length = 3
        iterable = iter([])
        actual = mi.take(slice_length, mi.repeat_last(iterable))
        expected = [None] * slice_length
        self.assertEqual(actual, expected)

    def test_default_value(self):
        slice_length = 3
        iterable = iter([])
        default = '3'
        actual = mi.take(slice_length, mi.repeat_last(iterable, default))
        expected = ['3'] * slice_length
        self.assertEqual(actual, expected)

    def test_basic(self):
        slice_length = 10
        iterable = (str(x) for x in range(5))
        actual = mi.take(slice_length, mi.repeat_last(iterable))
        expected = ['0', '1', '2', '3', '4', '4', '4', '4', '4', '4']
        self.assertEqual(actual, expected)


class DistributeTest(TestCase):
    """Tests for distribute()"""

    def test_invalid_n(self):
        self.assertRaises(ValueError, lambda: mi.distribute(-1, [1, 2, 3]))
        self.assertRaises(ValueError, lambda: mi.distribute(0, [1, 2, 3]))

    def test_basic(self):
        iterable = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        for n, expected in [
            (1, [iterable]),
            (2, [[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]]),
            (3, [[1, 4, 7, 10], [2, 5, 8], [3, 6, 9]]),
            (10, [[n] for n in range(1, 10 + 1)]),
        ]:
            self.assertEqual(
                [list(x) for x in mi.distribute(n, iterable)], expected
            )

    def test_large_n(self):
        iterable = [1, 2, 3, 4]
        self.assertEqual(
            [list(x) for x in mi.distribute(6, iterable)],
            [[1], [2], [3], [4], [], []],
        )


class StaggerTest(TestCase):
    """Tests for ``stagger()``"""

    def test_default(self):
        iterable = [0, 1, 2, 3]
        actual = list(mi.stagger(iterable))
        expected = [(None, 0, 1), (0, 1, 2), (1, 2, 3)]
        self.assertEqual(actual, expected)

    def test_offsets(self):
        iterable = [0, 1, 2, 3]
        for offsets, expected in [
            ((-2, 0, 2), [('', 0, 2), ('', 1, 3)]),
            ((-2, -1), [('', ''), ('', 0), (0, 1), (1, 2), (2, 3)]),
            ((1, 2), [(1, 2), (2, 3)]),
        ]:
            all_groups = mi.stagger(iterable, offsets=offsets, fillvalue='')
            self.assertEqual(list(all_groups), expected)

    def test_longest(self):
        iterable = [0, 1, 2, 3]
        for offsets, expected in [
            (
                (-1, 0, 1),
                [('', 0, 1), (0, 1, 2), (1, 2, 3), (2, 3, ''), (3, '', '')],
            ),
            ((-2, -1), [('', ''), ('', 0), (0, 1), (1, 2), (2, 3), (3, '')]),
            ((1, 2), [(1, 2), (2, 3), (3, '')]),
        ]:
            all_groups = mi.stagger(
                iterable, offsets=offsets, fillvalue='', longest=True
            )
            self.assertEqual(list(all_groups), expected)


class ZipEqualTest(TestCase):
    @skipIf(version_info[:2] < (3, 10), 'zip_equal deprecated for 3.10+')
    def test_deprecation(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            self.assertEqual(
                list(mi.zip_equal([1, 2], [3, 4])), [(1, 3), (2, 4)]
            )

        (warning,) = caught
        assert warning.category == DeprecationWarning

    def test_equal(self):
        lists = [0, 1, 2], [2, 3, 4]

        for iterables in [lists, map(iter, lists)]:
            actual = list(mi.zip_equal(*iterables))
            expected = [(0, 2), (1, 3), (2, 4)]
            self.assertEqual(actual, expected)

    def test_unequal_lists(self):
        two_items = [0, 1]
        three_items = [2, 3, 4]
        four_items = [5, 6, 7, 8]

        # the mismatch is at index 1
        try:
            list(mi.zip_equal(two_items, three_items, four_items))
        except mi.UnequalIterablesError as e:
            self.assertEqual(
                e.args[0],
                (
                    'Iterables have different lengths: '
                    'index 0 has length 2; index 1 has length 3'
                ),
            )

        # the mismatch is at index 2
        try:
            list(mi.zip_equal(two_items, two_items, four_items, four_items))
        except mi.UnequalIterablesError as e:
            self.assertEqual(
                e.args[0],
                (
                    'Iterables have different lengths: '
                    'index 0 has length 2; index 2 has length 4'
                ),
            )

        # One without length: delegate to _zip_equal_generator
        try:
            list(mi.zip_equal(two_items, iter(two_items), three_items))
        except mi.UnequalIterablesError as e:
            self.assertEqual(e.args[0], 'Iterables have different lengths')


class ZipOffsetTest(TestCase):
    """Tests for ``zip_offset()``"""

    def test_shortest(self):
        a_1 = [0, 1, 2, 3]
        a_2 = [0, 1, 2, 3, 4, 5]
        a_3 = [0, 1, 2, 3, 4, 5, 6, 7]
        actual = list(
            mi.zip_offset(a_1, a_2, a_3, offsets=(-1, 0, 1), fillvalue='')
        )
        expected = [('', 0, 1), (0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 5)]
        self.assertEqual(actual, expected)

    def test_longest(self):
        a_1 = [0, 1, 2, 3]
        a_2 = [0, 1, 2, 3, 4, 5]
        a_3 = [0, 1, 2, 3, 4, 5, 6, 7]
        actual = list(
            mi.zip_offset(a_1, a_2, a_3, offsets=(-1, 0, 1), longest=True)
        )
        expected = [
            (None, 0, 1),
            (0, 1, 2),
            (1, 2, 3),
            (2, 3, 4),
            (3, 4, 5),
            (None, 5, 6),
            (None, None, 7),
        ]
        self.assertEqual(actual, expected)

    def test_mismatch(self):
        iterables = [0, 1, 2], [2, 3, 4]
        offsets = (-1, 0, 1)
        self.assertRaises(
            ValueError,
            lambda: list(mi.zip_offset(*iterables, offsets=offsets)),
        )


class UnzipTests(TestCase):
    """Tests for unzip()"""

    def test_empty_iterable(self):
        self.assertEqual(list(mi.unzip([])), [])
        # in reality zip([], [], []) is equivalent to iter([])
        # but it doesn't hurt to test both
        self.assertEqual(list(mi.unzip(zip([], [], []))), [])

    def test_length_one_iterable(self):
        xs, ys, zs = mi.unzip(zip([1], [2], [3]))
        self.assertEqual(list(xs), [1])
        self.assertEqual(list(ys), [2])
        self.assertEqual(list(zs), [3])

    def test_normal_case(self):
        xs, ys, zs = range(10), range(1, 11), range(2, 12)
        zipped = zip(xs, ys, zs)
        xs, ys, zs = mi.unzip(zipped)
        self.assertEqual(list(xs), list(range(10)))
        self.assertEqual(list(ys), list(range(1, 11)))
        self.assertEqual(list(zs), list(range(2, 12)))

    def test_improperly_zipped(self):
        zipped = iter([(1, 2, 3), (4, 5), (6,)])
        xs, ys, zs = mi.unzip(zipped)
        self.assertEqual(list(xs), [1, 4, 6])
        self.assertEqual(list(ys), [2, 5])
        self.assertEqual(list(zs), [3])

    def test_increasingly_zipped(self):
        zipped = iter([(1, 2), (3, 4, 5), (6, 7, 8, 9)])
        unzipped = mi.unzip(zipped)
        # from the docstring:
        # len(first tuple) is the number of iterables zipped
        self.assertEqual(len(unzipped), 2)
        xs, ys = unzipped
        self.assertEqual(list(xs), [1, 3, 6])
        self.assertEqual(list(ys), [2, 4, 7])


class SortTogetherTest(TestCase):
    """Tests for sort_together()"""

    def test_key_list(self):
        """tests `key_list` including default, iterables include duplicates"""
        iterables = [
            ['GA', 'GA', 'GA', 'CT', 'CT', 'CT'],
            ['May', 'Aug.', 'May', 'June', 'July', 'July'],
            [97, 20, 100, 70, 100, 20],
        ]

        self.assertEqual(
            mi.sort_together(iterables),
            [
                ('CT', 'CT', 'CT', 'GA', 'GA', 'GA'),
                ('June', 'July', 'July', 'May', 'Aug.', 'May'),
                (70, 100, 20, 97, 20, 100),
            ],
        )

        self.assertEqual(
            mi.sort_together(iterables, key_list=(0, 1)),
            [
                ('CT', 'CT', 'CT', 'GA', 'GA', 'GA'),
                ('July', 'July', 'June', 'Aug.', 'May', 'May'),
                (100, 20, 70, 20, 97, 100),
            ],
        )

        self.assertEqual(
            mi.sort_together(iterables, key_list=(0, 1, 2)),
            [
                ('CT', 'CT', 'CT', 'GA', 'GA', 'GA'),
                ('July', 'July', 'June', 'Aug.', 'May', 'May'),
                (20, 100, 70, 20, 97, 100),
            ],
        )

        self.assertEqual(
            mi.sort_together(iterables, key_list=(2,)),
            [
                ('GA', 'CT', 'CT', 'GA', 'GA', 'CT'),
                ('Aug.', 'July', 'June', 'May', 'May', 'July'),
                (20, 20, 70, 97, 100, 100),
            ],
        )

    def test_invalid_key_list(self):
        """tests `key_list` for indexes not available in `iterables`"""
        iterables = [
            ['GA', 'GA', 'GA', 'CT', 'CT', 'CT'],
            ['May', 'Aug.', 'May', 'June', 'July', 'July'],
            [97, 20, 100, 70, 100, 20],
        ]

        self.assertRaises(
            IndexError, lambda: mi.sort_together(iterables, key_list=(5,))
        )

    def test_key_function(self):
        """tests `key` function, including interaction with `key_list`"""
        iterables = [
            ['GA', 'GA', 'GA', 'CT', 'CT', 'CT'],
            ['May', 'Aug.', 'May', 'June', 'July', 'July'],
            [97, 20, 100, 70, 100, 20],
        ]
        self.assertEqual(
            mi.sort_together(iterables, key=lambda x: x),
            [
                ('CT', 'CT', 'CT', 'GA', 'GA', 'GA'),
                ('June', 'July', 'July', 'May', 'Aug.', 'May'),
                (70, 100, 20, 97, 20, 100),
            ],
        )
        self.assertEqual(
            mi.sort_together(iterables, key=lambda x: x[::-1]),
            [
                ('GA', 'GA', 'GA', 'CT', 'CT', 'CT'),
                ('May', 'Aug.', 'May', 'June', 'July', 'July'),
                (97, 20, 100, 70, 100, 20),
            ],
        )
        self.assertEqual(
            mi.sort_together(
                iterables,
                key_list=(0, 2),
                key=lambda state, number: (
                    number if state == 'CT' else 2 * number
                ),
            ),
            [
                ('CT', 'GA', 'CT', 'CT', 'GA', 'GA'),
                ('July', 'Aug.', 'June', 'July', 'May', 'May'),
                (20, 20, 70, 100, 97, 100),
            ],
        )

    def test_reverse(self):
        """tests `reverse` to ensure a reverse sort for `key_list` iterables"""
        iterables = [
            ['GA', 'GA', 'GA', 'CT', 'CT', 'CT'],
            ['May', 'Aug.', 'May', 'June', 'July', 'July'],
            [97, 20, 100, 70, 100, 20],
        ]

        self.assertEqual(
            mi.sort_together(iterables, key_list=(0, 1, 2), reverse=True),
            [
                ('GA', 'GA', 'GA', 'CT', 'CT', 'CT'),
                ('May', 'May', 'Aug.', 'June', 'July', 'July'),
                (100, 97, 20, 70, 100, 20),
            ],
        )

    def test_uneven_iterables(self):
        """tests trimming of iterables to the shortest length before sorting"""
        iterables = [
            ['GA', 'GA', 'GA', 'CT', 'CT', 'CT', 'MA'],
            ['May', 'Aug.', 'May', 'June', 'July', 'July'],
            [97, 20, 100, 70, 100, 20, 0],
        ]

        self.assertEqual(
            mi.sort_together(iterables),
            [
                ('CT', 'CT', 'CT', 'GA', 'GA', 'GA'),
                ('June', 'July', 'July', 'May', 'Aug.', 'May'),
                (70, 100, 20, 97, 20, 100),
            ],
        )


class DivideTest(TestCase):
    """Tests for divide()"""

    def test_invalid_n(self):
        self.assertRaises(ValueError, lambda: mi.divide(-1, [1, 2, 3]))
        self.assertRaises(ValueError, lambda: mi.divide(0, [1, 2, 3]))

    def test_basic(self):
        iterable = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        for n, expected in [
            (1, [iterable]),
            (2, [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]),
            (3, [[1, 2, 3, 4], [5, 6, 7], [8, 9, 10]]),
            (10, [[n] for n in range(1, 10 + 1)]),
        ]:
            self.assertEqual(
                [list(x) for x in mi.divide(n, iterable)], expected
            )

    def test_large_n(self):
        self.assertEqual(
            [list(x) for x in mi.divide(6, iter(range(1, 4 + 1)))],
            [[1], [2], [3], [4], [], []],
        )


class TestAlwaysIterable(TestCase):
    """Tests for always_iterable()"""

    def test_single(self):
        self.assertEqual(list(mi.always_iterable(1)), [1])

    def test_strings(self):
        for obj in ['foo', b'bar', 'baz']:
            actual = list(mi.always_iterable(obj))
            expected = [obj]
            self.assertEqual(actual, expected)

    def test_base_type(self):
        dict_obj = {'a': 1, 'b': 2}
        str_obj = '123'

        # Default: dicts are iterable like they normally are
        default_actual = list(mi.always_iterable(dict_obj))
        default_expected = list(dict_obj)
        self.assertEqual(default_actual, default_expected)

        # Unitary types set: dicts are not iterable
        custom_actual = list(mi.always_iterable(dict_obj, base_type=dict))
        custom_expected = [dict_obj]
        self.assertEqual(custom_actual, custom_expected)

        # With unitary types set, strings are iterable
        str_actual = list(mi.always_iterable(str_obj, base_type=None))
        str_expected = list(str_obj)
        self.assertEqual(str_actual, str_expected)

        # base_type handles nested tuple (via isinstance).
        base_type = ((dict,),)
        custom_actual = list(mi.always_iterable(dict_obj, base_type=base_type))
        custom_expected = [dict_obj]
        self.assertEqual(custom_actual, custom_expected)

    def test_iterables(self):
        self.assertEqual(list(mi.always_iterable([0, 1])), [0, 1])
        self.assertEqual(
            list(mi.always_iterable([0, 1], base_type=list)), [[0, 1]]
        )
        self.assertEqual(
            list(mi.always_iterable(iter('foo'))), ['f', 'o', 'o']
        )
        self.assertEqual(list(mi.always_iterable([])), [])

    def test_none(self):
        self.assertEqual(list(mi.always_iterable(None)), [])

    def test_generator(self):
        def _gen():
            yield 0
            yield 1

        self.assertEqual(list(mi.always_iterable(_gen())), [0, 1])


class AdjacentTests(TestCase):
    def test_typical(self):
        actual = list(mi.adjacent(lambda x: x % 5 == 0, range(10)))
        expected = [
            (True, 0),
            (True, 1),
            (False, 2),
            (False, 3),
            (True, 4),
            (True, 5),
            (True, 6),
            (False, 7),
            (False, 8),
            (False, 9),
        ]
        self.assertEqual(actual, expected)

    def test_empty_iterable(self):
        actual = list(mi.adjacent(lambda x: x % 5 == 0, []))
        expected = []
        self.assertEqual(actual, expected)

    def test_length_one(self):
        actual = list(mi.adjacent(lambda x: x % 5 == 0, [0]))
        expected = [(True, 0)]
        self.assertEqual(actual, expected)

        actual = list(mi.adjacent(lambda x: x % 5 == 0, [1]))
        expected = [(False, 1)]
        self.assertEqual(actual, expected)

    def test_consecutive_true(self):
        """Test that when the predicate matches multiple consecutive elements
        it doesn't repeat elements in the output"""
        actual = list(mi.adjacent(lambda x: x % 5 < 2, range(10)))
        expected = [
            (True, 0),
            (True, 1),
            (True, 2),
            (False, 3),
            (True, 4),
            (True, 5),
            (True, 6),
            (True, 7),
            (False, 8),
            (False, 9),
        ]
        self.assertEqual(actual, expected)

    def test_distance(self):
        actual = list(mi.adjacent(lambda x: x % 5 == 0, range(10), distance=2))
        expected = [
            (True, 0),
            (True, 1),
            (True, 2),
            (True, 3),
            (True, 4),
            (True, 5),
            (True, 6),
            (True, 7),
            (False, 8),
            (False, 9),
        ]
        self.assertEqual(actual, expected)

        actual = list(mi.adjacent(lambda x: x % 5 == 0, range(10), distance=3))
        expected = [
            (True, 0),
            (True, 1),
            (True, 2),
            (True, 3),
            (True, 4),
            (True, 5),
            (True, 6),
            (True, 7),
            (True, 8),
            (False, 9),
        ]
        self.assertEqual(actual, expected)

    def test_large_distance(self):
        """Test distance larger than the length of the iterable"""
        iterable = range(10)
        actual = list(mi.adjacent(lambda x: x % 5 == 4, iterable, distance=20))
        expected = list(zip(repeat(True), iterable))
        self.assertEqual(actual, expected)

        actual = list(mi.adjacent(lambda x: False, iterable, distance=20))
        expected = list(zip(repeat(False), iterable))
        self.assertEqual(actual, expected)

    def test_zero_distance(self):
        """Test that adjacent() reduces to zip+map when distance is 0"""
        iterable = range(1000)
        predicate = lambda x: x % 4 == 2
        actual = mi.adjacent(predicate, iterable, 0)
        expected = zip(map(predicate, iterable), iterable)
        self.assertTrue(all(a == e for a, e in zip(actual, expected)))

    def test_negative_distance(self):
        """Test that adjacent() raises an error with negative distance"""
        pred = lambda x: x
        self.assertRaises(
            ValueError, lambda: mi.adjacent(pred, range(1000), -1)
        )
        self.assertRaises(
            ValueError, lambda: mi.adjacent(pred, range(10), -10)
        )

    def test_grouping(self):
        """Test interaction of adjacent() with groupby_transform()"""
        iterable = mi.adjacent(lambda x: x % 5 == 0, range(10))
        grouper = mi.groupby_transform(iterable, itemgetter(0), itemgetter(1))
        actual = [(k, list(g)) for k, g in grouper]
        expected = [
            (True, [0, 1]),
            (False, [2, 3]),
            (True, [4, 5, 6]),
            (False, [7, 8, 9]),
        ]
        self.assertEqual(actual, expected)

    def test_call_once(self):
        """Test that the predicate is only called once per item."""
        already_seen = set()
        iterable = range(10)

        def predicate(item):
            self.assertNotIn(item, already_seen)
            already_seen.add(item)
            return True

        actual = list(mi.adjacent(predicate, iterable))
        expected = [(True, x) for x in iterable]
        self.assertEqual(actual, expected)


class GroupByTransformTests(TestCase):
    def assertAllGroupsEqual(self, groupby1, groupby2):
        for a, b in zip(groupby1, groupby2):
            key1, group1 = a
            key2, group2 = b
            self.assertEqual(key1, key2)
            self.assertListEqual(list(group1), list(group2))
        self.assertRaises(StopIteration, lambda: next(groupby1))
        self.assertRaises(StopIteration, lambda: next(groupby2))

    def test_default_funcs(self):
        iterable = [(x // 5, x) for x in range(1000)]
        actual = mi.groupby_transform(iterable)
        expected = groupby(iterable)
        self.assertAllGroupsEqual(actual, expected)

    def test_valuefunc(self):
        iterable = [(int(x / 5), int(x / 3), x) for x in range(10)]

        # Test the standard usage of grouping one iterable using another's keys
        grouper = mi.groupby_transform(
            iterable, keyfunc=itemgetter(0), valuefunc=itemgetter(-1)
        )
        actual = [(k, list(g)) for k, g in grouper]
        expected = [(0, [0, 1, 2, 3, 4]), (1, [5, 6, 7, 8, 9])]
        self.assertEqual(actual, expected)

        grouper = mi.groupby_transform(
            iterable, keyfunc=itemgetter(1), valuefunc=itemgetter(-1)
        )
        actual = [(k, list(g)) for k, g in grouper]
        expected = [(0, [0, 1, 2]), (1, [3, 4, 5]), (2, [6, 7, 8]), (3, [9])]
        self.assertEqual(actual, expected)

        # and now for something a little different
        d = dict(zip(range(10), 'abcdefghij'))
        grouper = mi.groupby_transform(
            range(10), keyfunc=lambda x: x // 5, valuefunc=d.get
        )
        actual = [(k, ''.join(g)) for k, g in grouper]
        expected = [(0, 'abcde'), (1, 'fghij')]
        self.assertEqual(actual, expected)

    def test_no_valuefunc(self):
        iterable = range(1000)

        def key(x):
            return x // 5

        actual = mi.groupby_transform(iterable, key, valuefunc=None)
        expected = groupby(iterable, key)
        self.assertAllGroupsEqual(actual, expected)

        actual = mi.groupby_transform(iterable, key)  # default valuefunc
        expected = groupby(iterable, key)
        self.assertAllGroupsEqual(actual, expected)

    def test_reducefunc(self):
        iterable = range(50)
        keyfunc = lambda k: 10 * (k // 10)
        valuefunc = lambda v: v + 1
        reducefunc = sum
        actual = list(
            mi.groupby_transform(
                iterable,
                keyfunc=keyfunc,
                valuefunc=valuefunc,
                reducefunc=reducefunc,
            )
        )
        expected = [(0, 55), (10, 155), (20, 255), (30, 355), (40, 455)]
        self.assertEqual(actual, expected)


class NumericRangeTests(TestCase):
    def test_basic(self):
        for args, expected in [
            ((4,), [0, 1, 2, 3]),
            ((4.0,), [0.0, 1.0, 2.0, 3.0]),
            ((1.0, 4), [1.0, 2.0, 3.0]),
            ((1, 4.0), [1.0, 2.0, 3.0]),
            ((1.0, 5), [1.0, 2.0, 3.0, 4.0]),
            ((0, 20, 5), [0, 5, 10, 15]),
            ((0, 20, 5.0), [0.0, 5.0, 10.0, 15.0]),
            ((0, 10, 3), [0, 3, 6, 9]),
            ((0, 10, 3.0), [0.0, 3.0, 6.0, 9.0]),
            ((0, -5, -1), [0, -1, -2, -3, -4]),
            ((0.0, -5, -1), [0.0, -1.0, -2.0, -3.0, -4.0]),
            ((1, 2, Fraction(1, 2)), [Fraction(1, 1), Fraction(3, 2)]),
            ((0,), []),
            ((0.0,), []),
            ((1, 0), []),
            ((1.0, 0.0), []),
            ((0.1, 0.30000000000000001, 0.2), [0.1]),  # IEE 754 !
            (
                (
                    Decimal("0.1"),
                    Decimal("0.30000000000000001"),
                    Decimal("0.2"),
                ),
                [Decimal("0.1"), Decimal("0.3")],
            ),  # okay with Decimal
            (
                (
                    Fraction(1, 10),
                    Fraction(30000000000000001, 100000000000000000),
                    Fraction(2, 10),
                ),
                [Fraction(1, 10), Fraction(3, 10)],
            ),  # okay with Fraction
            ((Fraction(2, 1),), [Fraction(0, 1), Fraction(1, 1)]),
            ((Decimal('2.0'),), [Decimal('0.0'), Decimal('1.0')]),
            (
                (
                    datetime(2019, 3, 29, 12, 34, 56),
                    datetime(2019, 3, 29, 12, 37, 55),
                    timedelta(minutes=1),
                ),
                [
                    datetime(2019, 3, 29, 12, 34, 56),
                    datetime(2019, 3, 29, 12, 35, 56),
                    datetime(2019, 3, 29, 12, 36, 56),
                ],
            ),
        ]:
            actual = list(mi.numeric_range(*args))
            self.assertEqual(expected, actual)
            self.assertTrue(
                all(type(a) is type(e) for a, e in zip(actual, expected))
            )

    def test_arg_count(self):
        for args, message in [
            ((), 'numeric_range expected at least 1 argument, got 0'),
            (
                (0, 1, 2, 3),
                'numeric_range expected at most 3 arguments, got 4',
            ),
        ]:
            with self.assertRaisesRegex(TypeError, message):
                mi.numeric_range(*args)

    def test_zero_step(self):
        for args in [
            (1, 2, 0),
            (
                datetime(2019, 3, 29, 12, 34, 56),
                datetime(2019, 3, 29, 12, 37, 55),
                timedelta(minutes=0),
            ),
            (1.0, 2.0, 0.0),
            (Decimal("1.0"), Decimal("2.0"), Decimal("0.0")),
            (Fraction(2, 2), Fraction(4, 2), Fraction(0, 2)),
        ]:
            with self.assertRaises(ValueError):
                list(mi.numeric_range(*args))

    def test_bool(self):
        for args, expected in [
            ((1.0, 3.0, 1.5), True),
            ((1.0, 2.0, 1.5), True),
            ((1.0, 1.0, 1.5), False),
            ((1.0, 0.0, 1.5), False),
            ((3.0, 1.0, -1.5), True),
            ((2.0, 1.0, -1.5), True),
            ((1.0, 1.0, -1.5), False),
            ((0.0, 1.0, -1.5), False),
            ((Decimal("1.0"), Decimal("2.0"), Decimal("1.5")), True),
            ((Decimal("1.0"), Decimal("0.0"), Decimal("1.5")), False),
            ((Fraction(2, 2), Fraction(4, 2), Fraction(3, 2)), True),
            ((Fraction(2, 2), Fraction(0, 2), Fraction(3, 2)), False),
            (
                (
                    datetime(2019, 3, 29),
                    datetime(2019, 3, 30),
                    timedelta(hours=1),
                ),
                True,
            ),
            (
                (
                    datetime(2019, 3, 29),
                    datetime(2019, 3, 28),
                    timedelta(hours=1),
                ),
                False,
            ),
        ]:
            self.assertEqual(expected, bool(mi.numeric_range(*args)))

    def test_contains(self):
        for args, expected_in, expected_not_in in [
            ((10,), range(10), (0.5,)),
            ((1.0, 9.9, 1.5), (1.0, 2.5, 4.0, 5.5, 7.0, 8.5), (0.9,)),
            ((9.0, 1.0, -1.5), (1.5, 3.0, 4.5, 6.0, 7.5, 9.0), (0.0, 0.9)),
            (
                (Decimal("1.0"), Decimal("9.9"), Decimal("1.5")),
                (
                    Decimal("1.0"),
                    Decimal("2.5"),
                    Decimal("4.0"),
                    Decimal("5.5"),
                    Decimal("7.0"),
                    Decimal("8.5"),
                ),
                (Decimal("0.9"),),
            ),
            (
                (Fraction(0, 1), Fraction(5, 1), Fraction(1, 2)),
                (Fraction(0, 1), Fraction(1, 2), Fraction(9, 2)),
                (Fraction(10, 2),),
            ),
            (
                (
                    datetime(2019, 3, 29),
                    datetime(2019, 3, 30),
                    timedelta(hours=1),
                ),
                (datetime(2019, 3, 29, 15),),
                (datetime(2019, 3, 29, 15, 30),),
            ),
        ]:
            r = mi.numeric_range(*args)
            for v in expected_in:
                self.assertTrue(v in r)
                self.assertFalse(v not in r)

            for v in expected_not_in:
                self.assertFalse(v in r)
                self.assertTrue(v not in r)

    def test_eq(self):
        for args1, args2 in [
            ((0, 5, 2), (0, 6, 2)),
            ((1.0, 9.9, 1.5), (1.0, 8.6, 1.5)),
            ((8.5, 0.0, -1.5), (8.5, 0.7, -1.5)),
            ((7.0, 0.0, 1.0), (17.0, 7.0, 0.5)),
            (
                (Decimal("1.0"), Decimal("9.9"), Decimal("1.5")),
                (Decimal("1.0"), Decimal("8.6"), Decimal("1.5")),
            ),
            (
                (Fraction(1, 1), Fraction(10, 1), Fraction(3, 2)),
                (Fraction(1, 1), Fraction(9, 1), Fraction(3, 2)),
            ),
            (
                (
                    datetime(2019, 3, 29),
                    datetime(2019, 3, 30),
                    timedelta(hours=10),
                ),
                (
                    datetime(2019, 3, 29),
                    datetime(2019, 3, 30, 1),
                    timedelta(hours=10),
                ),
            ),
        ]:
            self.assertEqual(
                mi.numeric_range(*args1), mi.numeric_range(*args2)
            )

        for args1, args2 in [
            ((0, 5, 2), (0, 7, 2)),
            ((1.0, 9.9, 1.5), (1.2, 9.9, 1.5)),
            ((1.0, 9.9, 1.5), (1.0, 10.3, 1.5)),
            ((1.0, 9.9, 1.5), (1.0, 9.9, 1.4)),
            ((8.5, 0.0, -1.5), (8.4, 0.0, -1.5)),
            ((8.5, 0.0, -1.5), (8.5, -0.7, -1.5)),
            ((8.5, 0.0, -1.5), (8.5, 0.0, -1.4)),
            ((0.0, 7.0, 1.0), (7.0, 0.0, 1.0)),
            (
                (Decimal("1.0"), Decimal("10.0"), Decimal("1.5")),
                (Decimal("1.0"), Decimal("10.5"), Decimal("1.5")),
            ),
            (
                (Fraction(1, 1), Fraction(10, 1), Fraction(3, 2)),
                (Fraction(1, 1), Fraction(21, 2), Fraction(3, 2)),
            ),
            (
                (
                    datetime(2019, 3, 29),
                    datetime(2019, 3, 30),
                    timedelta(hours=10),
                ),
                (
                    datetime(2019, 3, 29),
                    datetime(2019, 3, 30, 15),
                    timedelta(hours=10),
                ),
            ),
        ]:
            self.assertNotEqual(
                mi.numeric_range(*args1), mi.numeric_range(*args2)
            )

        self.assertNotEqual(mi.numeric_range(7.0), 1)
        self.assertNotEqual(mi.numeric_range(7.0), "abc")

    def test_get_item_by_index(self):
        for args, index, expected in [
            ((1, 6), 2, 3),
            ((1.0, 6.0, 1.5), 0, 1.0),
            ((1.0, 6.0, 1.5), 1, 2.5),
            ((1.0, 6.0, 1.5), 2, 4.0),
            ((1.0, 6.0, 1.5), 3, 5.5),
            ((1.0, 6.0, 1.5), -1, 5.5),
            ((1.0, 6.0, 1.5), -2, 4.0),
            (
                (Decimal("1.0"), Decimal("9.0"), Decimal("1.5")),
                -1,
                Decimal("8.5"),
            ),
            (
                (Fraction(1, 1), Fraction(10, 1), Fraction(3, 2)),
                2,
                Fraction(4, 1),
            ),
            (
                (
                    datetime(2019, 3, 29),
                    datetime(2019, 3, 30),
                    timedelta(hours=10),
                ),
                1,
                datetime(2019, 3, 29, 10),
            ),
        ]:
            self.assertEqual(expected, mi.numeric_range(*args)[index])

        for args, index in [
            ((1.0, 6.0, 1.5), 4),
            ((1.0, 6.0, 1.5), -5),
            ((6.0, 1.0, 1.5), 0),
            ((6.0, 1.0, 1.5), -1),
            ((Decimal("1.0"), Decimal("9.0"), Decimal("-1.5")), -1),
            ((Fraction(1, 1), Fraction(2, 1), Fraction(3, 2)), 2),
            (
                (
                    datetime(2019, 3, 29),
                    datetime(2019, 3, 30),
                    timedelta(hours=10),
                ),
                8,
            ),
        ]:
            with self.assertRaises(IndexError):
                mi.numeric_range(*args)[index]

    def test_get_item_by_slice(self):
        for args, sl, expected_args in [
            ((1.0, 9.0, 1.5), slice(None, None, None), (1.0, 9.0, 1.5)),
            ((1.0, 9.0, 1.5), slice(None, 1, None), (1.0, 2.5, 1.5)),
            ((1.0, 9.0, 1.5), slice(None, None, 2), (1.0, 9.0, 3.0)),
            ((1.0, 9.0, 1.5), slice(None, 2, None), (1.0, 4.0, 1.5)),
            ((1.0, 9.0, 1.5), slice(1, 2, None), (2.5, 4.0, 1.5)),
            ((1.0, 9.0, 1.5), slice(1, -1, None), (2.5, 8.5, 1.5)),
            ((1.0, 9.0, 1.5), slice(10, None, 3), (9.0, 9.0, 4.5)),
            ((1.0, 9.0, 1.5), slice(-10, None, 3), (1.0, 9.0, 4.5)),
            ((1.0, 9.0, 1.5), slice(None, -10, 3), (1.0, 1.0, 4.5)),
            ((1.0, 9.0, 1.5), slice(None, 10, 3), (1.0, 9.0, 4.5)),
            (
                (Decimal("1.0"), Decimal("9.0"), Decimal("1.5")),
                slice(1, -1, None),
                (Decimal("2.5"), Decimal("8.5"), Decimal("1.5")),
            ),
            (
                (Fraction(1, 1), Fraction(5, 1), Fraction(3, 2)),
                slice(1, -1, None),
                (Fraction(5, 2), Fraction(4, 1), Fraction(3, 2)),
            ),
            (
                (
                    datetime(2019, 3, 29),
                    datetime(2019, 3, 30),
                    timedelta(hours=10),
                ),
                slice(1, -1, None),
                (
                    datetime(2019, 3, 29, 10),
                    datetime(2019, 3, 29, 20),
                    timedelta(hours=10),
                ),
            ),
        ]:
            self.assertEqual(
                mi.numeric_range(*expected_args), mi.numeric_range(*args)[sl]
            )

    def test_hash(self):
        for args, expected in [
            ((1.0, 6.0, 1.5), hash((1.0, 5.5, 1.5))),
            ((1.0, 7.0, 1.5), hash((1.0, 5.5, 1.5))),
            ((1.0, 7.5, 1.5), hash((1.0, 7.0, 1.5))),
            ((1.0, 1.5, 1.5), hash((1.0, 1.0, 1.5))),
            ((1.5, 1.0, 1.5), hash(range(0, 0))),
            ((1.5, 1.5, 1.5), hash(range(0, 0))),
            (
                (Decimal("1.0"), Decimal("9.0"), Decimal("1.5")),
                hash((Decimal("1.0"), Decimal("8.5"), Decimal("1.5"))),
            ),
            (
                (Fraction(1, 1), Fraction(5, 1), Fraction(3, 2)),
                hash((Fraction(1, 1), Fraction(4, 1), Fraction(3, 2))),
            ),
            (
                (
                    datetime(2019, 3, 29),
                    datetime(2019, 3, 30),
                    timedelta(hours=10),
                ),
                hash(
                    (
                        datetime(2019, 3, 29),
                        datetime(2019, 3, 29, 20),
                        timedelta(hours=10),
                    )
                ),
            ),
        ]:
            self.assertEqual(expected, hash(mi.numeric_range(*args)))

    def test_iter_twice(self):
        r1 = mi.numeric_range(1.0, 9.9, 1.5)
        r2 = mi.numeric_range(8.5, 0.0, -1.5)
        self.assertEqual([1.0, 2.5, 4.0, 5.5, 7.0, 8.5], list(r1))
        self.assertEqual([1.0, 2.5, 4.0, 5.5, 7.0, 8.5], list(r1))
        self.assertEqual([8.5, 7.0, 5.5, 4.0, 2.5, 1.0], list(r2))
        self.assertEqual([8.5, 7.0, 5.5, 4.0, 2.5, 1.0], list(r2))

    def test_len(self):
        for args, expected in [
            ((1.0, 7.0, 1.5), 4),
            ((1.0, 7.01, 1.5), 5),
            ((7.0, 1.0, -1.5), 4),
            ((7.01, 1.0, -1.5), 5),
            ((0.1, 0.30000000000000001, 0.2), 1),  # IEE 754 !
            (
                (
                    Decimal("0.1"),
                    Decimal("0.30000000000000001"),
                    Decimal("0.2"),
                ),
                2,
            ),  # works with Decimal
            ((Decimal("1.0"), Decimal("9.0"), Decimal("1.5")), 6),
            ((Fraction(1, 1), Fraction(5, 1), Fraction(3, 2)), 3),
            (
                (
                    datetime(2019, 3, 29),
                    datetime(2019, 3, 30),
                    timedelta(hours=10),
                ),
                3,
            ),
        ]:
            self.assertEqual(expected, len(mi.numeric_range(*args)))

    def test_repr(self):
        for args, *expected in [
            ((7.0,), "numeric_range(0.0, 7.0)"),
            ((1.0, 7.0), "numeric_range(1.0, 7.0)"),
            ((7.0, 1.0, -1.5), "numeric_range(7.0, 1.0, -1.5)"),
            (
                (Decimal("1.0"), Decimal("9.0"), Decimal("1.5")),
                (
                    "numeric_range(Decimal('1.0'), Decimal('9.0'), "
                    "Decimal('1.5'))"
                ),
            ),
            (
                (Fraction(7, 7), Fraction(10, 2), Fraction(3, 2)),
                (
                    "numeric_range(Fraction(1, 1), Fraction(5, 1), "
                    "Fraction(3, 2))"
                ),
            ),
            (
                (
                    datetime(2019, 3, 29),
                    datetime(2019, 3, 30),
                    timedelta(hours=10),
                ),
                "numeric_range(datetime.datetime(2019, 3, 29, 0, 0), "
                "datetime.datetime(2019, 3, 30, 0, 0), "
                "datetime.timedelta(seconds=36000))",
                "numeric_range(datetime.datetime(2019, 3, 29, 0, 0), "
                "datetime.datetime(2019, 3, 30, 0, 0), "
                "datetime.timedelta(0, 36000))",
            ),
        ]:
            with self.subTest(args=args):
                self.assertIn(repr(mi.numeric_range(*args)), expected)

    def test_reversed(self):
        for args, expected in [
            ((7.0,), [6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0]),
            ((1.0, 7.0), [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]),
            ((7.0, 1.0, -1.5), [2.5, 4.0, 5.5, 7.0]),
            ((7.0, 0.9, -1.5), [1.0, 2.5, 4.0, 5.5, 7.0]),
            (
                (Decimal("1.0"), Decimal("5.0"), Decimal("1.5")),
                [Decimal('4.0'), Decimal('2.5'), Decimal('1.0')],
            ),
            (
                (Fraction(1, 1), Fraction(5, 1), Fraction(3, 2)),
                [Fraction(4, 1), Fraction(5, 2), Fraction(1, 1)],
            ),
            (
                (
                    datetime(2019, 3, 29),
                    datetime(2019, 3, 30),
                    timedelta(hours=10),
                ),
                [
                    datetime(2019, 3, 29, 20),
                    datetime(2019, 3, 29, 10),
                    datetime(2019, 3, 29),
                ],
            ),
        ]:
            self.assertEqual(expected, list(reversed(mi.numeric_range(*args))))

    def test_count(self):
        for args, v, c in [
            ((7.0,), 0.0, 1),
            ((7.0,), 0.5, 0),
            ((7.0,), 6.0, 1),
            ((7.0,), 7.0, 0),
            ((7.0,), 10.0, 0),
            (
                (Decimal("1.0"), Decimal("5.0"), Decimal("1.5")),
                Decimal('4.0'),
                1,
            ),
            (
                (Fraction(1, 1), Fraction(5, 1), Fraction(3, 2)),
                Fraction(5, 2),
                1,
            ),
            (
                (
                    datetime(2019, 3, 29),
                    datetime(2019, 3, 30),
                    timedelta(hours=10),
                ),
                datetime(2019, 3, 29, 20),
                1,
            ),
        ]:
            self.assertEqual(c, mi.numeric_range(*args).count(v))

    def test_index(self):
        for args, v, i in [
            ((7.0,), 0.0, 0),
            ((7.0,), 6.0, 6),
            ((7.0, 0.0, -1.0), 7.0, 0),
            ((7.0, 0.0, -1.0), 1.0, 6),
            (
                (Decimal("1.0"), Decimal("5.0"), Decimal("1.5")),
                Decimal('4.0'),
                2,
            ),
            (
                (Fraction(1, 1), Fraction(5, 1), Fraction(3, 2)),
                Fraction(5, 2),
                1,
            ),
            (
                (
                    datetime(2019, 3, 29),
                    datetime(2019, 3, 30),
                    timedelta(hours=10),
                ),
                datetime(2019, 3, 29, 20),
                2,
            ),
        ]:
            self.assertEqual(i, mi.numeric_range(*args).index(v))

        for args, v in [
            ((0.7,), 0.5),
            ((0.7,), 7.0),
            ((0.7,), 10.0),
            ((7.0, 0.0, -1.0), 0.5),
            ((7.0, 0.0, -1.0), 0.0),
            ((7.0, 0.0, -1.0), 10.0),
            ((7.0, 0.0), 5.0),
            ((Decimal("1.0"), Decimal("5.0"), Decimal("1.5")), Decimal('4.5')),
            ((Fraction(1, 1), Fraction(5, 1), Fraction(3, 2)), Fraction(5, 3)),
            (
                (
                    datetime(2019, 3, 29),
                    datetime(2019, 3, 30),
                    timedelta(hours=10),
                ),
                datetime(2019, 3, 30),
            ),
        ]:
            with self.assertRaises(ValueError):
                mi.numeric_range(*args).index(v)

    def test_parent_classes(self):
        r = mi.numeric_range(7.0)
        self.assertTrue(isinstance(r, abc.Iterable))
        self.assertFalse(isinstance(r, abc.Iterator))
        self.assertTrue(isinstance(r, abc.Sequence))
        self.assertTrue(isinstance(r, abc.Hashable))

    def test_bad_key(self):
        r = mi.numeric_range(7.0)
        for arg, message in [
            ('a', 'numeric range indices must be integers or slices, not str'),
            (
                (),
                'numeric range indices must be integers or slices, not tuple',
            ),
        ]:
            with self.assertRaisesRegex(TypeError, message):
                r[arg]

    def test_pickle(self):
        for args in [
            (7.0,),
            (5.0, 7.0),
            (5.0, 7.0, 3.0),
            (7.0, 5.0),
            (7.0, 5.0, 4.0),
            (7.0, 5.0, -1.0),
            (Decimal("1.0"), Decimal("5.0"), Decimal("1.5")),
            (Fraction(1, 1), Fraction(5, 1), Fraction(3, 2)),
            (datetime(2019, 3, 29), datetime(2019, 3, 30)),
        ]:
            r = mi.numeric_range(*args)
            self.assertTrue(dumps(r))  # assert not empty
            self.assertEqual(r, loads(dumps(r)))


class CountCycleTests(TestCase):
    def test_basic(self):
        expected = [
            (0, 'a'),
            (0, 'b'),
            (0, 'c'),
            (1, 'a'),
            (1, 'b'),
            (1, 'c'),
            (2, 'a'),
            (2, 'b'),
            (2, 'c'),
        ]
        for actual in [
            mi.take(9, mi.count_cycle('abc')),  # n=None
            list(mi.count_cycle('abc', 3)),  # n=3
        ]:
            self.assertEqual(actual, expected)

    def test_empty(self):
        self.assertEqual(list(mi.count_cycle('')), [])
        self.assertEqual(list(mi.count_cycle('', 2)), [])

    def test_negative(self):
        self.assertEqual(list(mi.count_cycle('abc', -3)), [])


class MarkEndsTests(TestCase):
    def test_basic(self):
        for size, expected in [
            (0, []),
            (1, [(True, True, '0')]),
            (2, [(True, False, '0'), (False, True, '1')]),
            (3, [(True, False, '0'), (False, False, '1'), (False, True, '2')]),
            (
                4,
                [
                    (True, False, '0'),
                    (False, False, '1'),
                    (False, False, '2'),
                    (False, True, '3'),
                ],
            ),
        ]:
            with self.subTest(size=size):
                iterable = map(str, range(size))
                actual = list(mi.mark_ends(iterable))
                self.assertEqual(actual, expected)


class LocateTests(TestCase):
    def test_default_pred(self):
        iterable = [0, 1, 1, 0, 1, 0, 0]
        actual = list(mi.locate(iterable))
        expected = [1, 2, 4]
        self.assertEqual(actual, expected)

    def test_no_matches(self):
        iterable = [0, 0, 0]
        actual = list(mi.locate(iterable))
        expected = []
        self.assertEqual(actual, expected)

    def test_custom_pred(self):
        iterable = ['0', 1, 1, '0', 1, '0', '0']
        pred = lambda x: x == '0'
        actual = list(mi.locate(iterable, pred))
        expected = [0, 3, 5, 6]
        self.assertEqual(actual, expected)

    def test_window_size(self):
        iterable = ['0', 1, 1, '0', 1, '0', '0']
        pred = lambda *args: args == ('0', 1)
        actual = list(mi.locate(iterable, pred, window_size=2))
        expected = [0, 3]
        self.assertEqual(actual, expected)

    def test_window_size_large(self):
        iterable = [1, 2, 3, 4]
        pred = lambda a, b, c, d, e: True
        actual = list(mi.locate(iterable, pred, window_size=5))
        expected = [0]
        self.assertEqual(actual, expected)

    def test_window_size_zero(self):
        iterable = [1, 2, 3, 4]
        pred = lambda: True
        with self.assertRaises(ValueError):
            list(mi.locate(iterable, pred, window_size=0))


class StripFunctionTests(TestCase):
    def test_hashable(self):
        iterable = list('www.example.com')
        pred = lambda x: x in set('cmowz.')

        self.assertEqual(list(mi.lstrip(iterable, pred)), list('example.com'))
        self.assertEqual(list(mi.rstrip(iterable, pred)), list('www.example'))
        self.assertEqual(list(mi.strip(iterable, pred)), list('example'))

    def test_not_hashable(self):
        iterable = [
            list('http://'),
            list('www'),
            list('.example'),
            list('.com'),
        ]
        pred = lambda x: x in [list('http://'), list('www'), list('.com')]

        self.assertEqual(list(mi.lstrip(iterable, pred)), iterable[2:])
        self.assertEqual(list(mi.rstrip(iterable, pred)), iterable[:3])
        self.assertEqual(list(mi.strip(iterable, pred)), iterable[2:3])

    def test_math(self):
        iterable = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2]
        pred = lambda x: x <= 2

        self.assertEqual(list(mi.lstrip(iterable, pred)), iterable[3:])
        self.assertEqual(list(mi.rstrip(iterable, pred)), iterable[:-3])
        self.assertEqual(list(mi.strip(iterable, pred)), iterable[3:-3])


class IsliceExtendedTests(TestCase):
    def test_all(self):
        iterable = ['0', '1', '2', '3', '4', '5']
        indexes = [*range(-4, 10), None]
        steps = [1, 2, 3, 4, -1, -2, -3, -4]
        for slice_args in product(indexes, indexes, steps):
            with self.subTest(slice_args=slice_args):
                actual = list(mi.islice_extended(iterable, *slice_args))
                expected = iterable[slice(*slice_args)]
                self.assertEqual(actual, expected, slice_args)

    def test_zero_step(self):
        with self.assertRaises(ValueError):
            list(mi.islice_extended([1, 2, 3], 0, 1, 0))

    def test_slicing(self):
        iterable = map(str, count())
        first_slice = mi.islice_extended(iterable)[10:]
        second_slice = mi.islice_extended(first_slice)[:10]
        third_slice = mi.islice_extended(second_slice)[::2]
        self.assertEqual(list(third_slice), ['10', '12', '14', '16', '18'])

    def test_slicing_extensive(self):
        iterable = range(10)
        options = (None, 1, 2, 7, -1)
        for start, stop, step in product(options, options, options):
            with self.subTest(slice_args=(start, stop, step)):
                sliced_tuple_0 = tuple(
                    mi.islice_extended(iterable)[start:stop:step]
                )
                sliced_tuple_1 = tuple(
                    mi.islice_extended(iterable, start, stop, step)
                )
                sliced_range = tuple(iterable[start:stop:step])
                self.assertEqual(sliced_tuple_0, sliced_range)
                self.assertEqual(sliced_tuple_1, sliced_range)

    def test_invalid_slice(self):
        with self.assertRaises(TypeError):
            mi.islice_extended(count())[13]


class ConsecutiveGroupsTest(TestCase):
    def test_numbers(self):
        iterable = [-10, -8, -7, -6, 1, 2, 4, 5, -1, 7]
        actual = [list(g) for g in mi.consecutive_groups(iterable)]
        expected = [[-10], [-8, -7, -6], [1, 2], [4, 5], [-1], [7]]
        self.assertEqual(actual, expected)

    def test_custom_ordering(self):
        iterable = ['1', '10', '11', '20', '21', '22', '30', '31']
        ordering = lambda x: int(x)
        actual = [list(g) for g in mi.consecutive_groups(iterable, ordering)]
        expected = [['1'], ['10', '11'], ['20', '21', '22'], ['30', '31']]
        self.assertEqual(actual, expected)

    def test_exotic_ordering(self):
        iterable = [
            ('a', 'b', 'c', 'd'),
            ('a', 'c', 'b', 'd'),
            ('a', 'c', 'd', 'b'),
            ('a', 'd', 'b', 'c'),
            ('d', 'b', 'c', 'a'),
            ('d', 'c', 'a', 'b'),
        ]
        ordering = list(permutations('abcd')).index
        actual = [list(g) for g in mi.consecutive_groups(iterable, ordering)]
        expected = [
            [('a', 'b', 'c', 'd')],
            [('a', 'c', 'b', 'd'), ('a', 'c', 'd', 'b'), ('a', 'd', 'b', 'c')],
            [('d', 'b', 'c', 'a'), ('d', 'c', 'a', 'b')],
        ]
        self.assertEqual(actual, expected)


class DifferenceTest(TestCase):
    def test_normal(self):
        iterable = [10, 20, 30, 40, 50]
        actual = list(mi.difference(iterable))
        expected = [10, 10, 10, 10, 10]
        self.assertEqual(actual, expected)

    def test_custom(self):
        iterable = [10, 20, 30, 40, 50]
        actual = list(mi.difference(iterable, add))
        expected = [10, 30, 50, 70, 90]
        self.assertEqual(actual, expected)

    def test_roundtrip(self):
        original = list(range(100))
        accumulated = accumulate(original)
        actual = list(mi.difference(accumulated))
        self.assertEqual(actual, original)

    def test_one(self):
        self.assertEqual(list(mi.difference([0])), [0])

    def test_empty(self):
        self.assertEqual(list(mi.difference([])), [])

    def test_initial(self):
        original = list(range(100))
        accumulated = accumulate(original, initial=100)
        actual = list(mi.difference(accumulated, initial=100))
        self.assertEqual(actual, original)


class SeekableTest(PeekableMixinTests, TestCase):
    cls = mi.seekable

    def test_exhaustion_reset(self):
        iterable = [str(n) for n in range(10)]

        s = mi.seekable(iterable)
        self.assertEqual(list(s), iterable)  # Normal iteration
        self.assertEqual(list(s), [])  # Iterable is exhausted

        s.seek(0)
        self.assertEqual(list(s), iterable)  # Back in action

    def test_partial_reset(self):
        iterable = [str(n) for n in range(10)]

        s = mi.seekable(iterable)
        self.assertEqual(mi.take(5, s), iterable[:5])  # Normal iteration

        s.seek(1)
        self.assertEqual(list(s), iterable[1:])  # Get the rest of the iterable

    def test_forward(self):
        iterable = [str(n) for n in range(10)]

        s = mi.seekable(iterable)
        self.assertEqual(mi.take(1, s), iterable[:1])  # Normal iteration

        s.seek(3)  # Skip over index 2
        self.assertEqual(list(s), iterable[3:])  # Result is similar to slicing

        s.seek(0)  # Back to 0
        self.assertEqual(list(s), iterable)  # No difference in result

    def test_past_end(self):
        iterable = [str(n) for n in range(10)]

        s = mi.seekable(iterable)
        self.assertEqual(mi.take(1, s), iterable[:1])  # Normal iteration

        s.seek(20)
        self.assertEqual(list(s), [])  # Iterable is exhausted

        s.seek(0)  # Back to 0
        self.assertEqual(list(s), iterable)  # No difference in result

    def test_elements(self):
        iterable = map(str, count())

        s = mi.seekable(iterable)
        mi.take(10, s)

        elements = s.elements()
        self.assertEqual(
            [elements[i] for i in range(10)], [str(n) for n in range(10)]
        )
        self.assertEqual(len(elements), 10)

        mi.take(10, s)
        self.assertEqual(list(elements), [str(n) for n in range(20)])

    def test_maxlen(self):
        iterable = map(str, count())

        s = mi.seekable(iterable, maxlen=4)
        self.assertEqual(mi.take(10, s), [str(n) for n in range(10)])
        self.assertEqual(list(s.elements()), ['6', '7', '8', '9'])

        s.seek(0)
        self.assertEqual(mi.take(14, s), [str(n) for n in range(6, 20)])
        self.assertEqual(list(s.elements()), ['16', '17', '18', '19'])

    def test_maxlen_zero(self):
        iterable = [str(x) for x in range(5)]
        s = mi.seekable(iterable, maxlen=0)
        self.assertEqual(list(s), iterable)
        self.assertEqual(list(s.elements()), [])

    def test_relative_seek(self):
        iterable = [str(x) for x in range(5)]
        s = mi.seekable(iterable)
        s.relative_seek(2)
        self.assertEqual(next(s), '2')
        s.relative_seek(-2)
        self.assertEqual(next(s), '1')
        s.relative_seek(-10)  # Lower bound
        self.assertEqual(next(s), '0')
        s.relative_seek(10)  # Lower bound
        self.assertEqual(list(s.elements()), [str(x) for x in range(5)])


class SequenceViewTests(TestCase):
    def test_init(self):
        view = mi.SequenceView((1, 2, 3))
        self.assertEqual(repr(view), "SequenceView((1, 2, 3))")
        self.assertRaises(TypeError, lambda: mi.SequenceView({}))

    def test_update(self):
        seq = [1, 2, 3]
        view = mi.SequenceView(seq)
        self.assertEqual(len(view), 3)
        self.assertEqual(repr(view), "SequenceView([1, 2, 3])")

        seq.pop()
        self.assertEqual(len(view), 2)
        self.assertEqual(repr(view), "SequenceView([1, 2])")

    def test_indexing(self):
        seq = ('a', 'b', 'c', 'd', 'e', 'f')
        view = mi.SequenceView(seq)
        for i in range(-len(seq), len(seq)):
            self.assertEqual(view[i], seq[i])

    def test_slicing(self):
        seq = ('a', 'b', 'c', 'd', 'e', 'f')
        view = mi.SequenceView(seq)
        n = len(seq)
        indexes = list(range(-n - 1, n + 1)) + [None]
        steps = list(range(-n, n + 1))
        steps.remove(0)
        for slice_args in product(indexes, indexes, steps):
            i = slice(*slice_args)
            self.assertEqual(view[i], seq[i])

    def test_abc_methods(self):
        # collections.Sequence should provide all of this functionality
        seq = ('a', 'b', 'c', 'd', 'e', 'f', 'f')
        view = mi.SequenceView(seq)

        # __contains__
        self.assertIn('b', view)
        self.assertNotIn('g', view)

        # __iter__
        self.assertEqual(list(iter(view)), list(seq))

        # __reversed__
        self.assertEqual(list(reversed(view)), list(reversed(seq)))

        # index
        self.assertEqual(view.index('b'), 1)

        # count
        self.assertEqual(seq.count('f'), 2)


class RunLengthTest(TestCase):
    def test_encode(self):
        iterable = (int(str(n)[0]) for n in count(800))
        actual = mi.take(4, mi.run_length.encode(iterable))
        expected = [(8, 100), (9, 100), (1, 1000), (2, 1000)]
        self.assertEqual(actual, expected)

    def test_decode(self):
        iterable = [('d', 4), ('c', 3), ('b', 2), ('a', 1)]
        actual = ''.join(mi.run_length.decode(iterable))
        expected = 'ddddcccbba'
        self.assertEqual(actual, expected)


class ExactlyNTests(TestCase):
    """Tests for ``exactly_n()``"""

    def test_true(self):
        """Iterable has ``n`` ``True`` elements"""
        self.assertTrue(mi.exactly_n([True, False, True], 2))
        self.assertTrue(mi.exactly_n([1, 1, 1, 0], 3))
        self.assertTrue(mi.exactly_n([False, False], 0))
        self.assertTrue(mi.exactly_n(range(100), 10, lambda x: x < 10))

    def test_false(self):
        """Iterable does not have ``n`` ``True`` elements"""
        self.assertFalse(mi.exactly_n([True, False, False], 2))
        self.assertFalse(mi.exactly_n([True, True, False], 1))
        self.assertFalse(mi.exactly_n([False], 1))
        self.assertFalse(mi.exactly_n([True], -1))
        self.assertFalse(mi.exactly_n(repeat(True), 100))

    def test_empty(self):
        """Return ``True`` if the iterable is empty and ``n`` is 0"""
        self.assertTrue(mi.exactly_n([], 0))
        self.assertFalse(mi.exactly_n([], 1))


class AlwaysReversibleTests(TestCase):
    """Tests for ``always_reversible()``"""

    def test_regular_reversed(self):
        self.assertEqual(
            list(reversed(range(10))), list(mi.always_reversible(range(10)))
        )
        self.assertEqual(
            list(reversed([1, 2, 3])), list(mi.always_reversible([1, 2, 3]))
        )
        self.assertEqual(
            reversed([1, 2, 3]).__class__,
            mi.always_reversible([1, 2, 3]).__class__,
        )

    def test_nonseq_reversed(self):
        # Create a non-reversible generator from a sequence
        with self.assertRaises(TypeError):
            reversed(x for x in range(10))

        self.assertEqual(
            list(reversed(range(10))),
            list(mi.always_reversible(x for x in range(10))),
        )
        self.assertEqual(
            list(reversed([1, 2, 3])),
            list(mi.always_reversible(x for x in [1, 2, 3])),
        )
        self.assertNotEqual(
            reversed((1, 2)).__class__,
            mi.always_reversible(x for x in (1, 2)).__class__,
        )


class CircularShiftsTests(TestCase):
    def test_empty(self):
        # empty iterable -> empty list
        self.assertEqual(list(mi.circular_shifts([])), [])

    def test_simple_circular_shifts(self):
        # test the a simple iterator case
        self.assertEqual(
            mi.circular_shifts(range(4)),
            [(0, 1, 2, 3), (1, 2, 3, 0), (2, 3, 0, 1), (3, 0, 1, 2)],
        )

    def test_duplicates(self):
        # test non-distinct entries
        self.assertEqual(
            mi.circular_shifts([0, 1, 0, 1]),
            [(0, 1, 0, 1), (1, 0, 1, 0), (0, 1, 0, 1), (1, 0, 1, 0)],
        )


class MakeDecoratorTests(TestCase):
    def test_basic(self):
        slicer = mi.make_decorator(islice)

        @slicer(1, 10, 2)
        def user_function(arg_1, arg_2, kwarg_1=None):
            self.assertEqual(arg_1, 'arg_1')
            self.assertEqual(arg_2, 'arg_2')
            self.assertEqual(kwarg_1, 'kwarg_1')
            return map(str, count())

        it = user_function('arg_1', 'arg_2', kwarg_1='kwarg_1')
        actual = list(it)
        expected = ['1', '3', '5', '7', '9']
        self.assertEqual(actual, expected)

    def test_result_index(self):
        def stringify(*args, **kwargs):
            self.assertEqual(args[0], 'arg_0')
            iterable = args[1]
            self.assertEqual(args[2], 'arg_2')
            self.assertEqual(kwargs['kwarg_1'], 'kwarg_1')
            return map(str, iterable)

        stringifier = mi.make_decorator(stringify, result_index=1)

        @stringifier('arg_0', 'arg_2', kwarg_1='kwarg_1')
        def user_function(n):
            return count(n)

        it = user_function(1)
        actual = mi.take(5, it)
        expected = ['1', '2', '3', '4', '5']
        self.assertEqual(actual, expected)

    def test_wrap_class(self):
        seeker = mi.make_decorator(mi.seekable)

        @seeker()
        def user_function(n):
            return map(str, range(n))

        it = user_function(5)
        self.assertEqual(list(it), ['0', '1', '2', '3', '4'])

        it.seek(0)
        self.assertEqual(list(it), ['0', '1', '2', '3', '4'])


class MapReduceTests(TestCase):
    def test_default(self):
        iterable = (str(x) for x in range(5))
        keyfunc = lambda x: int(x) // 2
        actual = sorted(mi.map_reduce(iterable, keyfunc).items())
        expected = [(0, ['0', '1']), (1, ['2', '3']), (2, ['4'])]
        self.assertEqual(actual, expected)

    def test_valuefunc(self):
        iterable = (str(x) for x in range(5))
        keyfunc = lambda x: int(x) // 2
        valuefunc = int
        actual = sorted(mi.map_reduce(iterable, keyfunc, valuefunc).items())
        expected = [(0, [0, 1]), (1, [2, 3]), (2, [4])]
        self.assertEqual(actual, expected)

    def test_reducefunc(self):
        iterable = (str(x) for x in range(5))
        keyfunc = lambda x: int(x) // 2
        valuefunc = int
        reducefunc = lambda value_list: reduce(mul, value_list, 1)
        actual = sorted(
            mi.map_reduce(iterable, keyfunc, valuefunc, reducefunc).items()
        )
        expected = [(0, 0), (1, 6), (2, 4)]
        self.assertEqual(actual, expected)

    def test_ret(self):
        d = mi.map_reduce([1, 0, 2, 0, 1, 0], bool)
        self.assertEqual(d, {False: [0, 0, 0], True: [1, 2, 1]})
        self.assertRaises(KeyError, lambda: d[None].append(1))


class RlocateTests(TestCase):
    def test_default_pred(self):
        iterable = [0, 1, 1, 0, 1, 0, 0]
        for it in (iterable[:], iter(iterable)):
            actual = list(mi.rlocate(it))
            expected = [4, 2, 1]
            self.assertEqual(actual, expected)

    def test_no_matches(self):
        iterable = [0, 0, 0]
        for it in (iterable[:], iter(iterable)):
            actual = list(mi.rlocate(it))
            expected = []
            self.assertEqual(actual, expected)

    def test_custom_pred(self):
        iterable = ['0', 1, 1, '0', 1, '0', '0']
        pred = lambda x: x == '0'
        for it in (iterable[:], iter(iterable)):
            actual = list(mi.rlocate(it, pred))
            expected = [6, 5, 3, 0]
            self.assertEqual(actual, expected)

    def test_efficient_reversal(self):
        iterable = range(9**9)  # Is efficiently reversible
        target = 9**9 - 2
        pred = lambda x: x == target  # Find-able from the right
        actual = next(mi.rlocate(iterable, pred))
        self.assertEqual(actual, target)

    def test_window_size(self):
        iterable = ['0', 1, 1, '0', 1, '0', '0']
        pred = lambda *args: args == ('0', 1)
        for it in (iterable, iter(iterable)):
            actual = list(mi.rlocate(it, pred, window_size=2))
            expected = [3, 0]
            self.assertEqual(actual, expected)

    def test_window_size_large(self):
        iterable = [1, 2, 3, 4]
        pred = lambda a, b, c, d, e: True
        for it in (iterable, iter(iterable)):
            actual = list(mi.rlocate(iterable, pred, window_size=5))
            expected = [0]
            self.assertEqual(actual, expected)

    def test_window_size_zero(self):
        iterable = [1, 2, 3, 4]
        pred = lambda: True
        for it in (iterable, iter(iterable)):
            with self.assertRaises(ValueError):
                list(mi.locate(iterable, pred, window_size=0))


class ReplaceTests(TestCase):
    def test_basic(self):
        iterable = range(10)
        pred = lambda x: x % 2 == 0
        substitutes = []
        actual = list(mi.replace(iterable, pred, substitutes))
        expected = [1, 3, 5, 7, 9]
        self.assertEqual(actual, expected)

    def test_count(self):
        iterable = range(10)
        pred = lambda x: x % 2 == 0
        substitutes = []
        actual = list(mi.replace(iterable, pred, substitutes, count=4))
        expected = [1, 3, 5, 7, 8, 9]
        self.assertEqual(actual, expected)

    def test_window_size(self):
        iterable = range(10)
        pred = lambda *args: args == (0, 1, 2)
        substitutes = []
        actual = list(mi.replace(iterable, pred, substitutes, window_size=3))
        expected = [3, 4, 5, 6, 7, 8, 9]
        self.assertEqual(actual, expected)

    def test_window_size_end(self):
        iterable = range(10)
        pred = lambda *args: args == (7, 8, 9)
        substitutes = []
        actual = list(mi.replace(iterable, pred, substitutes, window_size=3))
        expected = [0, 1, 2, 3, 4, 5, 6]
        self.assertEqual(actual, expected)

    def test_window_size_count(self):
        iterable = range(10)
        pred = lambda *args: (args == (0, 1, 2)) or (args == (7, 8, 9))
        substitutes = []
        actual = list(
            mi.replace(iterable, pred, substitutes, count=1, window_size=3)
        )
        expected = [3, 4, 5, 6, 7, 8, 9]
        self.assertEqual(actual, expected)

    def test_window_size_large(self):
        iterable = range(4)
        pred = lambda a, b, c, d, e: True
        substitutes = [5, 6, 7]
        actual = list(mi.replace(iterable, pred, substitutes, window_size=5))
        expected = [5, 6, 7]
        self.assertEqual(actual, expected)

    def test_window_size_zero(self):
        iterable = range(10)
        pred = lambda *args: True
        substitutes = []
        with self.assertRaises(ValueError):
            list(mi.replace(iterable, pred, substitutes, window_size=0))

    def test_iterable_substitutes(self):
        iterable = range(5)
        pred = lambda x: x % 2 == 0
        substitutes = iter('__')
        actual = list(mi.replace(iterable, pred, substitutes))
        expected = ['_', '_', 1, '_', '_', 3, '_', '_']
        self.assertEqual(actual, expected)


class PartitionsTest(TestCase):
    def test_types(self):
        for iterable in ['abcd', ['a', 'b', 'c', 'd'], ('a', 'b', 'c', 'd')]:
            with self.subTest(iterable=iterable):
                actual = list(mi.partitions(iterable))
                expected = [
                    [['a', 'b', 'c', 'd']],
                    [['a'], ['b', 'c', 'd']],
                    [['a', 'b'], ['c', 'd']],
                    [['a', 'b', 'c'], ['d']],
                    [['a'], ['b'], ['c', 'd']],
                    [['a'], ['b', 'c'], ['d']],
                    [['a', 'b'], ['c'], ['d']],
                    [['a'], ['b'], ['c'], ['d']],
                ]
                self.assertEqual(actual, expected)

    def test_empty(self):
        iterable = []
        actual = list(mi.partitions(iterable))
        expected = [[[]]]
        self.assertEqual(actual, expected)

    def test_order(self):
        iterable = iter([3, 2, 1])
        actual = list(mi.partitions(iterable))
        expected = [[[3, 2, 1]], [[3], [2, 1]], [[3, 2], [1]], [[3], [2], [1]]]
        self.assertEqual(actual, expected)

    def test_duplicates(self):
        iterable = [1, 1, 1]
        actual = list(mi.partitions(iterable))
        expected = [[[1, 1, 1]], [[1], [1, 1]], [[1, 1], [1]], [[1], [1], [1]]]
        self.assertEqual(actual, expected)


class _FrozenMultiset(Set):
    """
    A helper class, useful to compare two lists without reference to the order
    of elements.

    FrozenMultiset represents a hashable set that allows duplicate elements.
    """

    def __init__(self, iterable):
        self._collection = frozenset(Counter(iterable).items())

    def __contains__(self, y):
        """
        >>> (0, 1) in _FrozenMultiset([(0, 1), (2,), (0, 1)])
        True
        """
        return any(y == x for x, _ in self._collection)

    def __iter__(self):
        """
        >>> sorted(_FrozenMultiset([(0, 1), (2,), (0, 1)]))
        [(0, 1), (0, 1), (2,)]
        """
        return (x for x, c in self._collection for _ in range(c))

    def __len__(self):
        """
        >>> len(_FrozenMultiset([(0, 1), (2,), (0, 1)]))
        3
        """
        return sum(c for x, c in self._collection)

    def has_duplicates(self):
        """
        >>> _FrozenMultiset([(0, 1), (2,), (0, 1)]).has_duplicates()
        True
        """
        return any(c != 1 for _, c in self._collection)

    def __hash__(self):
        return hash(self._collection)

    def __repr__(self):
        return "FrozenSet([{}]".format(", ".join(repr(x) for x in iter(self)))


class SetPartitionsTests(TestCase):
    @staticmethod
    def _normalize_partition(p):
        """
        Return a normalized, hashable, version of a partition using
        _FrozenMultiset
        """
        return _FrozenMultiset(_FrozenMultiset(g) for g in p)

    @staticmethod
    def _normalize_partitions(ps):
        """
        Return a normalized set of all normalized partitions using
        _FrozenMultiset
        """
        return _FrozenMultiset(
            SetPartitionsTests._normalize_partition(p) for p in ps
        )

    def test_repeated(self):
        it = 'aaa'
        actual = mi.set_partitions(it, 2)
        expected = [['a', 'aa'], ['a', 'aa'], ['a', 'aa']]
        self.assertEqual(
            self._normalize_partitions(expected),
            self._normalize_partitions(actual),
        )

    def test_each_correct(self):
        a = set(range(6))
        for p in mi.set_partitions(a):
            total = {e for g in p for e in g}
            self.assertEqual(a, total)

    def test_duplicates(self):
        a = set(range(6))
        for p in mi.set_partitions(a):
            self.assertFalse(self._normalize_partition(p).has_duplicates())

    def test_found_all(self):
        """small example, hand-checked"""
        expected = [
            [[0], [1], [2, 3, 4]],
            [[0], [1, 2], [3, 4]],
            [[0], [2], [1, 3, 4]],
            [[0], [3], [1, 2, 4]],
            [[0], [4], [1, 2, 3]],
            [[0], [1, 3], [2, 4]],
            [[0], [1, 4], [2, 3]],
            [[1], [2], [0, 3, 4]],
            [[1], [3], [0, 2, 4]],
            [[1], [4], [0, 2, 3]],
            [[1], [0, 2], [3, 4]],
            [[1], [0, 3], [2, 4]],
            [[1], [0, 4], [2, 3]],
            [[2], [3], [0, 1, 4]],
            [[2], [4], [0, 1, 3]],
            [[2], [0, 1], [3, 4]],
            [[2], [0, 3], [1, 4]],
            [[2], [0, 4], [1, 3]],
            [[3], [4], [0, 1, 2]],
            [[3], [0, 1], [2, 4]],
            [[3], [0, 2], [1, 4]],
            [[3], [0, 4], [1, 2]],
            [[4], [0, 1], [2, 3]],
            [[4], [0, 2], [1, 3]],
            [[4], [0, 3], [1, 2]],
        ]
        actual = mi.set_partitions(range(5), 3)
        self.assertEqual(
            self._normalize_partitions(expected),
            self._normalize_partitions(actual),
        )

    def test_stirling_numbers(self):
        """Check against https://en.wikipedia.org/wiki/
        Stirling_numbers_of_the_second_kind#Table_of_values"""
        cardinality_by_k_by_n = [
            [1],
            [1, 1],
            [1, 3, 1],
            [1, 7, 6, 1],
            [1, 15, 25, 10, 1],
            [1, 31, 90, 65, 15, 1],
        ]
        for n, cardinality_by_k in enumerate(cardinality_by_k_by_n, 1):
            for k, cardinality in enumerate(cardinality_by_k, 1):
                self.assertEqual(
                    cardinality, len(list(mi.set_partitions(range(n), k)))
                )

    def test_no_group(self):
        def helper():
            list(mi.set_partitions(range(4), -1))

        self.assertRaises(ValueError, helper)

    def test_to_many_groups(self):
        self.assertEqual([], list(mi.set_partitions(range(4), 5)))


class TimeLimitedTests(TestCase):
    def test_basic(self):
        def generator():
            yield 1
            yield 2
            sleep(0.2)
            yield 3

        iterable = mi.time_limited(0.1, generator())
        actual = list(iterable)
        expected = [1, 2]
        self.assertEqual(actual, expected)
        self.assertTrue(iterable.timed_out)

    def test_complete(self):
        iterable = mi.time_limited(2, iter(range(10)))
        actual = list(iterable)
        expected = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.assertEqual(actual, expected)
        self.assertFalse(iterable.timed_out)

    def test_zero_limit(self):
        iterable = mi.time_limited(0, count())
        actual = list(iterable)
        expected = []
        self.assertEqual(actual, expected)
        self.assertTrue(iterable.timed_out)

    def test_invalid_limit(self):
        with self.assertRaises(ValueError):
            list(mi.time_limited(-0.1, count()))


class OnlyTests(TestCase):
    def test_defaults(self):
        self.assertEqual(mi.only([]), None)
        self.assertEqual(mi.only([1]), 1)
        self.assertRaises(ValueError, lambda: mi.only([1, 2]))

    def test_custom_value(self):
        self.assertEqual(mi.only([], default='!'), '!')
        self.assertEqual(mi.only([1], default='!'), 1)
        self.assertRaises(ValueError, lambda: mi.only([1, 2], default='!'))

    def test_custom_exception(self):
        self.assertEqual(mi.only([], too_long=RuntimeError), None)
        self.assertEqual(mi.only([1], too_long=RuntimeError), 1)
        self.assertRaises(
            RuntimeError, lambda: mi.only([1, 2], too_long=RuntimeError)
        )

    def test_default_exception_message(self):
        self.assertRaisesRegex(
            ValueError,
            "Expected exactly one item in iterable, "
            "but got 'foo', 'bar', and perhaps more",
            lambda: mi.only(['foo', 'bar', 'baz']),
        )


class IchunkedTests(TestCase):
    def test_even(self):
        iterable = (str(x) for x in range(10))
        actual = [''.join(c) for c in mi.ichunked(iterable, 5)]
        expected = ['01234', '56789']
        self.assertEqual(actual, expected)

    def test_odd(self):
        iterable = (str(x) for x in range(10))
        actual = [''.join(c) for c in mi.ichunked(iterable, 4)]
        expected = ['0123', '4567', '89']
        self.assertEqual(actual, expected)

    def test_zero(self):
        iterable = []
        actual = [list(c) for c in mi.ichunked(iterable, 0)]
        expected = []
        self.assertEqual(actual, expected)

    def test_negative(self):
        iterable = count()
        with self.assertRaises(ValueError):
            [list(c) for c in mi.ichunked(iterable, -1)]

    def test_out_of_order(self):
        iterable = map(str, count())
        it = mi.ichunked(iterable, 4)
        chunk_1 = next(it)
        chunk_2 = next(it)
        self.assertEqual(''.join(chunk_2), '4567')
        self.assertEqual(''.join(chunk_1), '0123')

    def test_laziness(self):
        def gen():
            yield 0
            raise RuntimeError
            yield from count(1)

        it = mi.ichunked(gen(), 4)
        chunk = next(it)
        self.assertEqual(next(chunk), 0)
        self.assertRaises(RuntimeError, next, it)

    def test_memory_in_order(self):
        gen_numbers = []

        def gen():
            for gen_number in count():
                gen_numbers.append(gen_number)
                yield gen_number

        # No items should be kept in memory when a ichunked is first called
        all_chunks = mi.ichunked(gen(), 4)
        self.assertEqual(gen_numbers, [])

        # The first item of each chunk should be generated on chunk generation
        first_chunk = next(all_chunks)
        self.assertEqual(gen_numbers, [0])

        # If we don't read a chunk before getting its successor, its contents
        # will be cached
        second_chunk = next(all_chunks)
        self.assertEqual(gen_numbers, [0, 1, 2, 3, 4])

        # Check if we can read in cached values
        self.assertEqual(list(first_chunk), [0, 1, 2, 3])
        self.assertEqual(list(second_chunk), [4, 5, 6, 7])

        # Again only the most recent chunk should have an item cached
        third_chunk = next(all_chunks)
        self.assertEqual(len(gen_numbers), 9)

        # No new item should be cached when reading past the first number
        next(third_chunk)
        self.assertEqual(len(gen_numbers), 9)

        # we should not be able to read spent chunks
        self.assertEqual(list(first_chunk), [])
        self.assertEqual(list(second_chunk), [])


class DistinctCombinationsTests(TestCase):
    def test_basic(self):
        for iterable in [
            (1, 2, 2, 3, 3, 3),  # In order
            range(6),  # All distinct
            'abbccc',  # Not numbers
            'cccbba',  # Backward
            'mississippi',  # No particular order
        ]:
            for r in range(len(iterable)):
                with self.subTest(iterable=iterable, r=r):
                    actual = list(mi.distinct_combinations(iterable, r))
                    expected = list(
                        mi.unique_everseen(combinations(iterable, r))
                    )
                    self.assertEqual(actual, expected)

    def test_negative(self):
        with self.assertRaises(ValueError):
            list(mi.distinct_combinations([], -1))

    def test_empty(self):
        self.assertEqual(list(mi.distinct_combinations([], 2)), [])


class FilterExceptTests(TestCase):
    def test_no_exceptions_pass(self):
        iterable = '0123'
        actual = list(mi.filter_except(int, iterable))
        expected = ['0', '1', '2', '3']
        self.assertEqual(actual, expected)

    def test_no_exceptions_raise(self):
        iterable = ['0', '1', 'two', '3']
        with self.assertRaises(ValueError):
            list(mi.filter_except(int, iterable))

    def test_raise(self):
        iterable = ['0', '1' '2', 'three', None]
        with self.assertRaises(TypeError):
            list(mi.filter_except(int, iterable, ValueError))

    def test_false(self):
        # Even if the validator returns false, we pass through
        validator = lambda x: False
        iterable = ['0', '1', '2', 'three', None]
        actual = list(mi.filter_except(validator, iterable, Exception))
        expected = ['0', '1', '2', 'three', None]
        self.assertEqual(actual, expected)

    def test_multiple(self):
        iterable = ['0', '1', '2', 'three', None, '4']
        actual = list(mi.filter_except(int, iterable, ValueError, TypeError))
        expected = ['0', '1', '2', '4']
        self.assertEqual(actual, expected)


class MapExceptTests(TestCase):
    def test_no_exceptions_pass(self):
        iterable = '0123'
        actual = list(mi.map_except(int, iterable))
        expected = [0, 1, 2, 3]
        self.assertEqual(actual, expected)

    def test_no_exceptions_raise(self):
        iterable = ['0', '1', 'two', '3']
        with self.assertRaises(ValueError):
            list(mi.map_except(int, iterable))

    def test_raise(self):
        iterable = ['0', '1' '2', 'three', None]
        with self.assertRaises(TypeError):
            list(mi.map_except(int, iterable, ValueError))

    def test_multiple(self):
        iterable = ['0', '1', '2', 'three', None, '4']
        actual = list(mi.map_except(int, iterable, ValueError, TypeError))
        expected = [0, 1, 2, 4]
        self.assertEqual(actual, expected)


class MapIfTests(TestCase):
    def test_without_func_else(self):
        iterable = list(range(-5, 5))
        actual = list(mi.map_if(iterable, lambda x: x > 3, lambda x: 'toobig'))
        expected = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 'toobig']
        self.assertEqual(actual, expected)

    def test_with_func_else(self):
        iterable = list(range(-5, 5))
        actual = list(
            mi.map_if(
                iterable, lambda x: x >= 0, lambda x: 'notneg', lambda x: 'neg'
            )
        )
        expected = ['neg'] * 5 + ['notneg'] * 5
        self.assertEqual(actual, expected)

    def test_empty(self):
        actual = list(mi.map_if([], lambda x: len(x) > 5, lambda x: None))
        expected = []
        self.assertEqual(actual, expected)


class SampleTests(TestCase):
    def test_unit_case(self):
        """Test against a fixed case by seeding the random module."""
        # Beware that this test really just verifies random.random() behavior.
        # If the algorithm is changed (e.g. to a more naive implementation)
        # this test will fail, but the algorithm might be correct.
        # Also, this test can pass and the algorithm can be completely wrong.
        data = "abcdef"
        weights = list(range(1, len(data) + 1))
        seed(123)
        actual = mi.sample(data, k=2, weights=weights)
        expected = ['f', 'e']
        self.assertEqual(actual, expected)

    def test_length(self):
        """Check that *k* elements are sampled."""
        data = [1, 2, 3, 4, 5]
        for k in [0, 3, 5, 7]:
            sampled = mi.sample(data, k=k)
            actual = len(sampled)
            expected = min(k, len(data))
            self.assertEqual(actual, expected)

    def test_sampling_entire_iterable(self):
        """If k=len(iterable), the sample contains the original elements."""
        data = ["a", 2, "a", 4, (1, 2, 3)]
        actual = set(mi.sample(data, k=len(data)))
        expected = set(data)
        self.assertEqual(actual, expected)

    def test_scale_invariance_of_weights(self):
        """The probability of choosing element a_i is w_i / sum(weights).
        Scaling weights should not change the probability or outcome."""
        data = "abcdef"

        weights = list(range(1, len(data) + 1))
        seed(123)
        first_sample = mi.sample(data, k=2, weights=weights)

        # Scale the weights and sample again
        weights_scaled = [w / 1e10 for w in weights]
        seed(123)
        second_sample = mi.sample(data, k=2, weights=weights_scaled)

        self.assertEqual(first_sample, second_sample)

    def test_invariance_under_permutations_unweighted(self):
        """The order of the data should not matter. This is a stochastic test,
        but it will fail in less than 1 / 10_000 cases."""

        # Create a data set and a reversed data set
        data = list(range(100))
        data_rev = list(reversed(data))

        # Sample each data set 10 times
        data_means = [mean(mi.sample(data, k=50)) for _ in range(10)]
        data_rev_means = [mean(mi.sample(data_rev, k=50)) for _ in range(10)]

        # The difference in the means should be low, i.e. little bias
        difference_in_means = abs(mean(data_means) - mean(data_rev_means))

        # The observed largest difference in 10,000 simulations was 5.09599
        self.assertTrue(difference_in_means < 5.1)

    def test_invariance_under_permutations_weighted(self):
        """The order of the data should not matter. This is a stochastic test,
        but it will fail in less than 1 / 10_000 cases."""

        # Create a data set and a reversed data set
        data = list(range(1, 101))
        data_rev = list(reversed(data))

        # Sample each data set 10 times
        data_means = [
            mean(mi.sample(data, k=50, weights=data)) for _ in range(10)
        ]
        data_rev_means = [
            mean(mi.sample(data_rev, k=50, weights=data_rev))
            for _ in range(10)
        ]

        # The difference in the means should be low, i.e. little bias
        difference_in_means = abs(mean(data_means) - mean(data_rev_means))

        # The observed largest difference in 10,000 simulations was 4.337999
        self.assertTrue(difference_in_means < 4.4)


class IsSortedTests(TestCase):
    def test_basic(self):
        for iterable, kwargs, expected in [
            ([], {}, True),
            ([1], {}, True),
            ([1, 2, 3], {}, True),
            ([1, 1, 2, 3], {}, True),
            ([1, 10, 2, 3], {}, False),
            (['1', '10', '2', '3'], {}, True),
            (['1', '10', '2', '3'], {'key': int}, False),
            ([1, 2, 3], {'reverse': True}, False),
            ([1, 1, 2, 3], {'reverse': True}, False),
            ([1, 10, 2, 3], {'reverse': True}, False),
            (['3', '2', '10', '1'], {'reverse': True}, True),
            (['3', '2', '10', '1'], {'key': int, 'reverse': True}, False),
            # strict
            ([], {'strict': True}, True),
            ([1], {'strict': True}, True),
            ([1, 1], {'strict': True}, False),
            ([1, 2, 3], {'strict': True}, True),
            ([1, 1, 2, 3], {'strict': True}, False),
            ([1, 10, 2, 3], {'strict': True}, False),
            (['1', '10', '2', '3'], {'strict': True}, True),
            (['1', '10', '2', '3', '3'], {'strict': True}, False),
            (['1', '10', '2', '3'], {'strict': True, 'key': int}, False),
            ([1, 2, 3], {'strict': True, 'reverse': True}, False),
            ([1, 1, 2, 3], {'strict': True, 'reverse': True}, False),
            ([1, 10, 2, 3], {'strict': True, 'reverse': True}, False),
            (['3', '2', '10', '1'], {'strict': True, 'reverse': True}, True),
            (
                ['3', '2', '10', '10', '1'],
                {'strict': True, 'reverse': True},
                False,
            ),
            (
                ['3', '2', '10', '1'],
                {'strict': True, 'key': int, 'reverse': True},
                False,
            ),
            # We'll do the same weird thing as Python here
            (['nan', 0, 'nan', 0], {'key': float}, True),
            ([0, 'nan', 0, 'nan'], {'key': float}, True),
            (['nan', 0, 'nan', 0], {'key': float, 'reverse': True}, True),
            ([0, 'nan', 0, 'nan'], {'key': float, 'reverse': True}, True),
            ([0, 'nan', 0, 'nan'], {'strict': True, 'key': float}, True),
            (
                ['nan', 0, 'nan', 0],
                {'strict': True, 'key': float, 'reverse': True},
                True,
            ),
        ]:
            key = kwargs.get('key', None)
            reverse = kwargs.get('reverse', False)
            strict = kwargs.get('strict', False)

            with self.subTest(
                iterable=iterable, key=key, reverse=reverse, strict=strict
            ):
                mi_result = mi.is_sorted(
                    iter(iterable), key=key, reverse=reverse, strict=strict
                )

                sorted_iterable = sorted(iterable, key=key, reverse=reverse)
                if strict:
                    sorted_iterable = list(mi.unique_justseen(sorted_iterable))

                py_result = iterable == sorted_iterable

                self.assertEqual(mi_result, expected)
                self.assertEqual(mi_result, py_result)


class CallbackIterTests(TestCase):
    def _target(self, cb=None, exc=None, wait=0):
        total = 0
        for i, c in enumerate('abc', 1):
            total += i
            if wait:
                sleep(wait)
            if cb:
                cb(i, c, intermediate_total=total)
            if exc:
                raise exc('error in target')

        return total

    def test_basic(self):
        func = lambda callback=None: self._target(cb=callback, wait=0.02)
        with mi.callback_iter(func, wait_seconds=0.01) as it:
            # Execution doesn't start until we begin iterating
            self.assertFalse(it.done)

            # Consume everything
            self.assertEqual(
                list(it),
                [
                    ((1, 'a'), {'intermediate_total': 1}),
                    ((2, 'b'), {'intermediate_total': 3}),
                    ((3, 'c'), {'intermediate_total': 6}),
                ],
            )

            # After consuming everything the future is done and the
            # result is available.
            self.assertTrue(it.done)
            self.assertEqual(it.result, 6)

        # This examines the internal state of the ThreadPoolExecutor. This
        # isn't documented, so may break in future Python versions.
        self.assertTrue(it._executor._shutdown)

    def test_callback_kwd(self):
        with mi.callback_iter(self._target, callback_kwd='cb') as it:
            self.assertEqual(
                list(it),
                [
                    ((1, 'a'), {'intermediate_total': 1}),
                    ((2, 'b'), {'intermediate_total': 3}),
                    ((3, 'c'), {'intermediate_total': 6}),
                ],
            )

    def test_partial_consumption(self):
        func = lambda callback=None: self._target(cb=callback)
        with mi.callback_iter(func) as it:
            self.assertEqual(next(it), ((1, 'a'), {'intermediate_total': 1}))

        self.assertTrue(it._executor._shutdown)

    def test_abort(self):
        func = lambda callback=None: self._target(cb=callback, wait=0.1)
        with mi.callback_iter(func) as it:
            self.assertEqual(next(it), ((1, 'a'), {'intermediate_total': 1}))

        with self.assertRaises(mi.AbortThread):
            it.result

    def test_no_result(self):
        func = lambda callback=None: self._target(cb=callback)
        with mi.callback_iter(func) as it:
            with self.assertRaises(RuntimeError):
                it.result

    def test_exception(self):
        func = lambda callback=None: self._target(cb=callback, exc=ValueError)
        with mi.callback_iter(func) as it:
            self.assertEqual(
                next(it),
                ((1, 'a'), {'intermediate_total': 1}),
            )

            with self.assertRaises(ValueError):
                it.result


class WindowedCompleteTests(TestCase):
    """Tests for ``windowed_complete()``"""

    def test_basic(self):
        actual = list(mi.windowed_complete([1, 2, 3, 4, 5], 3))
        expected = [
            ((), (1, 2, 3), (4, 5)),
            ((1,), (2, 3, 4), (5,)),
            ((1, 2), (3, 4, 5), ()),
        ]
        self.assertEqual(actual, expected)

    def test_zero_length(self):
        actual = list(mi.windowed_complete([1, 2, 3], 0))
        expected = [
            ((), (), (1, 2, 3)),
            ((1,), (), (2, 3)),
            ((1, 2), (), (3,)),
            ((1, 2, 3), (), ()),
        ]
        self.assertEqual(actual, expected)

    def test_wrong_length(self):
        seq = [1, 2, 3, 4, 5]
        for n in (-10, -1, len(seq) + 1, len(seq) + 10):
            with self.subTest(n=n):
                with self.assertRaises(ValueError):
                    list(mi.windowed_complete(seq, n))

    def test_every_partition(self):
        every_partition = lambda seq: chain(
            *map(partial(mi.windowed_complete, seq), range(len(seq)))
        )

        seq = 'ABC'
        actual = list(every_partition(seq))
        expected = [
            ((), (), ('A', 'B', 'C')),
            (('A',), (), ('B', 'C')),
            (('A', 'B'), (), ('C',)),
            (('A', 'B', 'C'), (), ()),
            ((), ('A',), ('B', 'C')),
            (('A',), ('B',), ('C',)),
            (('A', 'B'), ('C',), ()),
            ((), ('A', 'B'), ('C',)),
            (('A',), ('B', 'C'), ()),
        ]
        self.assertEqual(actual, expected)


class AllUniqueTests(TestCase):
    def test_basic(self):
        for iterable, expected in [
            ([], True),
            ([1, 2, 3], True),
            ([1, 1], False),
            ([1, 2, 3, 1], False),
            ([1, 2, 3, '1'], True),
        ]:
            with self.subTest(args=(iterable,)):
                self.assertEqual(mi.all_unique(iterable), expected)

    def test_non_hashable(self):
        self.assertEqual(mi.all_unique([[1, 2], [3, 4]]), True)
        self.assertEqual(mi.all_unique([[1, 2], [3, 4], [1, 2]]), False)

    def test_partially_hashable(self):
        self.assertEqual(mi.all_unique([[1, 2], [3, 4], (5, 6)]), True)
        self.assertEqual(
            mi.all_unique([[1, 2], [3, 4], (5, 6), [1, 2]]), False
        )
        self.assertEqual(
            mi.all_unique([[1, 2], [3, 4], (5, 6), (5, 6)]), False
        )

    def test_key(self):
        iterable = ['A', 'B', 'C', 'b']
        self.assertEqual(mi.all_unique(iterable, lambda x: x), True)
        self.assertEqual(mi.all_unique(iterable, str.lower), False)

    def test_infinite(self):
        self.assertEqual(mi.all_unique(mi.prepend(3, count())), False)


class NthProductTests(TestCase):
    def test_basic(self):
        iterables = ['ab', 'cdef', 'ghi']
        for index, expected in enumerate(product(*iterables)):
            actual = mi.nth_product(index, *iterables)
            self.assertEqual(actual, expected)

    def test_long(self):
        actual = mi.nth_product(1337, range(101), range(22), range(53))
        expected = (1, 3, 12)
        self.assertEqual(actual, expected)

    def test_negative(self):
        iterables = ['abc', 'de', 'fghi']
        for index, expected in enumerate(product(*iterables)):
            actual = mi.nth_product(index - 24, *iterables)
            self.assertEqual(actual, expected)

    def test_invalid_index(self):
        with self.assertRaises(IndexError):
            mi.nth_product(24, 'ab', 'cde', 'fghi')


class NthCombinationWithReplacementTests(TestCase):
    def test_basic(self):
        iterable = 'abcdefg'
        r = 4
        for index, expected in enumerate(
            combinations_with_replacement(iterable, r)
        ):
            actual = mi.nth_combination_with_replacement(iterable, r, index)
            self.assertEqual(actual, expected)

    def test_long(self):
        actual = mi.nth_combination_with_replacement(range(90), 4, 2000000)
        expected = (22, 65, 68, 81)
        self.assertEqual(actual, expected)

    def test_invalid_r(self):
        for r in (-1, 3):
            with self.assertRaises(ValueError):
                mi.nth_combination_with_replacement([], r, 0)

    def test_invalid_index(self):
        with self.assertRaises(IndexError):
            mi.nth_combination_with_replacement('abcdefg', 3, -85)


class ValueChainTests(TestCase):
    def test_empty(self):
        actual = list(mi.value_chain())
        expected = []
        self.assertEqual(actual, expected)

    def test_simple(self):
        actual = list(mi.value_chain(1, 2.71828, False, 'foo'))
        expected = [1, 2.71828, False, 'foo']
        self.assertEqual(actual, expected)

    def test_more(self):
        actual = list(mi.value_chain(b'bar', [1, 2, 3], 4, {'key': 1}))
        expected = [b'bar', 1, 2, 3, 4, 'key']
        self.assertEqual(actual, expected)

    def test_empty_lists(self):
        actual = list(mi.value_chain(1, 2, [], [3, 4]))
        expected = [1, 2, 3, 4]
        self.assertEqual(actual, expected)

    def test_complex(self):
        obj = object()
        actual = list(
            mi.value_chain(
                (1, (2, (3,))),
                ['foo', ['bar', ['baz']], 'tic'],
                {'key': {'foo': 1}},
                obj,
            )
        )
        expected = [1, (2, (3,)), 'foo', ['bar', ['baz']], 'tic', 'key', obj]
        self.assertEqual(actual, expected)


class ProductIndexTests(TestCase):
    def test_basic(self):
        iterables = ['ab', 'cdef', 'ghi']
        first_index = {}
        for index, element in enumerate(product(*iterables)):
            actual = mi.product_index(element, *iterables)
            expected = first_index.setdefault(element, index)
            self.assertEqual(actual, expected)

    def test_multiplicity(self):
        iterables = ['ab', 'bab', 'cab']
        first_index = {}
        for index, element in enumerate(product(*iterables)):
            actual = mi.product_index(element, *iterables)
            expected = first_index.setdefault(element, index)
            self.assertEqual(actual, expected)

    def test_long(self):
        actual = mi.product_index((1, 3, 12), range(101), range(22), range(53))
        expected = 1337
        self.assertEqual(actual, expected)

    def test_invalid_empty(self):
        with self.assertRaises(ValueError):
            mi.product_index('', 'ab', 'cde', 'fghi')

    def test_invalid_small(self):
        with self.assertRaises(ValueError):
            mi.product_index('ac', 'ab', 'cde', 'fghi')

    def test_invalid_large(self):
        with self.assertRaises(ValueError):
            mi.product_index('achi', 'ab', 'cde', 'fghi')

    def test_invalid_match(self):
        with self.assertRaises(ValueError):
            mi.product_index('axf', 'ab', 'cde', 'fghi')


class CombinationIndexTests(TestCase):
    def test_r_less_than_n(self):
        iterable = 'abcdefg'
        r = 4
        first_index = {}
        for index, element in enumerate(combinations(iterable, r)):
            actual = mi.combination_index(element, iterable)
            expected = first_index.setdefault(element, index)
            self.assertEqual(actual, expected)

    def test_r_equal_to_n(self):
        iterable = 'abcd'
        r = len(iterable)
        first_index = {}
        for index, element in enumerate(combinations(iterable, r=r)):
            actual = mi.combination_index(element, iterable)
            expected = first_index.setdefault(element, index)
            self.assertEqual(actual, expected)

    def test_multiplicity(self):
        iterable = 'abacba'
        r = 3
        first_index = {}
        for index, element in enumerate(combinations(iterable, r)):
            actual = mi.combination_index(element, iterable)
            expected = first_index.setdefault(element, index)
            self.assertEqual(actual, expected)

    def test_null(self):
        actual = mi.combination_index(tuple(), [])
        expected = 0
        self.assertEqual(actual, expected)

    def test_long(self):
        actual = mi.combination_index((2, 12, 35, 126), range(180))
        expected = 2000000
        self.assertEqual(actual, expected)

    def test_invalid_order(self):
        with self.assertRaises(ValueError):
            mi.combination_index(tuple('acb'), 'abcde')

    def test_invalid_large(self):
        with self.assertRaises(ValueError):
            mi.combination_index(tuple('abcdefg'), 'abcdef')

    def test_invalid_match(self):
        with self.assertRaises(ValueError):
            mi.combination_index(tuple('axe'), 'abcde')


class CombinationWithReplacementIndexTests(TestCase):
    def test_r_less_than_n(self):
        iterable = 'abcdefg'
        r = 4
        first_index = {}
        for index, element in enumerate(
            combinations_with_replacement(iterable, r)
        ):
            actual = mi.combination_with_replacement_index(element, iterable)
            expected = first_index.setdefault(element, index)
            self.assertEqual(actual, expected)

    def test_r_equal_to_n(self):
        iterable = 'abcd'
        r = len(iterable)
        first_index = {}
        for index, element in enumerate(
            combinations_with_replacement(iterable, r=r)
        ):
            actual = mi.combination_with_replacement_index(element, iterable)
            expected = first_index.setdefault(element, index)
            self.assertEqual(actual, expected)

    def test_multiplicity(self):
        iterable = 'abacba'
        r = 3
        first_index = {}
        for index, element in enumerate(
            combinations_with_replacement(iterable, r)
        ):
            actual = mi.combination_with_replacement_index(element, iterable)
            expected = first_index.setdefault(element, index)
            self.assertEqual(actual, expected)

    def test_null(self):
        actual = mi.combination_with_replacement_index(tuple(), [])
        expected = 0
        self.assertEqual(actual, expected)

    def test_long(self):
        actual = mi.combination_with_replacement_index(
            (22, 65, 68, 81), range(90)
        )
        expected = 2000000
        self.assertEqual(actual, expected)

    def test_invalid_order(self):
        with self.assertRaises(ValueError):
            mi.combination_with_replacement_index(tuple('acb'), 'abcde')

    def test_invalid_large(self):
        with self.assertRaises(ValueError):
            mi.combination_with_replacement_index(tuple('abcdefg'), 'abcdef')

    def test_invalid_match(self):
        with self.assertRaises(ValueError):
            mi.combination_with_replacement_index(tuple('axe'), 'abcde')


class PermutationIndexTests(TestCase):
    def test_r_less_than_n(self):
        iterable = 'abcdefg'
        r = 4
        first_index = {}
        for index, element in enumerate(permutations(iterable, r)):
            actual = mi.permutation_index(element, iterable)
            expected = first_index.setdefault(element, index)
            self.assertEqual(actual, expected)

    def test_r_equal_to_n(self):
        iterable = 'abcd'
        first_index = {}
        for index, element in enumerate(permutations(iterable)):
            actual = mi.permutation_index(element, iterable)
            expected = first_index.setdefault(element, index)
            self.assertEqual(actual, expected)

    def test_multiplicity(self):
        iterable = 'abacba'
        r = 3
        first_index = {}
        for index, element in enumerate(permutations(iterable, r)):
            actual = mi.permutation_index(element, iterable)
            expected = first_index.setdefault(element, index)
            self.assertEqual(actual, expected)

    def test_null(self):
        actual = mi.permutation_index(tuple(), [])
        expected = 0
        self.assertEqual(actual, expected)

    def test_long(self):
        actual = mi.permutation_index((2, 12, 35, 126), range(180))
        expected = 11631678
        self.assertEqual(actual, expected)

    def test_invalid_large(self):
        with self.assertRaises(ValueError):
            mi.permutation_index(tuple('abcdefg'), 'abcdef')

    def test_invalid_match(self):
        with self.assertRaises(ValueError):
            mi.permutation_index(tuple('axe'), 'abcde')


class CountableTests(TestCase):
    def test_empty(self):
        iterable = []
        it = mi.countable(iterable)
        self.assertEqual(it.items_seen, 0)
        self.assertEqual(list(it), [])

    def test_basic(self):
        iterable = '0123456789'
        it = mi.countable(iterable)
        self.assertEqual(it.items_seen, 0)
        self.assertEqual(next(it), '0')
        self.assertEqual(it.items_seen, 1)
        self.assertEqual(''.join(it), '123456789')
        self.assertEqual(it.items_seen, 10)


class ChunkedEvenTests(TestCase):
    """Tests for ``chunked_even()``"""

    def test_0(self):
        self._test_finite('', 3, [])

    def test_1(self):
        self._test_finite('A', 1, [['A']])

    def test_4(self):
        self._test_finite('ABCD', 3, [['A', 'B'], ['C', 'D']])

    def test_5(self):
        self._test_finite('ABCDE', 3, [['A', 'B', 'C'], ['D', 'E']])

    def test_6(self):
        self._test_finite('ABCDEF', 3, [['A', 'B', 'C'], ['D', 'E', 'F']])

    def test_7(self):
        self._test_finite(
            'ABCDEFG', 3, [['A', 'B', 'C'], ['D', 'E'], ['F', 'G']]
        )

    def _test_finite(self, seq, n, expected):
        # Check with and without `len()`
        self.assertEqual(list(mi.chunked_even(seq, n)), expected)
        self.assertEqual(list(mi.chunked_even(iter(seq), n)), expected)

    def test_infinite(self):
        for n in range(1, 5):
            k = 0

            def count_with_assert():
                for i in count():
                    # Look-ahead should be less than n^2
                    self.assertLessEqual(i, n * k + n * n)
                    yield i

            ls = mi.chunked_even(count_with_assert(), n)
            while k < 2:
                self.assertEqual(next(ls), list(range(k * n, (k + 1) * n)))
                k += 1

    def test_evenness(self):
        for N in range(1, 50):
            for n in range(1, N + 2):
                lengths = []
                items = []
                for l in mi.chunked_even(range(N), n):
                    L = len(l)
                    self.assertLessEqual(L, n)
                    self.assertGreaterEqual(L, 1)
                    lengths.append(L)
                    items.extend(l)
                self.assertEqual(items, list(range(N)))
                self.assertLessEqual(max(lengths) - min(lengths), 1)


class ZipBroadcastTests(TestCase):
    def test_zip(self):
        for objects, zipped, strict_ok in [
            # Empty
            ([], [], True),
            # One argument
            ([1], [(1,)], True),
            ([[1]], [(1,)], True),
            ([[1, 2]], [(1,), (2,)], True),
            # All scalars
            ([1, 2], [(1, 2)], True),
            ([1, 2, 3], [(1, 2, 3)], True),
            # Iterables with length = 0
            ([[], 1], [], True),
            ([1, []], [], True),
            ([[], []], [], True),
            ([[], 1, 2], [], True),
            ([[], 1, []], [], True),
            ([1, [], 2], [], True),
            ([1, [], []], [], True),
            ([[], [], 1], [], True),
            ([[], [], []], [], True),
            # Iterables with length = 1
            ([1, [2]], [(1, 2)], True),
            ([[1], 2], [(1, 2)], True),
            ([[1], [2]], [(1, 2)], True),
            ([1, [2], 3], [(1, 2, 3)], True),
            ([1, [2], [3]], [(1, 2, 3)], True),
            ([[1], 2, 3], [(1, 2, 3)], True),
            ([[1], 2, [3]], [(1, 2, 3)], True),
            ([[1], [2], 3], [(1, 2, 3)], True),
            ([[1], [2], [3]], [(1, 2, 3)], True),
            # Iterables with length > 1
            ([1, [2, 3]], [(1, 2), (1, 3)], True),
            ([[1, 2], 3], [(1, 3), (2, 3)], True),
            ([[1, 2], [3, 4]], [(1, 3), (2, 4)], True),
            ([1, [2, 3], 4], [(1, 2, 4), (1, 3, 4)], True),
            ([1, [2, 3], [4, 5]], [(1, 2, 4), (1, 3, 5)], True),
            ([[1, 2], 3, 4], [(1, 3, 4), (2, 3, 4)], True),
            ([[1, 2], 3, [4, 5]], [(1, 3, 4), (2, 3, 5)], True),
            ([[1, 2], [3, 4], 5], [(1, 3, 5), (2, 4, 5)], True),
            ([[1, 2], [3, 4], [5, 6]], [(1, 3, 5), (2, 4, 6)], True),
            # Iterables with different lengths
            ([[], [1]], [], False),
            ([[1], []], [], False),
            ([[1], [2, 3]], [(1, 2)], False),
            ([[1, 2], [3]], [(1, 3)], False),
            ([[1, 2], [3], [4]], [(1, 3, 4)], False),
            ([[1], [2, 3], [4]], [(1, 2, 4)], False),
            ([[1], [2], [3, 4]], [(1, 2, 3)], False),
            ([[1], [2, 3], [4, 5]], [(1, 2, 4)], False),
            ([[1, 2], [3], [4, 5]], [(1, 3, 4)], False),
            ([[1, 2], [3, 4], [5]], [(1, 3, 5)], False),
            ([1, [2, 3], [4, 5, 6]], [(1, 2, 4), (1, 3, 5)], False),
            ([[1, 2], 3, [4, 5, 6]], [(1, 3, 4), (2, 3, 5)], False),
            ([1, [2, 3, 4], [5, 6]], [(1, 2, 5), (1, 3, 6)], False),
            ([[1, 2, 3], 4, [5, 6]], [(1, 4, 5), (2, 4, 6)], False),
            ([[1, 2], [3, 4, 5], 6], [(1, 3, 6), (2, 4, 6)], False),
            ([[1, 2, 3], [4, 5], 6], [(1, 4, 6), (2, 5, 6)], False),
            # Infinite
            ([count(), 1, [2]], [(0, 1, 2)], False),
            ([count(), 1, [2, 3]], [(0, 1, 2), (1, 1, 3)], False),
            # Miscellaneous
            (['a', [1, 2], [3, 4, 5]], [('a', 1, 3), ('a', 2, 4)], False),
        ]:
            # Truncate by default
            with self.subTest(objects=objects, strict=False, zipped=zipped):
                self.assertEqual(list(mi.zip_broadcast(*objects)), zipped)

            # Raise an exception for strict=True
            with self.subTest(objects=objects, strict=True, zipped=zipped):
                if strict_ok:
                    self.assertEqual(
                        list(mi.zip_broadcast(*objects, strict=True)),
                        zipped,
                    )
                else:
                    with self.assertRaises(ValueError):
                        list(mi.zip_broadcast(*objects, strict=True))

    def test_scalar_types(self):
        # Default: str and bytes are treated as scalar
        self.assertEqual(
            list(mi.zip_broadcast('ab', [1, 2, 3])),
            [('ab', 1), ('ab', 2), ('ab', 3)],
        )
        self.assertEqual(
            list(mi.zip_broadcast(b'ab', [1, 2, 3])),
            [(b'ab', 1), (b'ab', 2), (b'ab', 3)],
        )
        # scalar_types=None allows str and bytes to be treated as iterable
        self.assertEqual(
            list(mi.zip_broadcast('abc', [1, 2, 3], scalar_types=None)),
            [('a', 1), ('b', 2), ('c', 3)],
        )
        # Use a custom type
        self.assertEqual(
            list(mi.zip_broadcast({'a': 'b'}, [1, 2, 3], scalar_types=dict)),
            [({'a': 'b'}, 1), ({'a': 'b'}, 2), ({'a': 'b'}, 3)],
        )


class UniqueInWindowTests(TestCase):
    def test_invalid_n(self):
        with self.assertRaises(ValueError):
            list(mi.unique_in_window([], 0))

    def test_basic(self):
        for iterable, n, expected in [
            (range(9), 10, list(range(9))),
            (range(20), 10, list(range(20))),
            ([1, 2, 3, 4, 4, 4], 1, [1, 2, 3, 4, 4, 4]),
            ([1, 2, 3, 4, 4, 4], 2, [1, 2, 3, 4]),
            ([1, 2, 3, 4, 4, 4], 3, [1, 2, 3, 4]),
            ([1, 2, 3, 4, 4, 4], 4, [1, 2, 3, 4]),
            ([1, 2, 3, 4, 4, 4], 5, [1, 2, 3, 4]),
            (
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 2, 3, 4, 2],
                2,
                [0, 1, 0, 2, 3, 4, 2],
            ),
        ]:
            with self.subTest(expected=expected):
                actual = list(mi.unique_in_window(iterable, n))
                self.assertEqual(actual, expected)

    def test_key(self):
        iterable = [0, 1, 3, 4, 5, 6, 7, 8, 9]
        n = 3
        key = lambda x: x // 3
        actual = list(mi.unique_in_window(iterable, n, key=key))
        expected = [0, 3, 6, 9]
        self.assertEqual(actual, expected)


class StrictlyNTests(TestCase):
    def test_basic(self):
        iterable = ['a', 'b', 'c', 'd']
        n = 4
        actual = list(mi.strictly_n(iter(iterable), n))
        expected = iterable
        self.assertEqual(actual, expected)

    def test_too_short_default(self):
        iterable = ['a', 'b', 'c', 'd']
        n = 5
        with self.assertRaises(ValueError) as exc:
            list(mi.strictly_n(iter(iterable), n))

        self.assertEqual(
            'Too few items in iterable (got 4)', exc.exception.args[0]
        )

    def test_too_long_default(self):
        iterable = ['a', 'b', 'c', 'd']
        n = 3
        with self.assertRaises(ValueError) as cm:
            list(mi.strictly_n(iter(iterable), n))

        self.assertEqual(
            'Too many items in iterable (got at least 4)',
            cm.exception.args[0],
        )

    def test_too_short_custom(self):
        call_count = 0

        def too_short(item_count):
            nonlocal call_count
            call_count += 1

        iterable = ['a', 'b', 'c', 'd']
        n = 6
        actual = []
        for item in mi.strictly_n(iter(iterable), n, too_short=too_short):
            actual.append(item)
        expected = ['a', 'b', 'c', 'd']
        self.assertEqual(actual, expected)
        self.assertEqual(call_count, 1)

    def test_too_long_custom(self):
        import logging

        iterable = ['a', 'b', 'c', 'd']
        n = 2
        too_long = lambda item_count: logging.warning(
            'Picked the first %s items', n
        )

        with self.assertLogs(level='WARNING') as cm:
            actual = list(mi.strictly_n(iter(iterable), n, too_long=too_long))

        self.assertEqual(actual, ['a', 'b'])
        self.assertIn('Picked the first 2 items', cm.output[0])


class DuplicatesEverSeenTests(TestCase):
    def test_basic(self):
        for iterable, expected in [
            ([], []),
            ([1, 2, 3], []),
            ([1, 1], [1]),
            ([1, 2, 1, 2], [1, 2]),
            ([1, 2, 3, '1'], []),
        ]:
            with self.subTest(args=(iterable,)):
                self.assertEqual(
                    list(mi.duplicates_everseen(iterable)), expected
                )

    def test_non_hashable(self):
        self.assertEqual(list(mi.duplicates_everseen([[1, 2], [3, 4]])), [])
        self.assertEqual(
            list(mi.duplicates_everseen([[1, 2], [3, 4], [1, 2]])), [[1, 2]]
        )

    def test_partially_hashable(self):
        self.assertEqual(
            list(mi.duplicates_everseen([[1, 2], [3, 4], (5, 6)])), []
        )
        self.assertEqual(
            list(mi.duplicates_everseen([[1, 2], [3, 4], (5, 6), [1, 2]])),
            [[1, 2]],
        )
        self.assertEqual(
            list(mi.duplicates_everseen([[1, 2], [3, 4], (5, 6), (5, 6)])),
            [(5, 6)],
        )

    def test_key_hashable(self):
        iterable = 'HEheHEhe'
        self.assertEqual(list(mi.duplicates_everseen(iterable)), list('HEhe'))
        self.assertEqual(
            list(mi.duplicates_everseen(iterable, str.lower)),
            list('heHEhe'),
        )

    def test_key_non_hashable(self):
        iterable = [[1, 2], [3, 0], [5, -2], [5, 6]]
        self.assertEqual(
            list(mi.duplicates_everseen(iterable, lambda x: x)), []
        )
        self.assertEqual(
            list(mi.duplicates_everseen(iterable, sum)), [[3, 0], [5, -2]]
        )

    def test_key_partially_hashable(self):
        iterable = [[1, 2], (1, 2), [1, 2], [5, 6]]
        self.assertEqual(
            list(mi.duplicates_everseen(iterable, lambda x: x)), [[1, 2]]
        )
        self.assertEqual(
            list(mi.duplicates_everseen(iterable, list)), [(1, 2), [1, 2]]
        )


class DuplicatesJustSeenTests(TestCase):
    def test_basic(self):
        for iterable, expected in [
            ([], []),
            ([1, 2, 3, 3, 2, 2], [3, 2]),
            ([1, 1], [1]),
            ([1, 2, 1, 2], []),
            ([1, 2, 3, '1'], []),
        ]:
            with self.subTest(args=(iterable,)):
                self.assertEqual(
                    list(mi.duplicates_justseen(iterable)), expected
                )

    def test_non_hashable(self):
        self.assertEqual(list(mi.duplicates_justseen([[1, 2], [3, 4]])), [])
        self.assertEqual(
            list(
                mi.duplicates_justseen(
                    [[1, 2], [3, 4], [3, 4], [3, 4], [1, 2]]
                )
            ),
            [[3, 4], [3, 4]],
        )

    def test_partially_hashable(self):
        self.assertEqual(
            list(mi.duplicates_justseen([[1, 2], [3, 4], (5, 6)])), []
        )
        self.assertEqual(
            list(
                mi.duplicates_justseen(
                    [[1, 2], [3, 4], (5, 6), [1, 2], [1, 2]]
                )
            ),
            [[1, 2]],
        )
        self.assertEqual(
            list(
                mi.duplicates_justseen(
                    [[1, 2], [3, 4], (5, 6), (5, 6), (5, 6)]
                )
            ),
            [(5, 6), (5, 6)],
        )

    def test_key_hashable(self):
        iterable = 'HEheHHHhEheeEe'
        self.assertEqual(list(mi.duplicates_justseen(iterable)), list('HHe'))
        self.assertEqual(
            list(mi.duplicates_justseen(iterable, str.lower)),
            list('HHheEe'),
        )

    def test_key_non_hashable(self):
        iterable = [[1, 2], [3, 0], [5, -2], [5, 6], [1, 2]]
        self.assertEqual(
            list(mi.duplicates_justseen(iterable, lambda x: x)), []
        )
        self.assertEqual(
            list(mi.duplicates_justseen(iterable, sum)), [[3, 0], [5, -2]]
        )

    def test_key_partially_hashable(self):
        iterable = [[1, 2], (1, 2), [1, 2], [5, 6], [1, 2]]
        self.assertEqual(
            list(mi.duplicates_justseen(iterable, lambda x: x)), []
        )
        self.assertEqual(
            list(mi.duplicates_justseen(iterable, list)), [(1, 2), [1, 2]]
        )

    def test_nested(self):
        iterable = [[[1, 2], [1, 2]], [5, 6], [5, 6]]
        self.assertEqual(list(mi.duplicates_justseen(iterable)), [[5, 6]])


class ClassifyUniqueTests(TestCase):
    def test_basic(self):
        self.assertEqual(
            list(mi.classify_unique('mississippi')),
            [
                ('m', True, True),
                ('i', True, True),
                ('s', True, True),
                ('s', False, False),
                ('i', True, False),
                ('s', True, False),
                ('s', False, False),
                ('i', True, False),
                ('p', True, True),
                ('p', False, False),
                ('i', True, False),
            ],
        )

    def test_non_hashable(self):
        self.assertEqual(
            list(mi.classify_unique([[1, 2], [3, 4], [3, 4], [1, 2]])),
            [
                ([1, 2], True, True),
                ([3, 4], True, True),
                ([3, 4], False, False),
                ([1, 2], True, False),
            ],
        )

    def test_partially_hashable(self):
        self.assertEqual(
            list(
                mi.classify_unique(
                    [[1, 2], [3, 4], (5, 6), (5, 6), (3, 4), [1, 2]]
                )
            ),
            [
                ([1, 2], True, True),
                ([3, 4], True, True),
                ((5, 6), True, True),
                ((5, 6), False, False),
                ((3, 4), True, True),
                ([1, 2], True, False),
            ],
        )

    def test_key_hashable(self):
        iterable = 'HEheHHHhEheeEe'
        self.assertEqual(
            list(mi.classify_unique(iterable)),
            [
                ('H', True, True),
                ('E', True, True),
                ('h', True, True),
                ('e', True, True),
                ('H', True, False),
                ('H', False, False),
                ('H', False, False),
                ('h', True, False),
                ('E', True, False),
                ('h', True, False),
                ('e', True, False),
                ('e', False, False),
                ('E', True, False),
                ('e', True, False),
            ],
        )
        self.assertEqual(
            list(mi.classify_unique(iterable, str.lower)),
            [
                ('H', True, True),
                ('E', True, True),
                ('h', True, False),
                ('e', True, False),
                ('H', True, False),
                ('H', False, False),
                ('H', False, False),
                ('h', False, False),
                ('E', True, False),
                ('h', True, False),
                ('e', True, False),
                ('e', False, False),
                ('E', False, False),
                ('e', False, False),
            ],
        )

    def test_key_non_hashable(self):
        iterable = [[1, 2], [3, 0], [5, -2], [5, 6], [1, 2]]
        self.assertEqual(
            list(mi.classify_unique(iterable, lambda x: x)),
            [
                ([1, 2], True, True),
                ([3, 0], True, True),
                ([5, -2], True, True),
                ([5, 6], True, True),
                ([1, 2], True, False),
            ],
        )
        self.assertEqual(
            list(mi.classify_unique(iterable, sum)),
            [
                ([1, 2], True, True),
                ([3, 0], False, False),
                ([5, -2], False, False),
                ([5, 6], True, True),
                ([1, 2], True, False),
            ],
        )

    def test_key_partially_hashable(self):
        iterable = [[1, 2], (1, 2), [1, 2], [5, 6], [1, 2]]
        self.assertEqual(
            list(mi.classify_unique(iterable, lambda x: x)),
            [
                ([1, 2], True, True),
                ((1, 2), True, True),
                ([1, 2], True, False),
                ([5, 6], True, True),
                ([1, 2], True, False),
            ],
        )
        self.assertEqual(
            list(mi.classify_unique(iterable, list)),
            [
                ([1, 2], True, True),
                ((1, 2), False, False),
                ([1, 2], False, False),
                ([5, 6], True, True),
                ([1, 2], True, False),
            ],
        )

    def test_vs_unique_everseen(self):
        input = 'AAAABBBBCCDAABBB'
        output = [e for e, j, u in mi.classify_unique(input) if u]
        self.assertEqual(output, ['A', 'B', 'C', 'D'])
        self.assertEqual(list(mi.unique_everseen(input)), output)

    def test_vs_unique_everseen_key(self):
        input = 'aAbACCc'
        output = [e for e, j, u in mi.classify_unique(input, str.lower) if u]
        self.assertEqual(output, list('abC'))
        self.assertEqual(list(mi.unique_everseen(input, str.lower)), output)

    def test_vs_unique_justseen(self):
        input = 'AAAABBBCCDABB'
        output = [e for e, j, u in mi.classify_unique(input) if j]
        self.assertEqual(output, list('ABCDAB'))
        self.assertEqual(list(mi.unique_justseen(input)), output)

    def test_vs_unique_justseen_key(self):
        input = 'AABCcAD'
        output = [e for e, j, u in mi.classify_unique(input, str.lower) if j]
        self.assertEqual(output, list('ABCAD'))
        self.assertEqual(list(mi.unique_justseen(input, str.lower)), output)

    def test_vs_duplicates_everseen(self):
        input = [1, 2, 1, 2]
        output = [e for e, j, u in mi.classify_unique(input) if not u]
        self.assertEqual(output, [1, 2])
        self.assertEqual(list(mi.duplicates_everseen(input)), output)

    def test_vs_duplicates_everseen_key(self):
        input = 'HEheHEhe'
        output = [
            e for e, j, u in mi.classify_unique(input, str.lower) if not u
        ]
        self.assertEqual(output, list('heHEhe'))
        self.assertEqual(
            list(mi.duplicates_everseen(input, str.lower)), output
        )

    def test_vs_duplicates_justseen(self):
        input = [1, 2, 3, 3, 2, 2]
        output = [e for e, j, u in mi.classify_unique(input) if not j]
        self.assertEqual(output, [3, 2])
        self.assertEqual(list(mi.duplicates_justseen(input)), output)

    def test_vs_duplicates_justseen_key(self):
        input = 'HEheHHHhEheeEe'
        output = [
            e for e, j, u in mi.classify_unique(input, str.lower) if not j
        ]
        self.assertEqual(output, list('HHheEe'))
        self.assertEqual(
            list(mi.duplicates_justseen(input, str.lower)), output
        )


class LongestCommonPrefixTests(TestCase):
    def test_basic(self):
        iterables = [[1, 2], [1, 2, 3], [1, 2, 4]]
        self.assertEqual(list(mi.longest_common_prefix(iterables)), [1, 2])

    def test_iterators(self):
        iterables = iter([iter([1, 2]), iter([1, 2, 3]), iter([1, 2, 4])])
        self.assertEqual(list(mi.longest_common_prefix(iterables)), [1, 2])

    def test_no_iterables(self):
        iterables = []
        self.assertEqual(list(mi.longest_common_prefix(iterables)), [])

    def test_empty_iterables_only(self):
        iterables = [[], [], []]
        self.assertEqual(list(mi.longest_common_prefix(iterables)), [])

    def test_includes_empty_iterables(self):
        iterables = [[1, 2], [1, 2, 3], [1, 2, 4], []]
        self.assertEqual(list(mi.longest_common_prefix(iterables)), [])

    def test_non_hashable(self):
        # See https://github.com/more-itertools/more-itertools/issues/603
        iterables = [[[1], [2]], [[1], [2], [3]], [[1], [2], [4]]]
        self.assertEqual(list(mi.longest_common_prefix(iterables)), [[1], [2]])

    def test_prefix_contains_elements_of_the_first_iterable(self):
        iterables = [[[1], [2]], [[1], [2], [3]], [[1], [2], [4]]]
        prefix = list(mi.longest_common_prefix(iterables))
        self.assertIs(prefix[0], iterables[0][0])
        self.assertIs(prefix[1], iterables[0][1])
        self.assertIsNot(prefix[0], iterables[1][0])
        self.assertIsNot(prefix[1], iterables[1][1])
        self.assertIsNot(prefix[0], iterables[2][0])
        self.assertIsNot(prefix[1], iterables[2][1])

    def test_infinite_iterables(self):
        prefix = mi.longest_common_prefix([count(), count()])
        self.assertEqual(next(prefix), 0)
        self.assertEqual(next(prefix), 1)
        self.assertEqual(next(prefix), 2)

    def test_contains_infinite_iterables(self):
        iterables = [[0, 1, 2], count()]
        self.assertEqual(list(mi.longest_common_prefix(iterables)), [0, 1, 2])


class IequalsTests(TestCase):
    def test_basic(self):
        self.assertTrue(mi.iequals("abc", iter("abc")))
        self.assertTrue(mi.iequals(range(3), [0, 1, 2]))
        self.assertFalse(mi.iequals("abc", [0, 1, 2]))

    def test_no_iterables(self):
        self.assertTrue(mi.iequals())

    def test_one_iterable(self):
        self.assertTrue(mi.iequals("abc"))

    def test_more_than_two_iterable(self):
        self.assertTrue(mi.iequals("abc", iter("abc"), ['a', 'b', 'c']))
        self.assertFalse(mi.iequals("abc", iter("abc"), ['a', 'b', 'd']))

    def test_order_matters(self):
        self.assertFalse(mi.iequals("abc", "acb"))

    def test_not_equal_lengths(self):
        self.assertFalse(mi.iequals("abc", "ab"))
        self.assertFalse(mi.iequals("abc", "bc"))
        self.assertFalse(mi.iequals("aaa", "aaaa"))

    def test_empty_iterables(self):
        self.assertTrue(mi.iequals([], ""))

    def test_none_is_not_a_sentinel(self):
        # See https://stackoverflow.com/a/900444
        self.assertFalse(mi.iequals([1, 2], [1, 2, None]))
        self.assertFalse(mi.iequals([1, 2], [None, 1, 2]))

    def test_not_identical_but_equal(self):
        self.assertTrue([1, True], [1.0, complex(1, 0)])


class ConstrainedBatchesTests(TestCase):
    def test_basic(self):
        zen = [
            'Beautiful is better than ugly',
            'Explicit is better than implicit',
            'Simple is better than complex',
            'Complex is better than complicated',
            'Flat is better than nested',
            'Sparse is better than dense',
            'Readability counts',
        ]
        for size, expected in (
            (
                34,
                [
                    (zen[0],),
                    (zen[1],),
                    (zen[2],),
                    (zen[3],),
                    (zen[4],),
                    (zen[5],),
                    (zen[6],),
                ],
            ),
            (
                61,
                [
                    (zen[0], zen[1]),
                    (zen[2],),
                    (zen[3], zen[4]),
                    (zen[5], zen[6]),
                ],
            ),
            (
                90,
                [
                    (zen[0], zen[1], zen[2]),
                    (zen[3], zen[4], zen[5]),
                    (zen[6],),
                ],
            ),
            (
                124,
                [(zen[0], zen[1], zen[2], zen[3]), (zen[4], zen[5], zen[6])],
            ),
            (
                150,
                [(zen[0], zen[1], zen[2], zen[3], zen[4]), (zen[5], zen[6])],
            ),
            (
                177,
                [(zen[0], zen[1], zen[2], zen[3], zen[4], zen[5]), (zen[6],)],
            ),
        ):
            with self.subTest(size=size):
                actual = list(mi.constrained_batches(iter(zen), size))
                self.assertEqual(actual, expected)

    def test_max_count(self):
        iterable = ['1', '1', '12345678', '12345', '12345']
        max_size = 10
        max_count = 2
        actual = list(mi.constrained_batches(iterable, max_size, max_count))
        expected = [('1', '1'), ('12345678',), ('12345', '12345')]
        self.assertEqual(actual, expected)

    def test_strict(self):
        iterable = ['1', '123456789', '1']
        size = 8
        with self.assertRaises(ValueError):
            list(mi.constrained_batches(iterable, size))

        actual = list(mi.constrained_batches(iterable, size, strict=False))
        expected = [('1',), ('123456789',), ('1',)]
        self.assertEqual(actual, expected)

    def test_get_len(self):
        class Record(tuple):
            def total_size(self):
                return sum(len(x) for x in self)

        record_3 = Record(('1', '23'))
        record_5 = Record(('1234', '1'))
        record_10 = Record(('1', '12345678', '1'))
        record_2 = Record(('1', '1'))
        iterable = [record_3, record_5, record_10, record_2]

        self.assertEqual(
            list(
                mi.constrained_batches(
                    iterable, 10, get_len=lambda x: x.total_size()
                )
            ),
            [(record_3, record_5), (record_10,), (record_2,)],
        )

    def test_bad_max(self):
        with self.assertRaises(ValueError):
            list(mi.constrained_batches([], 0))


class GrayProductTests(TestCase):
    def test_basic(self):
        self.assertEqual(
            tuple(mi.gray_product(('a', 'b', 'c'), range(1, 3))),
            (("a", 1), ("b", 1), ("c", 1), ("c", 2), ("b", 2), ("a", 2)),
        )
        out = mi.gray_product(('foo', 'bar'), (3, 4, 5, 6), ['quz', 'baz'])
        self.assertEqual(next(out), ('foo', 3, 'quz'))
        self.assertEqual(
            list(out),
            [
                ('bar', 3, 'quz'),
                ('bar', 4, 'quz'),
                ('foo', 4, 'quz'),
                ('foo', 5, 'quz'),
                ('bar', 5, 'quz'),
                ('bar', 6, 'quz'),
                ('foo', 6, 'quz'),
                ('foo', 6, 'baz'),
                ('bar', 6, 'baz'),
                ('bar', 5, 'baz'),
                ('foo', 5, 'baz'),
                ('foo', 4, 'baz'),
                ('bar', 4, 'baz'),
                ('bar', 3, 'baz'),
                ('foo', 3, 'baz'),
            ],
        )
        self.assertEqual(tuple(mi.gray_product()), ((),))
        self.assertEqual(tuple(mi.gray_product((1, 2))), ((1,), (2,)))

    def test_errors(self):
        with self.assertRaises(ValueError):
            list(mi.gray_product((1, 2), ()))
        with self.assertRaises(ValueError):
            list(mi.gray_product((1, 2), (2,)))

    def test_vs_product(self):
        iters = (
            ("a", "b"),
            range(3, 6),
            [None, None],
            {"i", "j", "k", "l"},
            "XYZ",
        )
        self.assertEqual(
            sorted(product(*iters)), sorted(mi.gray_product(*iters))
        )


class PartialProductTests(TestCase):
    def test_no_iterables(self):
        self.assertEqual(tuple(mi.partial_product()), ((),))

    def test_empty_iterable(self):
        self.assertEqual(tuple(mi.partial_product('AB', '', 'CD')), ())

    def test_one_iterable(self):
        # a single iterable should pass through
        self.assertEqual(
            tuple(mi.partial_product('ABCD')),
            (
                ('A',),
                ('B',),
                ('C',),
                ('D',),
            ),
        )

    def test_two_iterables(self):
        self.assertEqual(
            list(mi.partial_product('ABCD', [1])),
            [('A', 1), ('B', 1), ('C', 1), ('D', 1)],
        )
        expected = [
            ('A', 1),
            ('B', 1),
            ('C', 1),
            ('D', 1),
            ('D', 2),
            ('D', 3),
            ('D', 4),
        ]
        self.assertEqual(
            list(mi.partial_product('ABCD', [1, 2, 3, 4])), expected
        )

    def test_basic(self):
        ones = [1, 2, 3]
        tens = [10, 20, 30, 40, 50]
        hundreds = [100, 200]

        expected = [
            (1, 10, 100),
            (2, 10, 100),
            (3, 10, 100),
            (3, 20, 100),
            (3, 30, 100),
            (3, 40, 100),
            (3, 50, 100),
            (3, 50, 200),
        ]

        actual = list(mi.partial_product(ones, tens, hundreds))
        self.assertEqual(actual, expected)

    def test_uneven_length_iterables(self):
        # this is also the docstring example
        expected = [
            ('A', 'C', 'D'),
            ('B', 'C', 'D'),
            ('B', 'C', 'E'),
            ('B', 'C', 'F'),
        ]

        self.assertEqual(list(mi.partial_product('AB', 'C', 'DEF')), expected)


class IterateTests(TestCase):
    def test_basic(self) -> None:
        result = list(islice(mi.iterate(lambda x: 2 * x, start=1), 10))
        expected = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        self.assertEqual(result, expected)

    def test_func_controls_iteration_stop(self) -> None:
        def func(num):
            if num > 100:
                raise StopIteration
            return num * 2

        result = list(islice(mi.iterate(func, start=1), 10))
        expected = [1, 2, 4, 8, 16, 32, 64, 128]
        self.assertEqual(result, expected)


class TakewhileInclusiveTests(TestCase):
    def test_basic(self) -> None:
        result = list(mi.takewhile_inclusive(lambda x: x < 5, [1, 4, 6, 4, 1]))
        expected = [1, 4, 6]
        self.assertEqual(result, expected)

    def test_empty_iterator(self) -> None:
        result = list(mi.takewhile_inclusive(lambda x: True, []))
        expected = []
        self.assertEqual(result, expected)

    def test_collatz_sequence(self) -> None:
        is_even = lambda n: n % 2 == 0
        start = 11
        result = list(
            mi.takewhile_inclusive(
                lambda n: n != 1,
                mi.iterate(
                    lambda n: n // 2 if is_even(n) else 3 * n + 1, start
                ),
            )
        )
        expected = [11, 34, 17, 52, 26, 13, 40, 20, 10, 5, 16, 8, 4, 2, 1]
        self.assertEqual(result, expected)


class OuterProductTests(TestCase):
    def test_basic(self) -> None:
        greetings = ['Hello', 'Goodbye']
        names = ['Alice', 'Bob', 'Carol']
        greet = lambda greeting, name: f'{greeting}, {name}!'
        result = list(mi.outer_product(greet, greetings, names))
        expected = [
            ('Hello, Alice!', 'Hello, Bob!', 'Hello, Carol!'),
            ('Goodbye, Alice!', 'Goodbye, Bob!', 'Goodbye, Carol!'),
        ]
        self.assertEqual(result, expected)


class IterSuppressTests(TestCase):
    class Producer:
        def __init__(self, exc, die_early=False):
            self.exc = exc
            self.pos = 0
            self.die_early = die_early

        def __iter__(self):
            if self.die_early:
                raise self.exc

            return self

        def __next__(self):
            ret = self.pos
            if self.pos >= 5:
                raise self.exc
            self.pos += 1
            return ret

    def test_no_error(self):
        iterator = range(5)
        actual = list(mi.iter_suppress(iterator, RuntimeError))
        expected = [0, 1, 2, 3, 4]
        self.assertEqual(actual, expected)

    def test_raises_error(self):
        iterator = self.Producer(ValueError)
        with self.assertRaises(ValueError):
            list(mi.iter_suppress(iterator, RuntimeError))

    def test_suppression(self):
        iterator = self.Producer(ValueError)
        actual = list(mi.iter_suppress(iterator, RuntimeError, ValueError))
        expected = [0, 1, 2, 3, 4]
        self.assertEqual(actual, expected)

    def test_early_suppression(self):
        iterator = self.Producer(ValueError, die_early=True)
        actual = list(mi.iter_suppress(iterator, RuntimeError, ValueError))
        expected = []
        self.assertEqual(actual, expected)


class FilterMapTests(TestCase):
    def test_no_iterables(self):
        actual = list(mi.filter_map(lambda _: None, []))
        expected = []
        self.assertEqual(actual, expected)

    def test_filter(self):
        actual = list(mi.filter_map(lambda _: None, [1, 2, 3]))
        expected = []
        self.assertEqual(actual, expected)

    def test_map(self):
        actual = list(mi.filter_map(lambda x: x + 1, [1, 2, 3]))
        expected = [2, 3, 4]
        self.assertEqual(actual, expected)

    def test_filter_map(self):
        actual = list(
            mi.filter_map(
                lambda x: int(x) if x.isnumeric() else None,
                ['1', 'a', '2', 'b', '3'],
            )
        )
        expected = [1, 2, 3]
        self.assertEqual(actual, expected)


class PowersetOfSetsTests(TestCase):
    def test_simple(self):
        iterable = [0, 1, 2]
        actual = list(mi.powerset_of_sets(iterable))
        expected = [set(), {0}, {1}, {2}, {0, 1}, {0, 2}, {1, 2}, {0, 1, 2}]
        self.assertEqual(actual, expected)

    def test_hash_count(self):
        hash_count = 0

        class Str(str):
            def __hash__(true_self):
                nonlocal hash_count
                hash_count += 1
                return super.__hash__(true_self)

        iterable = map(Str, 'ABBBCDD')
        self.assertEqual(len(list(mi.powerset_of_sets(iterable))), 128)
        self.assertLessEqual(hash_count, 14)


class JoinMappingTests(TestCase):
    def test_basic(self):
        salary_map = {'e1': 12, 'e2': 23, 'e3': 34}
        dept_map = {'e1': 'eng', 'e2': 'sales', 'e3': 'eng'}
        service_map = {'e1': 5, 'e2': 9, 'e3': 2}
        field_to_map = {
            'salary': salary_map,
            'dept': dept_map,
            'service': service_map,
        }
        expected = {
            'e1': {'salary': 12, 'dept': 'eng', 'service': 5},
            'e2': {'salary': 23, 'dept': 'sales', 'service': 9},
            'e3': {'salary': 34, 'dept': 'eng', 'service': 2},
        }
        self.assertEqual(dict(mi.join_mappings(**field_to_map)), expected)

    def test_empty(self):
        self.assertEqual(dict(mi.join_mappings()), {})


class DiscreteFourierTransformTests(TestCase):
    def test_basic(self):
        # Example calculation from:
        # https://en.wikipedia.org/wiki/Discrete_Fourier_transform#Example
        xarr = [1, 2 - 1j, -1j, -1 + 2j]
        Xarr = [2, -2 - 2j, -2j, 4 + 4j]
        self.assertTrue(all(map(cmath.isclose, mi.dft(xarr), Xarr)))
        self.assertTrue(all(map(cmath.isclose, mi.idft(Xarr), xarr)))

    def test_roundtrip(self):
        for _ in range(1_000):
            N = randrange(35)
            xarr = [complex(random(), random()) for i in range(N)]
            Xarr = list(mi.dft(xarr))
            assert all(map(cmath.isclose, mi.idft(Xarr), xarr))


class DoubleStarMapTests(TestCase):
    def test_construction(self):
        iterable = [{'price': 1.23}, {'price': 42}, {'price': 0.1}]
        actual = list(mi.doublestarmap('{price:.2f}'.format, iterable))
        expected = ['1.23', '42.00', '0.10']
        self.assertEqual(actual, expected)

    def test_identity(self):
        iterable = [{'x': 1}, {'x': 2}, {'x': 3}]
        actual = list(mi.doublestarmap(lambda x: x, iterable))
        expected = [1, 2, 3]
        self.assertEqual(actual, expected)

    def test_adding(self):
        iterable = [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]
        actual = list(mi.doublestarmap(lambda a, b: a + b, iterable))
        expected = [3, 7]
        self.assertEqual(actual, expected)

    def test_mismatch_function_smaller(self):
        iterable = [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]
        with self.assertRaises(TypeError):
            list(mi.doublestarmap(lambda a: a, iterable))

    def test_mismatch_function_different(self):
        iterable = [{'a': 1}, {'a': 2}]
        with self.assertRaises(TypeError):
            list(mi.doublestarmap(lambda x: x, iterable))

    def test_mismatch_function_larger(self):
        iterable = [{'a': 1}, {'a': 2}]
        with self.assertRaises(TypeError):
            list(mi.doublestarmap(lambda a, b: a + b, iterable))

    def test_no_mapping(self):
        iterable = [1, 2, 3, 4]
        with self.assertRaises(TypeError):
            list(mi.doublestarmap(lambda x: x, iterable))

    def test_empty(self):
        actual = list(mi.doublestarmap(lambda x: x, []))
        expected = []
        self.assertEqual(actual, expected)
