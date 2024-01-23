from decimal import Decimal
from doctest import DocTestSuite
from fractions import Fraction
from functools import reduce
from itertools import combinations, count, permutations
from operator import mul
from math import factorial
from sys import version_info
from unittest import TestCase, skipIf

import more_itertools as mi


def load_tests(loader, tests, ignore):
    # Add the doctests
    tests.addTests(DocTestSuite('more_itertools.recipes'))
    return tests


class TakeTests(TestCase):
    """Tests for ``take()``"""

    def test_simple_take(self):
        """Test basic usage"""
        t = mi.take(5, range(10))
        self.assertEqual(t, [0, 1, 2, 3, 4])

    def test_null_take(self):
        """Check the null case"""
        t = mi.take(0, range(10))
        self.assertEqual(t, [])

    def test_negative_take(self):
        """Make sure taking negative items results in a ValueError"""
        self.assertRaises(ValueError, lambda: mi.take(-3, range(10)))

    def test_take_too_much(self):
        """Taking more than an iterator has remaining should return what the
        iterator has remaining.

        """
        t = mi.take(10, range(5))
        self.assertEqual(t, [0, 1, 2, 3, 4])


class TabulateTests(TestCase):
    """Tests for ``tabulate()``"""

    def test_simple_tabulate(self):
        """Test the happy path"""
        t = mi.tabulate(lambda x: x)
        f = tuple([next(t) for _ in range(3)])
        self.assertEqual(f, (0, 1, 2))

    def test_count(self):
        """Ensure tabulate accepts specific count"""
        t = mi.tabulate(lambda x: 2 * x, -1)
        f = (next(t), next(t), next(t))
        self.assertEqual(f, (-2, 0, 2))


class TailTests(TestCase):
    """Tests for ``tail()``"""

    def test_iterator_greater(self):
        """Length of iterator is greater than requested tail"""
        self.assertEqual(list(mi.tail(3, iter('ABCDEFG'))), list('EFG'))

    def test_iterator_equal(self):
        """Length of iterator is equal to the requested tail"""
        self.assertEqual(list(mi.tail(7, iter('ABCDEFG'))), list('ABCDEFG'))

    def test_iterator_less(self):
        """Length of iterator is less than requested tail"""
        self.assertEqual(list(mi.tail(8, iter('ABCDEFG'))), list('ABCDEFG'))

    def test_sized_greater(self):
        """Length of sized iterable is greater than requested tail"""
        self.assertEqual(list(mi.tail(3, 'ABCDEFG')), list('EFG'))

    def test_sized_equal(self):
        """Length of sized iterable is less than requested tail"""
        self.assertEqual(list(mi.tail(7, 'ABCDEFG')), list('ABCDEFG'))

    def test_sized_less(self):
        """Length of sized iterable is less than requested tail"""
        self.assertEqual(list(mi.tail(8, 'ABCDEFG')), list('ABCDEFG'))


class ConsumeTests(TestCase):
    """Tests for ``consume()``"""

    def test_sanity(self):
        """Test basic functionality"""
        r = (x for x in range(10))
        mi.consume(r, 3)
        self.assertEqual(3, next(r))

    def test_null_consume(self):
        """Check the null case"""
        r = (x for x in range(10))
        mi.consume(r, 0)
        self.assertEqual(0, next(r))

    def test_negative_consume(self):
        """Check that negative consumption throws an error"""
        r = (x for x in range(10))
        self.assertRaises(ValueError, lambda: mi.consume(r, -1))

    def test_total_consume(self):
        """Check that iterator is totally consumed by default"""
        r = (x for x in range(10))
        mi.consume(r)
        self.assertRaises(StopIteration, lambda: next(r))


class NthTests(TestCase):
    """Tests for ``nth()``"""

    def test_basic(self):
        """Make sure the nth item is returned"""
        l = range(10)
        for i, v in enumerate(l):
            self.assertEqual(mi.nth(l, i), v)

    def test_default(self):
        """Ensure a default value is returned when nth item not found"""
        l = range(3)
        self.assertEqual(mi.nth(l, 100, "zebra"), "zebra")

    def test_negative_item_raises(self):
        """Ensure asking for a negative item raises an exception"""
        self.assertRaises(ValueError, lambda: mi.nth(range(10), -3))


class AllEqualTests(TestCase):
    """Tests for ``all_equal()``"""

    def test_true(self):
        """Everything is equal"""
        self.assertTrue(mi.all_equal('aaaaaa'))
        self.assertTrue(mi.all_equal([0, 0, 0, 0]))

    def test_false(self):
        """Not everything is equal"""
        self.assertFalse(mi.all_equal('aaaaab'))
        self.assertFalse(mi.all_equal([0, 0, 0, 1]))

    def test_tricky(self):
        """Not everything is identical, but everything is equal"""
        items = [1, complex(1, 0), 1.0]
        self.assertTrue(mi.all_equal(items))

    def test_empty(self):
        """Return True if the iterable is empty"""
        self.assertTrue(mi.all_equal(''))
        self.assertTrue(mi.all_equal([]))

    def test_one(self):
        """Return True if the iterable is singular"""
        self.assertTrue(mi.all_equal('0'))
        self.assertTrue(mi.all_equal([0]))


class QuantifyTests(TestCase):
    """Tests for ``quantify()``"""

    def test_happy_path(self):
        """Make sure True count is returned"""
        q = [True, False, True]
        self.assertEqual(mi.quantify(q), 2)

    def test_custom_predicate(self):
        """Ensure non-default predicates return as expected"""
        q = range(10)
        self.assertEqual(mi.quantify(q, lambda x: x % 2 == 0), 5)


class PadnoneTests(TestCase):
    def test_basic(self):
        iterable = range(2)
        for func in (mi.pad_none, mi.padnone):
            with self.subTest(func=func):
                p = func(iterable)
                self.assertEqual(
                    [0, 1, None, None], [next(p) for _ in range(4)]
                )


class NcyclesTests(TestCase):
    """Tests for ``nyclces()``"""

    def test_happy_path(self):
        """cycle a sequence three times"""
        r = ["a", "b", "c"]
        n = mi.ncycles(r, 3)
        self.assertEqual(
            ["a", "b", "c", "a", "b", "c", "a", "b", "c"], list(n)
        )

    def test_null_case(self):
        """asking for 0 cycles should return an empty iterator"""
        n = mi.ncycles(range(100), 0)
        self.assertRaises(StopIteration, lambda: next(n))

    def test_pathological_case(self):
        """asking for negative cycles should return an empty iterator"""
        n = mi.ncycles(range(100), -10)
        self.assertRaises(StopIteration, lambda: next(n))


class DotproductTests(TestCase):
    """Tests for ``dotproduct()``'"""

    def test_happy_path(self):
        """simple dotproduct example"""
        self.assertEqual(400, mi.dotproduct([10, 10], [20, 20]))


class FlattenTests(TestCase):
    """Tests for ``flatten()``"""

    def test_basic_usage(self):
        """ensure list of lists is flattened one level"""
        f = [[0, 1, 2], [3, 4, 5]]
        self.assertEqual(list(range(6)), list(mi.flatten(f)))

    def test_single_level(self):
        """ensure list of lists is flattened only one level"""
        f = [[0, [1, 2]], [[3, 4], 5]]
        self.assertEqual([0, [1, 2], [3, 4], 5], list(mi.flatten(f)))


class RepeatfuncTests(TestCase):
    """Tests for ``repeatfunc()``"""

    def test_simple_repeat(self):
        """test simple repeated functions"""
        r = mi.repeatfunc(lambda: 5)
        self.assertEqual([5, 5, 5, 5, 5], [next(r) for _ in range(5)])

    def test_finite_repeat(self):
        """ensure limited repeat when times is provided"""
        r = mi.repeatfunc(lambda: 5, times=5)
        self.assertEqual([5, 5, 5, 5, 5], list(r))

    def test_added_arguments(self):
        """ensure arguments are applied to the function"""
        r = mi.repeatfunc(lambda x: x, 2, 3)
        self.assertEqual([3, 3], list(r))

    def test_null_times(self):
        """repeat 0 should return an empty iterator"""
        r = mi.repeatfunc(range, 0, 3)
        self.assertRaises(StopIteration, lambda: next(r))


class PairwiseTests(TestCase):
    """Tests for ``pairwise()``"""

    def test_base_case(self):
        """ensure an iterable will return pairwise"""
        p = mi.pairwise([1, 2, 3])
        self.assertEqual([(1, 2), (2, 3)], list(p))

    def test_short_case(self):
        """ensure an empty iterator if there's not enough values to pair"""
        p = mi.pairwise("a")
        self.assertRaises(StopIteration, lambda: next(p))


class GrouperTests(TestCase):
    def test_basic(self):
        seq = 'ABCDEF'
        for n, expected in [
            (3, [('A', 'B', 'C'), ('D', 'E', 'F')]),
            (4, [('A', 'B', 'C', 'D'), ('E', 'F', None, None)]),
            (5, [('A', 'B', 'C', 'D', 'E'), ('F', None, None, None, None)]),
            (6, [('A', 'B', 'C', 'D', 'E', 'F')]),
            (7, [('A', 'B', 'C', 'D', 'E', 'F', None)]),
        ]:
            with self.subTest(n=n):
                actual = list(mi.grouper(iter(seq), n))
                self.assertEqual(actual, expected)

    def test_fill(self):
        seq = 'ABCDEF'
        fillvalue = 'x'
        for n, expected in [
            (1, ['A', 'B', 'C', 'D', 'E', 'F']),
            (2, ['AB', 'CD', 'EF']),
            (3, ['ABC', 'DEF']),
            (4, ['ABCD', 'EFxx']),
            (5, ['ABCDE', 'Fxxxx']),
            (6, ['ABCDEF']),
            (7, ['ABCDEFx']),
        ]:
            with self.subTest(n=n):
                it = mi.grouper(
                    iter(seq), n, incomplete='fill', fillvalue=fillvalue
                )
                actual = [''.join(x) for x in it]
                self.assertEqual(actual, expected)

    def test_ignore(self):
        seq = 'ABCDEF'
        for n, expected in [
            (1, ['A', 'B', 'C', 'D', 'E', 'F']),
            (2, ['AB', 'CD', 'EF']),
            (3, ['ABC', 'DEF']),
            (4, ['ABCD']),
            (5, ['ABCDE']),
            (6, ['ABCDEF']),
            (7, []),
        ]:
            with self.subTest(n=n):
                it = mi.grouper(iter(seq), n, incomplete='ignore')
                actual = [''.join(x) for x in it]
                self.assertEqual(actual, expected)

    def test_strict(self):
        seq = 'ABCDEF'
        for n, expected in [
            (1, ['A', 'B', 'C', 'D', 'E', 'F']),
            (2, ['AB', 'CD', 'EF']),
            (3, ['ABC', 'DEF']),
            (6, ['ABCDEF']),
        ]:
            with self.subTest(n=n):
                it = mi.grouper(iter(seq), n, incomplete='strict')
                actual = [''.join(x) for x in it]
                self.assertEqual(actual, expected)

    def test_strict_fails(self):
        seq = 'ABCDEF'
        for n in [4, 5, 7]:
            with self.subTest(n=n):
                with self.assertRaises(ValueError):
                    list(mi.grouper(iter(seq), n, incomplete='strict'))

    def test_invalid_incomplete(self):
        with self.assertRaises(ValueError):
            list(mi.grouper('ABCD', 3, incomplete='bogus'))


class RoundrobinTests(TestCase):
    """Tests for ``roundrobin()``"""

    def test_even_groups(self):
        """Ensure ordered output from evenly populated iterables"""
        self.assertEqual(
            list(mi.roundrobin('ABC', [1, 2, 3], range(3))),
            ['A', 1, 0, 'B', 2, 1, 'C', 3, 2],
        )

    def test_uneven_groups(self):
        """Ensure ordered output from unevenly populated iterables"""
        self.assertEqual(
            list(mi.roundrobin('ABCD', [1, 2], range(0))),
            ['A', 1, 'B', 2, 'C', 'D'],
        )


class PartitionTests(TestCase):
    """Tests for ``partition()``"""

    def test_bool(self):
        lesser, greater = mi.partition(lambda x: x > 5, range(10))
        self.assertEqual(list(lesser), [0, 1, 2, 3, 4, 5])
        self.assertEqual(list(greater), [6, 7, 8, 9])

    def test_arbitrary(self):
        divisibles, remainders = mi.partition(lambda x: x % 3, range(10))
        self.assertEqual(list(divisibles), [0, 3, 6, 9])
        self.assertEqual(list(remainders), [1, 2, 4, 5, 7, 8])

    def test_pred_is_none(self):
        falses, trues = mi.partition(None, range(3))
        self.assertEqual(list(falses), [0])
        self.assertEqual(list(trues), [1, 2])


class PowersetTests(TestCase):
    """Tests for ``powerset()``"""

    def test_combinatorics(self):
        """Ensure a proper enumeration"""
        p = mi.powerset([1, 2, 3])
        self.assertEqual(
            list(p), [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
        )


class UniqueEverseenTests(TestCase):
    """Tests for ``unique_everseen()``"""

    def test_everseen(self):
        """ensure duplicate elements are ignored"""
        u = mi.unique_everseen('AAAABBBBCCDAABBB')
        self.assertEqual(['A', 'B', 'C', 'D'], list(u))

    def test_custom_key(self):
        """ensure the custom key comparison works"""
        u = mi.unique_everseen('aAbACCc', key=str.lower)
        self.assertEqual(list('abC'), list(u))

    def test_unhashable(self):
        """ensure things work for unhashable items"""
        iterable = ['a', [1, 2, 3], [1, 2, 3], 'a']
        u = mi.unique_everseen(iterable)
        self.assertEqual(list(u), ['a', [1, 2, 3]])

    def test_unhashable_key(self):
        """ensure things work for unhashable items with a custom key"""
        iterable = ['a', [1, 2, 3], [1, 2, 3], 'a']
        u = mi.unique_everseen(iterable, key=lambda x: x)
        self.assertEqual(list(u), ['a', [1, 2, 3]])


class UniqueJustseenTests(TestCase):
    """Tests for ``unique_justseen()``"""

    def test_justseen(self):
        """ensure only last item is remembered"""
        u = mi.unique_justseen('AAAABBBCCDABB')
        self.assertEqual(list('ABCDAB'), list(u))

    def test_custom_key(self):
        """ensure the custom key comparison works"""
        u = mi.unique_justseen('AABCcAD', str.lower)
        self.assertEqual(list('ABCAD'), list(u))


class IterExceptTests(TestCase):
    """Tests for ``iter_except()``"""

    def test_exact_exception(self):
        """ensure the exact specified exception is caught"""
        l = [1, 2, 3]
        i = mi.iter_except(l.pop, IndexError)
        self.assertEqual(list(i), [3, 2, 1])

    def test_generic_exception(self):
        """ensure the generic exception can be caught"""
        l = [1, 2]
        i = mi.iter_except(l.pop, Exception)
        self.assertEqual(list(i), [2, 1])

    def test_uncaught_exception_is_raised(self):
        """ensure a non-specified exception is raised"""
        l = [1, 2, 3]
        i = mi.iter_except(l.pop, KeyError)
        self.assertRaises(IndexError, lambda: list(i))

    def test_first(self):
        """ensure first is run before the function"""
        l = [1, 2, 3]
        f = lambda: 25
        i = mi.iter_except(l.pop, IndexError, f)
        self.assertEqual(list(i), [25, 3, 2, 1])

    def test_multiple(self):
        """ensure can catch multiple exceptions"""

        class Fiz(Exception):
            pass

        class Buzz(Exception):
            pass

        i = 0

        def fizbuzz():
            nonlocal i
            i += 1
            if i % 3 == 0:
                raise Fiz
            if i % 5 == 0:
                raise Buzz
            return i

        expected = ([1, 2], [4], [], [7, 8], [])
        for x in expected:
            self.assertEqual(list(mi.iter_except(fizbuzz, (Fiz, Buzz))), x)


class FirstTrueTests(TestCase):
    """Tests for ``first_true()``"""

    def test_something_true(self):
        """Test with no keywords"""
        self.assertEqual(mi.first_true(range(10)), 1)

    def test_nothing_true(self):
        """Test default return value."""
        self.assertIsNone(mi.first_true([0, 0, 0]))

    def test_default(self):
        """Test with a default keyword"""
        self.assertEqual(mi.first_true([0, 0, 0], default='!'), '!')

    def test_pred(self):
        """Test with a custom predicate"""
        self.assertEqual(
            mi.first_true([2, 4, 6], pred=lambda x: x % 3 == 0), 6
        )


class RandomProductTests(TestCase):
    """Tests for ``random_product()``

    Since random.choice() has different results with the same seed across
    python versions 2.x and 3.x, these tests use highly probably events to
    create predictable outcomes across platforms.
    """

    def test_simple_lists(self):
        """Ensure that one item is chosen from each list in each pair.
        Also ensure that each item from each list eventually appears in
        the chosen combinations.

        Odds are roughly 1 in 7.1 * 10e16 that one item from either list will
        not be chosen after 100 samplings of one item from each list. Just to
        be safe, better use a known random seed, too.

        """
        nums = [1, 2, 3]
        lets = ['a', 'b', 'c']
        n, m = zip(*[mi.random_product(nums, lets) for _ in range(100)])
        n, m = set(n), set(m)
        self.assertEqual(n, set(nums))
        self.assertEqual(m, set(lets))
        self.assertEqual(len(n), len(nums))
        self.assertEqual(len(m), len(lets))

    def test_list_with_repeat(self):
        """ensure multiple items are chosen, and that they appear to be chosen
        from one list then the next, in proper order.

        """
        nums = [1, 2, 3]
        lets = ['a', 'b', 'c']
        r = list(mi.random_product(nums, lets, repeat=100))
        self.assertEqual(2 * 100, len(r))
        n, m = set(r[::2]), set(r[1::2])
        self.assertEqual(n, set(nums))
        self.assertEqual(m, set(lets))
        self.assertEqual(len(n), len(nums))
        self.assertEqual(len(m), len(lets))


class RandomPermutationTests(TestCase):
    """Tests for ``random_permutation()``"""

    def test_full_permutation(self):
        """ensure every item from the iterable is returned in a new ordering

        15 elements have a 1 in 1.3 * 10e12 of appearing in sorted order, so
        we fix a seed value just to be sure.

        """
        i = range(15)
        r = mi.random_permutation(i)
        self.assertEqual(set(i), set(r))
        if i == r:
            raise AssertionError("Values were not permuted")

    def test_partial_permutation(self):
        """ensure all returned items are from the iterable, that the returned
        permutation is of the desired length, and that all items eventually
        get returned.

        Sampling 100 permutations of length 5 from a set of 15 leaves a
        (2/3)^100 chance that an item will not be chosen. Multiplied by 15
        items, there is a 1 in 2.6e16 chance that at least 1 item will not
        show up in the resulting output. Using a random seed will fix that.

        """
        items = range(15)
        item_set = set(items)
        all_items = set()
        for _ in range(100):
            permutation = mi.random_permutation(items, 5)
            self.assertEqual(len(permutation), 5)
            permutation_set = set(permutation)
            self.assertLessEqual(permutation_set, item_set)
            all_items |= permutation_set
        self.assertEqual(all_items, item_set)


class RandomCombinationTests(TestCase):
    """Tests for ``random_combination()``"""

    def test_pseudorandomness(self):
        """ensure different subsets of the iterable get returned over many
        samplings of random combinations"""
        items = range(15)
        all_items = set()
        for _ in range(50):
            combination = mi.random_combination(items, 5)
            all_items |= set(combination)
        self.assertEqual(all_items, set(items))

    def test_no_replacement(self):
        """ensure that elements are sampled without replacement"""
        items = range(15)
        for _ in range(50):
            combination = mi.random_combination(items, len(items))
            self.assertEqual(len(combination), len(set(combination)))
        self.assertRaises(
            ValueError, lambda: mi.random_combination(items, len(items) + 1)
        )


class RandomCombinationWithReplacementTests(TestCase):
    """Tests for ``random_combination_with_replacement()``"""

    def test_replacement(self):
        """ensure that elements are sampled with replacement"""
        items = range(5)
        combo = mi.random_combination_with_replacement(items, len(items) * 2)
        self.assertEqual(2 * len(items), len(combo))
        if len(set(combo)) == len(combo):
            raise AssertionError("Combination contained no duplicates")

    def test_pseudorandomness(self):
        """ensure different subsets of the iterable get returned over many
        samplings of random combinations"""
        items = range(15)
        all_items = set()
        for _ in range(50):
            combination = mi.random_combination_with_replacement(items, 5)
            all_items |= set(combination)
        self.assertEqual(all_items, set(items))


class NthCombinationTests(TestCase):
    def test_basic(self):
        iterable = 'abcdefg'
        r = 4
        for index, expected in enumerate(combinations(iterable, r)):
            actual = mi.nth_combination(iterable, r, index)
            self.assertEqual(actual, expected)

    def test_long(self):
        actual = mi.nth_combination(range(180), 4, 2000000)
        expected = (2, 12, 35, 126)
        self.assertEqual(actual, expected)

    def test_invalid_r(self):
        for r in (-1, 3):
            with self.assertRaises(ValueError):
                mi.nth_combination([], r, 0)

    def test_invalid_index(self):
        with self.assertRaises(IndexError):
            mi.nth_combination('abcdefg', 3, -36)


class NthPermutationTests(TestCase):
    def test_r_less_than_n(self):
        iterable = 'abcde'
        r = 4
        for index, expected in enumerate(permutations(iterable, r)):
            actual = mi.nth_permutation(iterable, r, index)
            self.assertEqual(actual, expected)

    def test_r_equal_to_n(self):
        iterable = 'abcde'
        for index, expected in enumerate(permutations(iterable)):
            actual = mi.nth_permutation(iterable, None, index)
            self.assertEqual(actual, expected)

    def test_long(self):
        iterable = tuple(range(180))
        r = 4
        index = 1000000
        actual = mi.nth_permutation(iterable, r, index)
        expected = mi.nth(permutations(iterable, r), index)
        self.assertEqual(actual, expected)

    def test_null(self):
        actual = mi.nth_permutation([], 0, 0)
        expected = tuple()
        self.assertEqual(actual, expected)

    def test_negative_index(self):
        iterable = 'abcde'
        r = 4
        n = factorial(len(iterable)) // factorial(len(iterable) - r)
        for index, expected in enumerate(permutations(iterable, r)):
            actual = mi.nth_permutation(iterable, r, index - n)
            self.assertEqual(actual, expected)

    def test_invalid_index(self):
        iterable = 'abcde'
        r = 4
        n = factorial(len(iterable)) // factorial(len(iterable) - r)
        for index in [-1 - n, n + 1]:
            with self.assertRaises(IndexError):
                mi.nth_combination(iterable, r, index)

    def test_invalid_r(self):
        iterable = 'abcde'
        r = 4
        n = factorial(len(iterable)) // factorial(len(iterable) - r)
        for r in [-1, n + 1]:
            with self.assertRaises(ValueError):
                mi.nth_combination(iterable, r, 0)


class PrependTests(TestCase):
    def test_basic(self):
        value = 'a'
        iterator = iter('bcdefg')
        actual = list(mi.prepend(value, iterator))
        expected = list('abcdefg')
        self.assertEqual(actual, expected)

    def test_multiple(self):
        value = 'ab'
        iterator = iter('cdefg')
        actual = tuple(mi.prepend(value, iterator))
        expected = ('ab',) + tuple('cdefg')
        self.assertEqual(actual, expected)


class Convolvetests(TestCase):
    def test_moving_average(self):
        signal = iter([10, 20, 30, 40, 50])
        kernel = [0.5, 0.5]
        actual = list(mi.convolve(signal, kernel))
        expected = [
            (10 + 0) / 2,
            (20 + 10) / 2,
            (30 + 20) / 2,
            (40 + 30) / 2,
            (50 + 40) / 2,
            (0 + 50) / 2,
        ]
        self.assertEqual(actual, expected)

    def test_derivative(self):
        signal = iter([10, 20, 30, 40, 50])
        kernel = [1, -1]
        actual = list(mi.convolve(signal, kernel))
        expected = [10 - 0, 20 - 10, 30 - 20, 40 - 30, 50 - 40, 0 - 50]
        self.assertEqual(actual, expected)

    def test_infinite_signal(self):
        signal = count()
        kernel = [1, -1]
        actual = mi.take(5, mi.convolve(signal, kernel))
        expected = [0, 1, 1, 1, 1]
        self.assertEqual(actual, expected)


class BeforeAndAfterTests(TestCase):
    def test_empty(self):
        before, after = mi.before_and_after(bool, [])
        self.assertEqual(list(before), [])
        self.assertEqual(list(after), [])

    def test_never_true(self):
        before, after = mi.before_and_after(bool, [0, False, None, ''])
        self.assertEqual(list(before), [])
        self.assertEqual(list(after), [0, False, None, ''])

    def test_never_false(self):
        before, after = mi.before_and_after(bool, [1, True, Ellipsis, ' '])
        self.assertEqual(list(before), [1, True, Ellipsis, ' '])
        self.assertEqual(list(after), [])

    def test_some_true(self):
        before, after = mi.before_and_after(bool, [1, True, 0, False])
        self.assertEqual(list(before), [1, True])
        self.assertEqual(list(after), [0, False])

    @staticmethod
    def _group_events(events):
        events = iter(events)

        while True:
            try:
                operation = next(events)
            except StopIteration:
                break
            assert operation in ["SUM", "MULTIPLY"]

            # Here, the remainder `events` is passed into `before_and_after`
            # again, which would be problematic if the remainder is a
            # generator function (as in Python 3.10 itertools recipes), since
            # that creates recursion. `itertools.chain` solves this problem.
            numbers, events = mi.before_and_after(
                lambda e: isinstance(e, int), events
            )

            yield (operation, numbers)

    def test_nested_remainder(self):
        events = ["SUM", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 1000
        events += ["MULTIPLY", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 1000

        for operation, numbers in self._group_events(events):
            if operation == "SUM":
                res = sum(numbers)
                self.assertEqual(res, 55)
            elif operation == "MULTIPLY":
                res = reduce(lambda a, b: a * b, numbers)
                self.assertEqual(res, 3628800)


class TriplewiseTests(TestCase):
    def test_basic(self):
        for iterable, expected in [
            ([0], []),
            ([0, 1], []),
            ([0, 1, 2], [(0, 1, 2)]),
            ([0, 1, 2, 3], [(0, 1, 2), (1, 2, 3)]),
            ([0, 1, 2, 3, 4], [(0, 1, 2), (1, 2, 3), (2, 3, 4)]),
        ]:
            with self.subTest(expected=expected):
                actual = list(mi.triplewise(iterable))
                self.assertEqual(actual, expected)


class SlidingWindowTests(TestCase):
    def test_basic(self):
        for iterable, n, expected in [
            ([], 1, []),
            ([0], 1, [(0,)]),
            ([0, 1], 1, [(0,), (1,)]),
            ([0, 1, 2], 2, [(0, 1), (1, 2)]),
            ([0, 1, 2], 3, [(0, 1, 2)]),
            ([0, 1, 2], 4, []),
            ([0, 1, 2, 3], 4, [(0, 1, 2, 3)]),
            ([0, 1, 2, 3, 4], 4, [(0, 1, 2, 3), (1, 2, 3, 4)]),
        ]:
            with self.subTest(expected=expected):
                actual = list(mi.sliding_window(iterable, n))
                self.assertEqual(actual, expected)


class SubslicesTests(TestCase):
    def test_basic(self):
        for iterable, expected in [
            ([], []),
            ([1], [[1]]),
            ([1, 2], [[1], [1, 2], [2]]),
            (iter([1, 2]), [[1], [1, 2], [2]]),
            ([2, 1], [[2], [2, 1], [1]]),
            (
                'ABCD',
                [
                    ['A'],
                    ['A', 'B'],
                    ['A', 'B', 'C'],
                    ['A', 'B', 'C', 'D'],
                    ['B'],
                    ['B', 'C'],
                    ['B', 'C', 'D'],
                    ['C'],
                    ['C', 'D'],
                    ['D'],
                ],
            ),
        ]:
            with self.subTest(expected=expected):
                actual = list(mi.subslices(iterable))
                self.assertEqual(actual, expected)


class PolynomialFromRootsTests(TestCase):
    def test_basic(self):
        for roots, expected in [
            ((2, 1, -1), [1, -2, -1, 2]),
            ((2, 3), [1, -5, 6]),
            ((1, 2, 3), [1, -6, 11, -6]),
            ((2, 4, 1), [1, -7, 14, -8]),
        ]:
            with self.subTest(roots=roots):
                actual = mi.polynomial_from_roots(roots)
                self.assertEqual(actual, expected)


class PolynomialEvalTests(TestCase):
    def test_basic(self):
        for coefficients, x, expected in [
            ([1, -4, -17, 60], 2, 18),
            ([1, -4, -17, 60], 2.5, 8.125),
            ([1, -4, -17, 60], Fraction(2, 3), Fraction(1274, 27)),
            ([1, -4, -17, 60], Decimal('1.75'), Decimal('23.359375')),
            ([], 2, 0),
            ([], 2.5, 0.0),
            ([], Fraction(2, 3), Fraction(0, 1)),
            ([], Decimal('1.75'), Decimal('0.00')),
            ([11], 7, 11),
            ([11, 2], 7, 79),
        ]:
            with self.subTest(x=x):
                actual = mi.polynomial_eval(coefficients, x)
                self.assertEqual(actual, expected)
                self.assertEqual(type(actual), type(x))


class IterIndexTests(TestCase):
    def test_basic(self):
        iterable = 'AABCADEAF'
        for wrapper in (list, iter):
            with self.subTest(wrapper=wrapper):
                actual = list(mi.iter_index(wrapper(iterable), 'A'))
                expected = [0, 1, 4, 7]
                self.assertEqual(actual, expected)

    def test_start(self):
        for wrapper in (list, iter):
            with self.subTest(wrapper=wrapper):
                iterable = 'AABCADEAF'
                i = -1
                actual = []
                while True:
                    try:
                        i = next(
                            mi.iter_index(wrapper(iterable), 'A', start=i + 1)
                        )
                    except StopIteration:
                        break
                    else:
                        actual.append(i)

                expected = [0, 1, 4, 7]
                self.assertEqual(actual, expected)

    def test_stop(self):
        actual = list(mi.iter_index('AABCADEAF', 'A', stop=7))
        expected = [0, 1, 4]
        self.assertEqual(actual, expected)


class SieveTests(TestCase):
    def test_basic(self):
        self.assertEqual(
            list(mi.sieve(67)),
            [
                2,
                3,
                5,
                7,
                11,
                13,
                17,
                19,
                23,
                29,
                31,
                37,
                41,
                43,
                47,
                53,
                59,
                61,
            ],
        )
        self.assertEqual(list(mi.sieve(68))[-1], 67)

    def test_prime_counts(self):
        for n, expected in (
            (100, 25),
            (1_000, 168),
            (10_000, 1229),
            (100_000, 9592),
            (1_000_000, 78498),
        ):
            with self.subTest(n=n):
                self.assertEqual(mi.ilen(mi.sieve(n)), expected)

    def test_small_numbers(self):
        with self.assertRaises(ValueError):
            list(mi.sieve(-1))

        for n in (0, 1, 2):
            with self.subTest(n=n):
                self.assertEqual(list(mi.sieve(n)), [])


class BatchedTests(TestCase):
    def test_basic(self):
        iterable = range(1, 5 + 1)
        for n, expected in (
            (1, [(1,), (2,), (3,), (4,), (5,)]),
            (2, [(1, 2), (3, 4), (5,)]),
            (3, [(1, 2, 3), (4, 5)]),
            (4, [(1, 2, 3, 4), (5,)]),
            (5, [(1, 2, 3, 4, 5)]),
            (6, [(1, 2, 3, 4, 5)]),
        ):
            with self.subTest(n=n):
                actual = list(mi.batched(iterable, n))
                self.assertEqual(actual, expected)

    def test_strict(self):
        with self.assertRaises(ValueError):
            list(mi.batched('ABCDEFG', 3, strict=True))

        self.assertEqual(
            list(mi.batched('ABCDEF', 3, strict=True)),
            [('A', 'B', 'C'), ('D', 'E', 'F')],
        )


class TransposeTests(TestCase):
    def test_empty(self):
        it = []
        actual = list(mi.transpose(it))
        expected = []
        self.assertEqual(actual, expected)

    def test_basic(self):
        it = [(10, 11, 12), (20, 21, 22), (30, 31, 32)]
        actual = list(mi.transpose(it))
        expected = [(10, 20, 30), (11, 21, 31), (12, 22, 32)]
        self.assertEqual(actual, expected)

    @skipIf(version_info[:2] < (3, 10), 'strict=True missing on 3.9')
    def test_incompatible_error(self):
        it = [(10, 11, 12, 13), (20, 21, 22), (30, 31, 32)]
        with self.assertRaises(ValueError):
            list(mi.transpose(it))

    @skipIf(version_info[:2] >= (3, 9), 'strict=True missing on 3.9')
    def test_incompatible_allow(self):
        it = [(10, 11, 12, 13), (20, 21, 22), (30, 31, 32)]
        actual = list(mi.transpose(it))
        expected = [(10, 20, 30), (11, 21, 31), (12, 22, 32)]
        self.assertEqual(actual, expected)


class ReshapeTests(TestCase):
    def test_empty(self):
        actual = list(mi.reshape([], 3))
        self.assertEqual(actual, [])

    def test_zero(self):
        matrix = [(0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11)]
        with self.assertRaises(ValueError):
            list(mi.reshape(matrix, 0))

    def test_basic(self):
        matrix = [(0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11)]
        for cols, expected in (
            (
                1,
                [
                    (0,),
                    (1,),
                    (2,),
                    (3,),
                    (4,),
                    (5,),
                    (6,),
                    (7,),
                    (8,),
                    (9,),
                    (10,),
                    (11,),
                ],
            ),
            (2, [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]),
            (3, [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11)]),
            (4, [(0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11)]),
            (6, [(0, 1, 2, 3, 4, 5), (6, 7, 8, 9, 10, 11)]),
            (12, [(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)]),
        ):
            with self.subTest(cols=cols):
                actual = list(mi.reshape(matrix, cols))
                self.assertEqual(actual, expected)


class MatMulTests(TestCase):
    def test_n_by_n(self):
        actual = list(mi.matmul([(7, 5), (3, 5)], [[2, 5], [7, 9]]))
        expected = [(49, 80), (41, 60)]
        self.assertEqual(actual, expected)

    def test_m_by_n(self):
        m1 = [[2, 5], [7, 9], [3, 4]]
        m2 = [[7, 11, 5, 4, 9], [3, 5, 2, 6, 3]]
        actual = list(mi.matmul(m1, m2))
        expected = [
            (29, 47, 20, 38, 33),
            (76, 122, 53, 82, 90),
            (33, 53, 23, 36, 39),
        ]
        self.assertEqual(actual, expected)


class FactorTests(TestCase):
    def test_basic(self):
        for n, expected in (
            (0, []),
            (1, []),
            (2, [2]),
            (3, [3]),
            (4, [2, 2]),
            (6, [2, 3]),
            (360, [2, 2, 2, 3, 3, 5]),
            (128_884_753_939, [128_884_753_939]),
            (999953 * 999983, [999953, 999983]),
            (909_909_090_909, [3, 3, 7, 13, 13, 751, 113797]),
        ):
            with self.subTest(n=n):
                actual = list(mi.factor(n))
                self.assertEqual(actual, expected)

    def test_cross_check(self):
        prod = lambda x: reduce(mul, x, 1)
        self.assertTrue(all(prod(mi.factor(n)) == n for n in range(1, 2000)))
        self.assertTrue(
            all(set(mi.factor(n)) <= set(mi.sieve(n + 1)) for n in range(2000))
        )
        self.assertTrue(
            all(
                list(mi.factor(n)) == sorted(mi.factor(n)) for n in range(2000)
            )
        )


class SumOfSquaresTests(TestCase):
    def test_basic(self):
        for it, expected in (
            ([], 0),
            ([1, 2, 3], 1 + 4 + 9),
            ([2, 4, 6, 8], 4 + 16 + 36 + 64),
        ):
            with self.subTest(it=it):
                actual = mi.sum_of_squares(it)
                self.assertEqual(actual, expected)


class PolynomialDerivativeTests(TestCase):
    def test_basic(self):
        for coefficients, expected in [
            ([], []),
            ([1], []),
            ([1, 2], [1]),
            ([1, 2, 3], [2, 2]),
            ([1, 2, 3, 4], [3, 4, 3]),
            ([1.1, 2, 3, 4], [(1.1 * 3), 4, 3]),
        ]:
            with self.subTest(coefficients=coefficients):
                actual = mi.polynomial_derivative(coefficients)
                self.assertEqual(actual, expected)


class TotientTests(TestCase):
    def test_basic(self):
        for n, expected in (
            (1, 1),
            (2, 1),
            (3, 2),
            (4, 2),
            (9, 6),
            (12, 4),
            (128_884_753_939, 128_884_753_938),
            (999953 * 999983, 999952 * 999982),
            (6**20, 1 * 2**19 * 2 * 3**19),
        ):
            with self.subTest(n=n):
                self.assertEqual(mi.totient(n), expected)
