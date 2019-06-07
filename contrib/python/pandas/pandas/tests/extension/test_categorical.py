"""
This file contains a minimal set of tests for compliance with the extension
array interface test suite, and should contain no other tests.
The test suite for the full functionality of the array is located in
`pandas/tests/arrays/`.

The tests in this file are inherited from the BaseExtensionTests, and only
minimal tweaks should be applied to get the tests passing (by overwriting a
parent method).

Additional tests should either be added to one of the BaseExtensionTests
classes (if they are relevant for the extension interface for all dtypes), or
be added to the array-specific tests in `pandas/tests/arrays/`.

"""
import string

import numpy as np
import pytest

import pandas as pd
from pandas import Categorical
from pandas.api.types import CategoricalDtype
from pandas.tests.extension import base


def make_data():
    while True:
        values = np.random.choice(list(string.ascii_letters), size=100)
        # ensure we meet the requirements
        # 1. first two not null
        # 2. first and second are different
        if values[0] != values[1]:
            break
    return values


@pytest.fixture
def dtype():
    return CategoricalDtype()


@pytest.fixture
def data():
    """Length-100 array for this type.

    * data[0] and data[1] should both be non missing
    * data[0] and data[1] should not gbe equal
    """
    return Categorical(make_data())


@pytest.fixture
def data_missing():
    """Length 2 array with [NA, Valid]"""
    return Categorical([np.nan, 'A'])


@pytest.fixture
def data_for_sorting():
    return Categorical(['A', 'B', 'C'], categories=['C', 'A', 'B'],
                       ordered=True)


@pytest.fixture
def data_missing_for_sorting():
    return Categorical(['A', None, 'B'], categories=['B', 'A'],
                       ordered=True)


@pytest.fixture
def na_value():
    return np.nan


@pytest.fixture
def data_for_grouping():
    return Categorical(['a', 'a', None, None, 'b', 'b', 'a', 'c'])


class TestDtype(base.BaseDtypeTests):
    pass


class TestInterface(base.BaseInterfaceTests):
    @pytest.mark.skip(reason="Memory usage doesn't match")
    def test_memory_usage(self, data):
        # Is this deliberate?
        super(TestInterface, self).test_memory_usage(data)


class TestConstructors(base.BaseConstructorsTests):
    pass


class TestReshaping(base.BaseReshapingTests):
    pass


class TestGetitem(base.BaseGetitemTests):
    skip_take = pytest.mark.skip(reason="GH-20664.")

    @pytest.mark.skip(reason="Backwards compatibility")
    def test_getitem_scalar(self, data):
        # CategoricalDtype.type isn't "correct" since it should
        # be a parent of the elements (object). But don't want
        # to break things by changing.
        super(TestGetitem, self).test_getitem_scalar(data)

    @skip_take
    def test_take(self, data, na_value, na_cmp):
        # TODO remove this once Categorical.take is fixed
        super(TestGetitem, self).test_take(data, na_value, na_cmp)

    @skip_take
    def test_take_negative(self, data):
        super().test_take_negative(data)

    @skip_take
    def test_take_pandas_style_negative_raises(self, data, na_value):
        super().test_take_pandas_style_negative_raises(data, na_value)

    @skip_take
    def test_take_non_na_fill_value(self, data_missing):
        super().test_take_non_na_fill_value(data_missing)

    @skip_take
    def test_take_out_of_bounds_raises(self, data, allow_fill):
        return super().test_take_out_of_bounds_raises(data, allow_fill)

    @pytest.mark.skip(reason="GH-20747. Unobserved categories.")
    def test_take_series(self, data):
        super().test_take_series(data)

    @skip_take
    def test_reindex_non_na_fill_value(self, data_missing):
        super().test_reindex_non_na_fill_value(data_missing)

    @pytest.mark.skip(reason="Categorical.take buggy")
    def test_take_empty(self, data, na_value, na_cmp):
        super().test_take_empty(data, na_value, na_cmp)

    @pytest.mark.skip(reason="test not written correctly for categorical")
    def test_reindex(self, data, na_value):
        super().test_reindex(data, na_value)


class TestSetitem(base.BaseSetitemTests):
    pass


class TestMissing(base.BaseMissingTests):

    @pytest.mark.skip(reason="Not implemented")
    def test_fillna_limit_pad(self, data_missing):
        super().test_fillna_limit_pad(data_missing)

    @pytest.mark.skip(reason="Not implemented")
    def test_fillna_limit_backfill(self, data_missing):
        super().test_fillna_limit_backfill(data_missing)


class TestReduce(base.BaseNoReduceTests):
    pass


class TestMethods(base.BaseMethodsTests):
    @pytest.mark.skip(reason="Unobserved categories included")
    def test_value_counts(self, all_data, dropna):
        return super().test_value_counts(all_data, dropna)

    def test_combine_add(self, data_repeated):
        # GH 20825
        # When adding categoricals in combine, result is a string
        orig_data1, orig_data2 = data_repeated(2)
        s1 = pd.Series(orig_data1)
        s2 = pd.Series(orig_data2)
        result = s1.combine(s2, lambda x1, x2: x1 + x2)
        expected = pd.Series(([a + b for (a, b) in
                               zip(list(orig_data1), list(orig_data2))]))
        self.assert_series_equal(result, expected)

        val = s1.iloc[0]
        result = s1.combine(val, lambda x1, x2: x1 + x2)
        expected = pd.Series([a + val for a in list(orig_data1)])
        self.assert_series_equal(result, expected)

    @pytest.mark.skip(reason="Not Applicable")
    def test_fillna_length_mismatch(self, data_missing):
        super().test_fillna_length_mismatch(data_missing)

    def test_searchsorted(self, data_for_sorting):
        if not data_for_sorting.ordered:
            raise pytest.skip(reason="searchsorted requires ordered data.")


class TestCasting(base.BaseCastingTests):
    pass


class TestArithmeticOps(base.BaseArithmeticOpsTests):

    def test_arith_series_with_scalar(self, data, all_arithmetic_operators):

        op_name = all_arithmetic_operators
        if op_name != '__rmod__':
            super(TestArithmeticOps, self).test_arith_series_with_scalar(
                data, op_name)
        else:
            pytest.skip('rmod never called when string is first argument')

    def test_add_series_with_extension_array(self, data):
        ser = pd.Series(data)
        with pytest.raises(TypeError, match="cannot perform"):
            ser + data

    def _check_divmod_op(self, s, op, other, exc=NotImplementedError):
        return super(TestArithmeticOps, self)._check_divmod_op(
            s, op, other, exc=TypeError
        )


class TestComparisonOps(base.BaseComparisonOpsTests):

    def _compare_other(self, s, data, op_name, other):
        op = self.get_op_from_name(op_name)
        if op_name == '__eq__':
            result = op(s, other)
            expected = s.combine(other, lambda x, y: x == y)
            assert (result == expected).all()

        elif op_name == '__ne__':
            result = op(s, other)
            expected = s.combine(other, lambda x, y: x != y)
            assert (result == expected).all()

        else:
            with pytest.raises(TypeError):
                op(data, other)


class TestParsing(base.BaseParsingTests):
    pass
