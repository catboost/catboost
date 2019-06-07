import numpy as np
import pytest

from pandas.errors import PerformanceWarning

import pandas as pd
from pandas import SparseArray, SparseDtype
from pandas.tests.extension import base
import pandas.util.testing as tm


def make_data(fill_value):
    if np.isnan(fill_value):
        data = np.random.uniform(size=100)
    else:
        data = np.random.randint(1, 100, size=100)
        if data[0] == data[1]:
            data[0] += 1

    data[2::3] = fill_value
    return data


@pytest.fixture
def dtype():
    return SparseDtype()


@pytest.fixture(params=[0, np.nan])
def data(request):
    """Length-100 PeriodArray for semantics test."""
    res = SparseArray(make_data(request.param),
                      fill_value=request.param)
    return res


@pytest.fixture(params=[0, np.nan])
def data_missing(request):
    """Length 2 array with [NA, Valid]"""
    return SparseArray([np.nan, 1], fill_value=request.param)


@pytest.fixture(params=[0, np.nan])
def data_repeated(request):
    """Return different versions of data for count times"""
    def gen(count):
        for _ in range(count):
            yield SparseArray(make_data(request.param),
                              fill_value=request.param)
    yield gen


@pytest.fixture(params=[0, np.nan])
def data_for_sorting(request):
    return SparseArray([2, 3, 1], fill_value=request.param)


@pytest.fixture(params=[0, np.nan])
def data_missing_for_sorting(request):
    return SparseArray([2, np.nan, 1], fill_value=request.param)


@pytest.fixture
def na_value():
    return np.nan


@pytest.fixture
def na_cmp():
    return lambda left, right: pd.isna(left) and pd.isna(right)


@pytest.fixture(params=[0, np.nan])
def data_for_grouping(request):
    return SparseArray([1, 1, np.nan, np.nan, 2, 2, 1, 3],
                       fill_value=request.param)


class BaseSparseTests(object):

    def _check_unsupported(self, data):
        if data.dtype == SparseDtype(int, 0):
            pytest.skip("Can't store nan in int array.")


class TestDtype(BaseSparseTests, base.BaseDtypeTests):

    def test_array_type_with_arg(self, data, dtype):
        assert dtype.construct_array_type() is SparseArray


class TestInterface(BaseSparseTests, base.BaseInterfaceTests):
    def test_no_values_attribute(self, data):
        pytest.skip("We have values")


class TestConstructors(BaseSparseTests, base.BaseConstructorsTests):
    pass


class TestReshaping(BaseSparseTests, base.BaseReshapingTests):

    def test_concat_mixed_dtypes(self, data):
        # https://github.com/pandas-dev/pandas/issues/20762
        # This should be the same, aside from concat([sparse, float])
        df1 = pd.DataFrame({'A': data[:3]})
        df2 = pd.DataFrame({"A": [1, 2, 3]})
        df3 = pd.DataFrame({"A": ['a', 'b', 'c']}).astype('category')
        dfs = [df1, df2, df3]

        # dataframes
        result = pd.concat(dfs)
        expected = pd.concat([x.apply(lambda s: np.asarray(s).astype(object))
                              for x in dfs])
        self.assert_frame_equal(result, expected)

    def test_concat_columns(self, data, na_value):
        self._check_unsupported(data)
        super(TestReshaping, self).test_concat_columns(data, na_value)

    def test_align(self, data, na_value):
        self._check_unsupported(data)
        super(TestReshaping, self).test_align(data, na_value)

    def test_align_frame(self, data, na_value):
        self._check_unsupported(data)
        super(TestReshaping, self).test_align_frame(data, na_value)

    def test_align_series_frame(self, data, na_value):
        self._check_unsupported(data)
        super(TestReshaping, self).test_align_series_frame(data, na_value)

    def test_merge(self, data, na_value):
        self._check_unsupported(data)
        super(TestReshaping, self).test_merge(data, na_value)


class TestGetitem(BaseSparseTests, base.BaseGetitemTests):

    def test_get(self, data):
        s = pd.Series(data, index=[2 * i for i in range(len(data))])
        if np.isnan(s.values.fill_value):
            assert np.isnan(s.get(4)) and np.isnan(s.iloc[2])
        else:
            assert s.get(4) == s.iloc[2]
        assert s.get(2) == s.iloc[1]

    def test_reindex(self, data, na_value):
        self._check_unsupported(data)
        super(TestGetitem, self).test_reindex(data, na_value)


# Skipping TestSetitem, since we don't implement it.

class TestMissing(BaseSparseTests, base.BaseMissingTests):

    def test_isna(self, data_missing):
        expected_dtype = SparseDtype(bool,
                                     pd.isna(data_missing.dtype.fill_value))
        expected = SparseArray([True, False], dtype=expected_dtype)

        result = pd.isna(data_missing)
        self.assert_equal(result, expected)

        result = pd.Series(data_missing).isna()
        expected = pd.Series(expected)
        self.assert_series_equal(result, expected)

        # GH 21189
        result = pd.Series(data_missing).drop([0, 1]).isna()
        expected = pd.Series([], dtype=expected_dtype)
        self.assert_series_equal(result, expected)

    def test_fillna_limit_pad(self, data_missing):
        with tm.assert_produces_warning(PerformanceWarning):
            super(TestMissing, self).test_fillna_limit_pad(data_missing)

    def test_fillna_limit_backfill(self, data_missing):
        with tm.assert_produces_warning(PerformanceWarning):
            super(TestMissing, self).test_fillna_limit_backfill(data_missing)

    def test_fillna_series_method(self, data_missing):
        with tm.assert_produces_warning(PerformanceWarning):
            super(TestMissing, self).test_fillna_limit_backfill(data_missing)

    @pytest.mark.skip(reason="Unsupported")
    def test_fillna_series(self):
        # this one looks doable.
        pass

    def test_fillna_frame(self, data_missing):
        # Have to override to specify that fill_value will change.
        fill_value = data_missing[1]

        result = pd.DataFrame({
            "A": data_missing,
            "B": [1, 2]
        }).fillna(fill_value)

        if pd.isna(data_missing.fill_value):
            dtype = SparseDtype(data_missing.dtype, fill_value)
        else:
            dtype = data_missing.dtype

        expected = pd.DataFrame({
            "A": data_missing._from_sequence([fill_value, fill_value],
                                             dtype=dtype),
            "B": [1, 2],
        })

        self.assert_frame_equal(result, expected)


class TestMethods(BaseSparseTests, base.BaseMethodsTests):

    def test_combine_le(self, data_repeated):
        # We return a Series[SparseArray].__le__ returns a
        # Series[Sparse[bool]]
        # rather than Series[bool]
        orig_data1, orig_data2 = data_repeated(2)
        s1 = pd.Series(orig_data1)
        s2 = pd.Series(orig_data2)
        result = s1.combine(s2, lambda x1, x2: x1 <= x2)
        expected = pd.Series(pd.SparseArray([
            a <= b for (a, b) in
            zip(list(orig_data1), list(orig_data2))
        ], fill_value=False))
        self.assert_series_equal(result, expected)

        val = s1.iloc[0]
        result = s1.combine(val, lambda x1, x2: x1 <= x2)
        expected = pd.Series(pd.SparseArray([
            a <= val for a in list(orig_data1)
        ], fill_value=False))
        self.assert_series_equal(result, expected)

    def test_fillna_copy_frame(self, data_missing):
        arr = data_missing.take([1, 1])
        df = pd.DataFrame({"A": arr})

        filled_val = df.iloc[0, 0]
        result = df.fillna(filled_val)

        assert df.values.base is not result.values.base
        assert df.A._values.to_dense() is arr.to_dense()

    def test_fillna_copy_series(self, data_missing):
        arr = data_missing.take([1, 1])
        ser = pd.Series(arr)

        filled_val = ser[0]
        result = ser.fillna(filled_val)

        assert ser._values is not result._values
        assert ser._values.to_dense() is arr.to_dense()

    @pytest.mark.skip(reason="Not Applicable")
    def test_fillna_length_mismatch(self, data_missing):
        pass

    def test_where_series(self, data, na_value):
        assert data[0] != data[1]
        cls = type(data)
        a, b = data[:2]

        ser = pd.Series(cls._from_sequence([a, a, b, b], dtype=data.dtype))

        cond = np.array([True, True, False, False])
        result = ser.where(cond)

        new_dtype = SparseDtype('float', 0.0)
        expected = pd.Series(cls._from_sequence([a, a, na_value, na_value],
                                                dtype=new_dtype))
        self.assert_series_equal(result, expected)

        other = cls._from_sequence([a, b, a, b], dtype=data.dtype)
        cond = np.array([True, False, True, True])
        result = ser.where(cond, other)
        expected = pd.Series(cls._from_sequence([a, b, b, b],
                                                dtype=data.dtype))
        self.assert_series_equal(result, expected)

    def test_combine_first(self, data):
        if data.dtype.subtype == 'int':
            # Right now this is upcasted to float, just like combine_first
            # for Series[int]
            pytest.skip("TODO(SparseArray.__setitem__ will preserve dtype.")
        super(TestMethods, self).test_combine_first(data)

    @pytest.mark.parametrize("as_series", [True, False])
    def test_searchsorted(self, data_for_sorting, as_series):
        with tm.assert_produces_warning(PerformanceWarning):
            super(TestMethods, self).test_searchsorted(data_for_sorting,
                                                       as_series=as_series)


class TestCasting(BaseSparseTests, base.BaseCastingTests):
    pass


class TestArithmeticOps(BaseSparseTests, base.BaseArithmeticOpsTests):
    series_scalar_exc = None
    frame_scalar_exc = None
    divmod_exc = None
    series_array_exc = None

    def _skip_if_different_combine(self, data):
        if data.fill_value == 0:
            # arith ops call on dtype.fill_value so that the sparsity
            # is maintained. Combine can't be called on a dtype in
            # general, so we can't make the expected. This is tested elsewhere
            raise pytest.skip("Incorrected expected from Series.combine")

    def test_error(self, data, all_arithmetic_operators):
        pass

    def test_arith_series_with_scalar(self, data, all_arithmetic_operators):
        self._skip_if_different_combine(data)
        super(TestArithmeticOps, self).test_arith_series_with_scalar(
            data,
            all_arithmetic_operators
        )

    def test_arith_series_with_array(self, data, all_arithmetic_operators):
        self._skip_if_different_combine(data)
        super(TestArithmeticOps, self).test_arith_series_with_array(
            data,
            all_arithmetic_operators
        )


class TestComparisonOps(BaseSparseTests, base.BaseComparisonOpsTests):

    def _compare_other(self, s, data, op_name, other):
        op = self.get_op_from_name(op_name)

        # array
        result = pd.Series(op(data, other))
        # hard to test the fill value, since we don't know what expected
        # is in general.
        # Rely on tests in `tests/sparse` to validate that.
        assert isinstance(result.dtype, SparseDtype)
        assert result.dtype.subtype == np.dtype('bool')

        with np.errstate(all='ignore'):
            expected = pd.Series(
                pd.SparseArray(op(np.asarray(data), np.asarray(other)),
                               fill_value=result.values.fill_value)
            )

        tm.assert_series_equal(result, expected)

        # series
        s = pd.Series(data)
        result = op(s, other)
        tm.assert_series_equal(result, expected)


class TestPrinting(BaseSparseTests, base.BasePrintingTests):
    @pytest.mark.xfail(reason='Different repr', strict=True)
    def test_array_repr(self, data, size):
        super(TestPrinting, self).test_array_repr(data, size)


class TestParsing(BaseSparseTests, base.BaseParsingTests):
    @pytest.mark.parametrize('engine', ['c', 'python'])
    def test_EA_types(self, engine, data):
        expected_msg = r'.*must implement _from_sequence_of_strings.*'
        with pytest.raises(NotImplementedError, match=expected_msg):
            super(TestParsing, self).test_EA_types(engine, data)
