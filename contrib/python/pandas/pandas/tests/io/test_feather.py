""" test feather-format compat """
from distutils.version import LooseVersion

import numpy as np
import pytest

import pandas as pd
import pandas.util.testing as tm
from pandas.util.testing import assert_frame_equal, ensure_clean

from pandas.io.feather_format import read_feather, to_feather  # noqa:E402

pyarrow = pytest.importorskip('pyarrow')


pyarrow_version = LooseVersion(pyarrow.__version__)


@pytest.mark.single
class TestFeather(object):

    def check_error_on_write(self, df, exc):
        # check that we are raising the exception
        # on writing

        with pytest.raises(exc):
            with ensure_clean() as path:
                to_feather(df, path)

    def check_round_trip(self, df, expected=None, **kwargs):

        if expected is None:
            expected = df

        with ensure_clean() as path:
            to_feather(df, path)

            result = read_feather(path, **kwargs)
            assert_frame_equal(result, expected)

    def test_error(self):

        for obj in [pd.Series([1, 2, 3]), 1, 'foo', pd.Timestamp('20130101'),
                    np.array([1, 2, 3])]:
            self.check_error_on_write(obj, ValueError)

    def test_basic(self):

        df = pd.DataFrame({'string': list('abc'),
                           'int': list(range(1, 4)),
                           'uint': np.arange(3, 6).astype('u1'),
                           'float': np.arange(4.0, 7.0, dtype='float64'),
                           'float_with_null': [1., np.nan, 3],
                           'bool': [True, False, True],
                           'bool_with_null': [True, np.nan, False],
                           'cat': pd.Categorical(list('abc')),
                           'dt': pd.date_range('20130101', periods=3),
                           'dttz': pd.date_range('20130101', periods=3,
                                                 tz='US/Eastern'),
                           'dt_with_null': [pd.Timestamp('20130101'), pd.NaT,
                                            pd.Timestamp('20130103')],
                           'dtns': pd.date_range('20130101', periods=3,
                                                 freq='ns')})

        assert df.dttz.dtype.tz.zone == 'US/Eastern'
        self.check_round_trip(df)

    def test_duplicate_columns(self):

        # https://github.com/wesm/feather/issues/53
        # not currently able to handle duplicate columns
        df = pd.DataFrame(np.arange(12).reshape(4, 3),
                          columns=list('aaa')).copy()
        self.check_error_on_write(df, ValueError)

    def test_stringify_columns(self):

        df = pd.DataFrame(np.arange(12).reshape(4, 3)).copy()
        self.check_error_on_write(df, ValueError)

    def test_read_columns(self):
        # GH 24025
        df = pd.DataFrame({'col1': list('abc'),
                           'col2': list(range(1, 4)),
                           'col3': list('xyz'),
                           'col4': list(range(4, 7))})
        columns = ['col1', 'col3']
        self.check_round_trip(df, expected=df[columns],
                              columns=columns)

    def test_unsupported_other(self):

        # period
        df = pd.DataFrame({'a': pd.period_range('2013', freq='M', periods=3)})
        # Some versions raise ValueError, others raise ArrowInvalid.
        self.check_error_on_write(df, Exception)

    def test_rw_nthreads(self):
        df = pd.DataFrame({'A': np.arange(100000)})
        expected_warning = (
            "the 'nthreads' keyword is deprecated, "
            "use 'use_threads' instead"
        )
        # TODO: make the warning work with check_stacklevel=True
        with tm.assert_produces_warning(
                FutureWarning, check_stacklevel=False) as w:
            self.check_round_trip(df, nthreads=2)
        # we have an extra FutureWarning because of #GH23752
        assert any(expected_warning in str(x) for x in w)

        # TODO: make the warning work with check_stacklevel=True
        with tm.assert_produces_warning(
                FutureWarning, check_stacklevel=False) as w:
            self.check_round_trip(df, nthreads=1)
        # we have an extra FutureWarnings because of #GH23752
        assert any(expected_warning in str(x) for x in w)

    def test_rw_use_threads(self):
        df = pd.DataFrame({'A': np.arange(100000)})
        self.check_round_trip(df, use_threads=True)
        self.check_round_trip(df, use_threads=False)

    def test_write_with_index(self):

        df = pd.DataFrame({'A': [1, 2, 3]})
        self.check_round_trip(df)

        # non-default index
        for index in [[2, 3, 4],
                      pd.date_range('20130101', periods=3),
                      list('abc'),
                      [1, 3, 4],
                      pd.MultiIndex.from_tuples([('a', 1), ('a', 2),
                                                 ('b', 1)]),
                      ]:

            df.index = index
            self.check_error_on_write(df, ValueError)

        # index with meta-data
        df.index = [0, 1, 2]
        df.index.name = 'foo'
        self.check_error_on_write(df, ValueError)

        # column multi-index
        df.index = [0, 1, 2]
        df.columns = pd.MultiIndex.from_tuples([('a', 1), ('a', 2), ('b', 1)]),
        self.check_error_on_write(df, ValueError)

    def test_path_pathlib(self):
        df = tm.makeDataFrame().reset_index()
        result = tm.round_trip_pathlib(df.to_feather, pd.read_feather)
        tm.assert_frame_equal(df, result)

    def test_path_localpath(self):
        df = tm.makeDataFrame().reset_index()
        result = tm.round_trip_localpath(df.to_feather, pd.read_feather)
        tm.assert_frame_equal(df, result)
