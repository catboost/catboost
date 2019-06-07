from datetime import timedelta
from functools import partial
from operator import attrgetter

import dateutil
import numpy as np
import pytest
import pytz

from pandas._libs.tslibs import OutOfBoundsDatetime, conversion

import pandas as pd
from pandas import (
    DatetimeIndex, Index, Timestamp, date_range, datetime, offsets,
    to_datetime)
from pandas.core.arrays import DatetimeArray, period_array
import pandas.util.testing as tm


class TestDatetimeIndex(object):

    @pytest.mark.parametrize('dt_cls', [DatetimeIndex,
                                        DatetimeArray._from_sequence])
    def test_freq_validation_with_nat(self, dt_cls):
        # GH#11587 make sure we get a useful error message when generate_range
        #  raises
        msg = ("Inferred frequency None from passed values does not conform "
               "to passed frequency D")
        with pytest.raises(ValueError, match=msg):
            dt_cls([pd.NaT, pd.Timestamp('2011-01-01')], freq='D')
        with pytest.raises(ValueError, match=msg):
            dt_cls([pd.NaT, pd.Timestamp('2011-01-01').value],
                   freq='D')

    def test_categorical_preserves_tz(self):
        # GH#18664 retain tz when going DTI-->Categorical-->DTI
        # TODO: parametrize over DatetimeIndex/DatetimeArray
        #  once CategoricalIndex(DTA) works

        dti = pd.DatetimeIndex(
            [pd.NaT, '2015-01-01', '1999-04-06 15:14:13', '2015-01-01'],
            tz='US/Eastern')

        ci = pd.CategoricalIndex(dti)
        carr = pd.Categorical(dti)
        cser = pd.Series(ci)

        for obj in [ci, carr, cser]:
            result = pd.DatetimeIndex(obj)
            tm.assert_index_equal(result, dti)

    def test_dti_with_period_data_raises(self):
        # GH#23675
        data = pd.PeriodIndex(['2016Q1', '2016Q2'], freq='Q')

        with pytest.raises(TypeError, match="PeriodDtype data is invalid"):
            DatetimeIndex(data)

        with pytest.raises(TypeError, match="PeriodDtype data is invalid"):
            to_datetime(data)

        with pytest.raises(TypeError, match="PeriodDtype data is invalid"):
            DatetimeIndex(period_array(data))

        with pytest.raises(TypeError, match="PeriodDtype data is invalid"):
            to_datetime(period_array(data))

    def test_dti_with_timedelta64_data_deprecation(self):
        # GH#23675
        data = np.array([0], dtype='m8[ns]')
        with tm.assert_produces_warning(FutureWarning):
            result = DatetimeIndex(data)

        assert result[0] == Timestamp('1970-01-01')

        with tm.assert_produces_warning(FutureWarning, check_stacklevel=False):
            result = to_datetime(data)

        assert result[0] == Timestamp('1970-01-01')

        with tm.assert_produces_warning(FutureWarning):
            result = DatetimeIndex(pd.TimedeltaIndex(data))

        assert result[0] == Timestamp('1970-01-01')

        with tm.assert_produces_warning(FutureWarning, check_stacklevel=False):
            result = to_datetime(pd.TimedeltaIndex(data))

        assert result[0] == Timestamp('1970-01-01')

    def test_construction_caching(self):

        df = pd.DataFrame({'dt': pd.date_range('20130101', periods=3),
                           'dttz': pd.date_range('20130101', periods=3,
                                                 tz='US/Eastern'),
                           'dt_with_null': [pd.Timestamp('20130101'), pd.NaT,
                                            pd.Timestamp('20130103')],
                           'dtns': pd.date_range('20130101', periods=3,
                                                 freq='ns')})
        assert df.dttz.dtype.tz.zone == 'US/Eastern'

    @pytest.mark.parametrize('kwargs', [
        {'tz': 'dtype.tz'},
        {'dtype': 'dtype'},
        {'dtype': 'dtype', 'tz': 'dtype.tz'}])
    def test_construction_with_alt(self, kwargs, tz_aware_fixture):
        tz = tz_aware_fixture
        i = pd.date_range('20130101', periods=5, freq='H', tz=tz)
        kwargs = {key: attrgetter(val)(i) for key, val in kwargs.items()}
        result = DatetimeIndex(i, **kwargs)
        tm.assert_index_equal(i, result)

    @pytest.mark.parametrize('kwargs', [
        {'tz': 'dtype.tz'},
        {'dtype': 'dtype'},
        {'dtype': 'dtype', 'tz': 'dtype.tz'}])
    def test_construction_with_alt_tz_localize(self, kwargs, tz_aware_fixture):
        tz = tz_aware_fixture
        i = pd.date_range('20130101', periods=5, freq='H', tz=tz)
        kwargs = {key: attrgetter(val)(i) for key, val in kwargs.items()}

        if str(tz) in ('UTC', 'tzutc()'):
            warn = None
        else:
            warn = FutureWarning

        with tm.assert_produces_warning(warn, check_stacklevel=False):
            result = DatetimeIndex(i.tz_localize(None).asi8, **kwargs)
        expected = DatetimeIndex(i, **kwargs)
        tm.assert_index_equal(result, expected)

        # localize into the provided tz
        i2 = DatetimeIndex(i.tz_localize(None).asi8, tz='UTC')
        expected = i.tz_localize(None).tz_localize('UTC')
        tm.assert_index_equal(i2, expected)

        # incompat tz/dtype
        pytest.raises(ValueError, lambda: DatetimeIndex(
            i.tz_localize(None).asi8, dtype=i.dtype, tz='US/Pacific'))

    def test_construction_index_with_mixed_timezones(self):
        # gh-11488: no tz results in DatetimeIndex
        result = Index([Timestamp('2011-01-01'),
                        Timestamp('2011-01-02')], name='idx')
        exp = DatetimeIndex([Timestamp('2011-01-01'),
                             Timestamp('2011-01-02')], name='idx')
        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)
        assert result.tz is None

        # same tz results in DatetimeIndex
        result = Index([Timestamp('2011-01-01 10:00', tz='Asia/Tokyo'),
                        Timestamp('2011-01-02 10:00', tz='Asia/Tokyo')],
                       name='idx')
        exp = DatetimeIndex(
            [Timestamp('2011-01-01 10:00'), Timestamp('2011-01-02 10:00')
             ], tz='Asia/Tokyo', name='idx')
        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)
        assert result.tz is not None
        assert result.tz == exp.tz

        # same tz results in DatetimeIndex (DST)
        result = Index([Timestamp('2011-01-01 10:00', tz='US/Eastern'),
                        Timestamp('2011-08-01 10:00', tz='US/Eastern')],
                       name='idx')
        exp = DatetimeIndex([Timestamp('2011-01-01 10:00'),
                             Timestamp('2011-08-01 10:00')],
                            tz='US/Eastern', name='idx')
        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)
        assert result.tz is not None
        assert result.tz == exp.tz

        # Different tz results in Index(dtype=object)
        result = Index([Timestamp('2011-01-01 10:00'),
                        Timestamp('2011-01-02 10:00', tz='US/Eastern')],
                       name='idx')
        exp = Index([Timestamp('2011-01-01 10:00'),
                     Timestamp('2011-01-02 10:00', tz='US/Eastern')],
                    dtype='object', name='idx')
        tm.assert_index_equal(result, exp, exact=True)
        assert not isinstance(result, DatetimeIndex)

        result = Index([Timestamp('2011-01-01 10:00', tz='Asia/Tokyo'),
                        Timestamp('2011-01-02 10:00', tz='US/Eastern')],
                       name='idx')
        exp = Index([Timestamp('2011-01-01 10:00', tz='Asia/Tokyo'),
                     Timestamp('2011-01-02 10:00', tz='US/Eastern')],
                    dtype='object', name='idx')
        tm.assert_index_equal(result, exp, exact=True)
        assert not isinstance(result, DatetimeIndex)

        # length = 1
        result = Index([Timestamp('2011-01-01')], name='idx')
        exp = DatetimeIndex([Timestamp('2011-01-01')], name='idx')
        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)
        assert result.tz is None

        # length = 1 with tz
        result = Index(
            [Timestamp('2011-01-01 10:00', tz='Asia/Tokyo')], name='idx')
        exp = DatetimeIndex([Timestamp('2011-01-01 10:00')], tz='Asia/Tokyo',
                            name='idx')
        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)
        assert result.tz is not None
        assert result.tz == exp.tz

    def test_construction_index_with_mixed_timezones_with_NaT(self):
        # see gh-11488
        result = Index([pd.NaT, Timestamp('2011-01-01'),
                        pd.NaT, Timestamp('2011-01-02')], name='idx')
        exp = DatetimeIndex([pd.NaT, Timestamp('2011-01-01'),
                             pd.NaT, Timestamp('2011-01-02')], name='idx')
        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)
        assert result.tz is None

        # Same tz results in DatetimeIndex
        result = Index([pd.NaT, Timestamp('2011-01-01 10:00', tz='Asia/Tokyo'),
                        pd.NaT, Timestamp('2011-01-02 10:00',
                                          tz='Asia/Tokyo')],
                       name='idx')
        exp = DatetimeIndex([pd.NaT, Timestamp('2011-01-01 10:00'),
                             pd.NaT, Timestamp('2011-01-02 10:00')],
                            tz='Asia/Tokyo', name='idx')
        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)
        assert result.tz is not None
        assert result.tz == exp.tz

        # same tz results in DatetimeIndex (DST)
        result = Index([Timestamp('2011-01-01 10:00', tz='US/Eastern'),
                        pd.NaT,
                        Timestamp('2011-08-01 10:00', tz='US/Eastern')],
                       name='idx')
        exp = DatetimeIndex([Timestamp('2011-01-01 10:00'), pd.NaT,
                             Timestamp('2011-08-01 10:00')],
                            tz='US/Eastern', name='idx')
        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)
        assert result.tz is not None
        assert result.tz == exp.tz

        # different tz results in Index(dtype=object)
        result = Index([pd.NaT, Timestamp('2011-01-01 10:00'),
                        pd.NaT, Timestamp('2011-01-02 10:00',
                                          tz='US/Eastern')],
                       name='idx')
        exp = Index([pd.NaT, Timestamp('2011-01-01 10:00'),
                     pd.NaT, Timestamp('2011-01-02 10:00', tz='US/Eastern')],
                    dtype='object', name='idx')
        tm.assert_index_equal(result, exp, exact=True)
        assert not isinstance(result, DatetimeIndex)

        result = Index([pd.NaT, Timestamp('2011-01-01 10:00', tz='Asia/Tokyo'),
                        pd.NaT, Timestamp('2011-01-02 10:00',
                                          tz='US/Eastern')], name='idx')
        exp = Index([pd.NaT, Timestamp('2011-01-01 10:00', tz='Asia/Tokyo'),
                     pd.NaT, Timestamp('2011-01-02 10:00', tz='US/Eastern')],
                    dtype='object', name='idx')
        tm.assert_index_equal(result, exp, exact=True)
        assert not isinstance(result, DatetimeIndex)

        # all NaT
        result = Index([pd.NaT, pd.NaT], name='idx')
        exp = DatetimeIndex([pd.NaT, pd.NaT], name='idx')
        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)
        assert result.tz is None

        # all NaT with tz
        result = Index([pd.NaT, pd.NaT], tz='Asia/Tokyo', name='idx')
        exp = DatetimeIndex([pd.NaT, pd.NaT], tz='Asia/Tokyo', name='idx')

        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)
        assert result.tz is not None
        assert result.tz == exp.tz

    def test_construction_dti_with_mixed_timezones(self):
        # GH 11488 (not changed, added explicit tests)

        # no tz results in DatetimeIndex
        result = DatetimeIndex(
            [Timestamp('2011-01-01'), Timestamp('2011-01-02')], name='idx')
        exp = DatetimeIndex(
            [Timestamp('2011-01-01'), Timestamp('2011-01-02')], name='idx')
        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)

        # same tz results in DatetimeIndex
        result = DatetimeIndex([Timestamp('2011-01-01 10:00', tz='Asia/Tokyo'),
                                Timestamp('2011-01-02 10:00',
                                          tz='Asia/Tokyo')],
                               name='idx')
        exp = DatetimeIndex([Timestamp('2011-01-01 10:00'),
                             Timestamp('2011-01-02 10:00')],
                            tz='Asia/Tokyo', name='idx')
        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)

        # same tz results in DatetimeIndex (DST)
        result = DatetimeIndex([Timestamp('2011-01-01 10:00', tz='US/Eastern'),
                                Timestamp('2011-08-01 10:00',
                                          tz='US/Eastern')],
                               name='idx')
        exp = DatetimeIndex([Timestamp('2011-01-01 10:00'),
                             Timestamp('2011-08-01 10:00')],
                            tz='US/Eastern', name='idx')
        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)

        # tz mismatch affecting to tz-aware raises TypeError/ValueError

        with pytest.raises(ValueError):
            DatetimeIndex([Timestamp('2011-01-01 10:00', tz='Asia/Tokyo'),
                           Timestamp('2011-01-02 10:00', tz='US/Eastern')],
                          name='idx')

        msg = 'cannot be converted to datetime64'
        with pytest.raises(ValueError, match=msg):
            DatetimeIndex([Timestamp('2011-01-01 10:00'),
                           Timestamp('2011-01-02 10:00', tz='US/Eastern')],
                          tz='Asia/Tokyo', name='idx')

        with pytest.raises(ValueError):
            DatetimeIndex([Timestamp('2011-01-01 10:00', tz='Asia/Tokyo'),
                           Timestamp('2011-01-02 10:00', tz='US/Eastern')],
                          tz='US/Eastern', name='idx')

        with pytest.raises(ValueError, match=msg):
            # passing tz should results in DatetimeIndex, then mismatch raises
            # TypeError
            Index([pd.NaT, Timestamp('2011-01-01 10:00'),
                   pd.NaT, Timestamp('2011-01-02 10:00', tz='US/Eastern')],
                  tz='Asia/Tokyo', name='idx')

    def test_construction_base_constructor(self):
        arr = [pd.Timestamp('2011-01-01'), pd.NaT, pd.Timestamp('2011-01-03')]
        tm.assert_index_equal(pd.Index(arr), pd.DatetimeIndex(arr))
        tm.assert_index_equal(pd.Index(np.array(arr)),
                              pd.DatetimeIndex(np.array(arr)))

        arr = [np.nan, pd.NaT, pd.Timestamp('2011-01-03')]
        tm.assert_index_equal(pd.Index(arr), pd.DatetimeIndex(arr))
        tm.assert_index_equal(pd.Index(np.array(arr)),
                              pd.DatetimeIndex(np.array(arr)))

    def test_construction_outofbounds(self):
        # GH 13663
        dates = [datetime(3000, 1, 1), datetime(4000, 1, 1),
                 datetime(5000, 1, 1), datetime(6000, 1, 1)]
        exp = Index(dates, dtype=object)
        # coerces to object
        tm.assert_index_equal(Index(dates), exp)

        with pytest.raises(OutOfBoundsDatetime):
            # can't create DatetimeIndex
            DatetimeIndex(dates)

    def test_construction_with_ndarray(self):
        # GH 5152
        dates = [datetime(2013, 10, 7),
                 datetime(2013, 10, 8),
                 datetime(2013, 10, 9)]
        data = DatetimeIndex(dates, freq=pd.offsets.BDay()).values
        result = DatetimeIndex(data, freq=pd.offsets.BDay())
        expected = DatetimeIndex(['2013-10-07',
                                  '2013-10-08',
                                  '2013-10-09'],
                                 freq='B')
        tm.assert_index_equal(result, expected)

    def test_verify_integrity_deprecated(self):
        # GH#23919
        with tm.assert_produces_warning(FutureWarning):
            DatetimeIndex(['1/1/2000'], verify_integrity=False)

    def test_range_kwargs_deprecated(self):
        # GH#23919
        with tm.assert_produces_warning(FutureWarning):
            DatetimeIndex(start='1/1/2000', end='1/10/2000', freq='D')

    def test_integer_values_and_tz_deprecated(self):
        # GH-24559
        values = np.array([946684800000000000])
        with tm.assert_produces_warning(FutureWarning):
            result = DatetimeIndex(values, tz='US/Central')
        expected = pd.DatetimeIndex(['2000-01-01T00:00:00'], tz="US/Central")
        tm.assert_index_equal(result, expected)

        # but UTC is *not* deprecated.
        with tm.assert_produces_warning(None):
            result = DatetimeIndex(values, tz='UTC')
        expected = pd.DatetimeIndex(['2000-01-01T00:00:00'], tz="US/Central")

    def test_constructor_coverage(self):
        rng = date_range('1/1/2000', periods=10.5)
        exp = date_range('1/1/2000', periods=10)
        tm.assert_index_equal(rng, exp)

        msg = 'periods must be a number, got foo'
        with pytest.raises(TypeError, match=msg):
            date_range(start='1/1/2000', periods='foo', freq='D')

        with pytest.raises(ValueError):
            with tm.assert_produces_warning(FutureWarning):
                DatetimeIndex(start='1/1/2000', end='1/10/2000')

        with pytest.raises(TypeError):
            DatetimeIndex('1/1/2000')

        # generator expression
        gen = (datetime(2000, 1, 1) + timedelta(i) for i in range(10))
        result = DatetimeIndex(gen)
        expected = DatetimeIndex([datetime(2000, 1, 1) + timedelta(i)
                                  for i in range(10)])
        tm.assert_index_equal(result, expected)

        # NumPy string array
        strings = np.array(['2000-01-01', '2000-01-02', '2000-01-03'])
        result = DatetimeIndex(strings)
        expected = DatetimeIndex(strings.astype('O'))
        tm.assert_index_equal(result, expected)

        from_ints = DatetimeIndex(expected.asi8)
        tm.assert_index_equal(from_ints, expected)

        # string with NaT
        strings = np.array(['2000-01-01', '2000-01-02', 'NaT'])
        result = DatetimeIndex(strings)
        expected = DatetimeIndex(strings.astype('O'))
        tm.assert_index_equal(result, expected)

        from_ints = DatetimeIndex(expected.asi8)
        tm.assert_index_equal(from_ints, expected)

        # non-conforming
        pytest.raises(ValueError, DatetimeIndex,
                      ['2000-01-01', '2000-01-02', '2000-01-04'], freq='D')

        pytest.raises(ValueError, date_range, start='2011-01-01',
                      freq='b')
        pytest.raises(ValueError, date_range, end='2011-01-01',
                      freq='B')
        pytest.raises(ValueError, date_range, periods=10, freq='D')

    @pytest.mark.parametrize('freq', ['AS', 'W-SUN'])
    def test_constructor_datetime64_tzformat(self, freq):
        # see GH#6572: ISO 8601 format results in pytz.FixedOffset
        idx = date_range('2013-01-01T00:00:00-05:00',
                         '2016-01-01T23:59:59-05:00', freq=freq)
        expected = date_range('2013-01-01T00:00:00', '2016-01-01T23:59:59',
                              freq=freq, tz=pytz.FixedOffset(-300))
        tm.assert_index_equal(idx, expected)
        # Unable to use `US/Eastern` because of DST
        expected_i8 = date_range('2013-01-01T00:00:00',
                                 '2016-01-01T23:59:59', freq=freq,
                                 tz='America/Lima')
        tm.assert_numpy_array_equal(idx.asi8, expected_i8.asi8)

        idx = date_range('2013-01-01T00:00:00+09:00',
                         '2016-01-01T23:59:59+09:00', freq=freq)
        expected = date_range('2013-01-01T00:00:00', '2016-01-01T23:59:59',
                              freq=freq, tz=pytz.FixedOffset(540))
        tm.assert_index_equal(idx, expected)
        expected_i8 = date_range('2013-01-01T00:00:00',
                                 '2016-01-01T23:59:59', freq=freq,
                                 tz='Asia/Tokyo')
        tm.assert_numpy_array_equal(idx.asi8, expected_i8.asi8)

        # Non ISO 8601 format results in dateutil.tz.tzoffset
        idx = date_range('2013/1/1 0:00:00-5:00', '2016/1/1 23:59:59-5:00',
                         freq=freq)
        expected = date_range('2013-01-01T00:00:00', '2016-01-01T23:59:59',
                              freq=freq, tz=pytz.FixedOffset(-300))
        tm.assert_index_equal(idx, expected)
        # Unable to use `US/Eastern` because of DST
        expected_i8 = date_range('2013-01-01T00:00:00',
                                 '2016-01-01T23:59:59', freq=freq,
                                 tz='America/Lima')
        tm.assert_numpy_array_equal(idx.asi8, expected_i8.asi8)

        idx = date_range('2013/1/1 0:00:00+9:00',
                         '2016/1/1 23:59:59+09:00', freq=freq)
        expected = date_range('2013-01-01T00:00:00', '2016-01-01T23:59:59',
                              freq=freq, tz=pytz.FixedOffset(540))
        tm.assert_index_equal(idx, expected)
        expected_i8 = date_range('2013-01-01T00:00:00',
                                 '2016-01-01T23:59:59', freq=freq,
                                 tz='Asia/Tokyo')
        tm.assert_numpy_array_equal(idx.asi8, expected_i8.asi8)

    def test_constructor_dtype(self):

        # passing a dtype with a tz should localize
        idx = DatetimeIndex(['2013-01-01', '2013-01-02'],
                            dtype='datetime64[ns, US/Eastern]')
        expected = DatetimeIndex(['2013-01-01', '2013-01-02']
                                 ).tz_localize('US/Eastern')
        tm.assert_index_equal(idx, expected)

        idx = DatetimeIndex(['2013-01-01', '2013-01-02'],
                            tz='US/Eastern')
        tm.assert_index_equal(idx, expected)

        # if we already have a tz and its not the same, then raise
        idx = DatetimeIndex(['2013-01-01', '2013-01-02'],
                            dtype='datetime64[ns, US/Eastern]')

        pytest.raises(ValueError,
                      lambda: DatetimeIndex(idx,
                                            dtype='datetime64[ns]'))

        # this is effectively trying to convert tz's
        pytest.raises(TypeError,
                      lambda: DatetimeIndex(idx,
                                            dtype='datetime64[ns, CET]'))
        pytest.raises(ValueError,
                      lambda: DatetimeIndex(
                          idx, tz='CET',
                          dtype='datetime64[ns, US/Eastern]'))
        result = DatetimeIndex(idx, dtype='datetime64[ns, US/Eastern]')
        tm.assert_index_equal(idx, result)

    def test_constructor_name(self):
        idx = date_range(start='2000-01-01', periods=1, freq='A',
                         name='TEST')
        assert idx.name == 'TEST'

    def test_000constructor_resolution(self):
        # 2252
        t1 = Timestamp((1352934390 * 1000000000) + 1000000 + 1000 + 1)
        idx = DatetimeIndex([t1])

        assert idx.nanosecond[0] == t1.nanosecond

    def test_disallow_setting_tz(self):
        # GH 3746
        dti = DatetimeIndex(['2010'], tz='UTC')
        with pytest.raises(AttributeError):
            dti.tz = pytz.timezone('US/Pacific')

    @pytest.mark.parametrize('tz', [
        None, 'America/Los_Angeles', pytz.timezone('America/Los_Angeles'),
        Timestamp('2000', tz='America/Los_Angeles').tz])
    def test_constructor_start_end_with_tz(self, tz):
        # GH 18595
        start = Timestamp('2013-01-01 06:00:00', tz='America/Los_Angeles')
        end = Timestamp('2013-01-02 06:00:00', tz='America/Los_Angeles')
        result = date_range(freq='D', start=start, end=end, tz=tz)
        expected = DatetimeIndex(['2013-01-01 06:00:00',
                                  '2013-01-02 06:00:00'],
                                 tz='America/Los_Angeles')
        tm.assert_index_equal(result, expected)
        # Especially assert that the timezone is consistent for pytz
        assert pytz.timezone('America/Los_Angeles') is result.tz

    @pytest.mark.parametrize('tz', ['US/Pacific', 'US/Eastern', 'Asia/Tokyo'])
    def test_constructor_with_non_normalized_pytz(self, tz):
        # GH 18595
        non_norm_tz = Timestamp('2010', tz=tz).tz
        result = DatetimeIndex(['2010'], tz=non_norm_tz)
        assert pytz.timezone(tz) is result.tz

    def test_constructor_timestamp_near_dst(self):
        # GH 20854
        ts = [Timestamp('2016-10-30 03:00:00+0300', tz='Europe/Helsinki'),
              Timestamp('2016-10-30 03:00:00+0200', tz='Europe/Helsinki')]
        result = DatetimeIndex(ts)
        expected = DatetimeIndex([ts[0].to_pydatetime(),
                                  ts[1].to_pydatetime()])
        tm.assert_index_equal(result, expected)

    # TODO(GH-24559): Remove the xfail for the tz-aware case.
    @pytest.mark.parametrize('klass', [Index, DatetimeIndex])
    @pytest.mark.parametrize('box', [
        np.array, partial(np.array, dtype=object), list])
    @pytest.mark.parametrize('tz, dtype', [
        pytest.param('US/Pacific', 'datetime64[ns, US/Pacific]',
                     marks=[pytest.mark.xfail(),
                            pytest.mark.filterwarnings(
                                "ignore:\\n    Passing:FutureWarning")]),
        [None, 'datetime64[ns]'],
    ])
    def test_constructor_with_int_tz(self, klass, box, tz, dtype):
        # GH 20997, 20964
        ts = Timestamp('2018-01-01', tz=tz)
        result = klass(box([ts.value]), dtype=dtype)
        expected = klass([ts])
        assert result == expected

    # This is the desired future behavior
    @pytest.mark.xfail(reason="Future behavior", strict=False)
    @pytest.mark.filterwarnings("ignore:\\n    Passing:FutureWarning")
    def test_construction_int_rountrip(self, tz_naive_fixture):
        # GH 12619
        # TODO(GH-24559): Remove xfail
        tz = tz_naive_fixture
        result = 1293858000000000000
        expected = DatetimeIndex([1293858000000000000], tz=tz).asi8[0]
        assert result == expected

    def test_construction_from_replaced_timestamps_with_dst(self):
        # GH 18785
        index = pd.date_range(pd.Timestamp(2000, 1, 1),
                              pd.Timestamp(2005, 1, 1),
                              freq='MS', tz='Australia/Melbourne')
        test = pd.DataFrame({'data': range(len(index))}, index=index)
        test = test.resample('Y').mean()
        result = pd.DatetimeIndex([x.replace(month=6, day=1)
                                   for x in test.index])
        expected = pd.DatetimeIndex(['2000-06-01 00:00:00',
                                     '2001-06-01 00:00:00',
                                     '2002-06-01 00:00:00',
                                     '2003-06-01 00:00:00',
                                     '2004-06-01 00:00:00',
                                     '2005-06-01 00:00:00'],
                                    tz='Australia/Melbourne')
        tm.assert_index_equal(result, expected)

    def test_construction_with_tz_and_tz_aware_dti(self):
        # GH 23579
        dti = date_range('2016-01-01', periods=3, tz='US/Central')
        with pytest.raises(TypeError):
            DatetimeIndex(dti, tz='Asia/Tokyo')

    def test_construction_with_nat_and_tzlocal(self):
        tz = dateutil.tz.tzlocal()
        result = DatetimeIndex(['2018', 'NaT'], tz=tz)
        expected = DatetimeIndex([Timestamp('2018', tz=tz), pd.NaT])
        tm.assert_index_equal(result, expected)

    def test_constructor_no_precision_warns(self):
        # GH-24753, GH-24739
        expected = pd.DatetimeIndex(['2000'], dtype='datetime64[ns]')

        # we set the stacklevel for DatetimeIndex
        with tm.assert_produces_warning(FutureWarning):
            result = pd.DatetimeIndex(['2000'], dtype='datetime64')
        tm.assert_index_equal(result, expected)

        with tm.assert_produces_warning(FutureWarning, check_stacklevel=False):
            result = pd.Index(['2000'], dtype='datetime64')
        tm.assert_index_equal(result, expected)

    def test_constructor_wrong_precision_raises(self):
        with pytest.raises(ValueError):
            pd.DatetimeIndex(['2000'], dtype='datetime64[us]')


class TestTimeSeries(object):

    def test_dti_constructor_preserve_dti_freq(self):
        rng = date_range('1/1/2000', '1/2/2000', freq='5min')

        rng2 = DatetimeIndex(rng)
        assert rng.freq == rng2.freq

    def test_dti_constructor_years_only(self, tz_naive_fixture):
        tz = tz_naive_fixture
        # GH 6961
        rng1 = date_range('2014', '2015', freq='M', tz=tz)
        expected1 = date_range('2014-01-31', '2014-12-31', freq='M', tz=tz)

        rng2 = date_range('2014', '2015', freq='MS', tz=tz)
        expected2 = date_range('2014-01-01', '2015-01-01', freq='MS', tz=tz)

        rng3 = date_range('2014', '2020', freq='A', tz=tz)
        expected3 = date_range('2014-12-31', '2019-12-31', freq='A', tz=tz)

        rng4 = date_range('2014', '2020', freq='AS', tz=tz)
        expected4 = date_range('2014-01-01', '2020-01-01', freq='AS', tz=tz)

        for rng, expected in [(rng1, expected1), (rng2, expected2),
                              (rng3, expected3), (rng4, expected4)]:
            tm.assert_index_equal(rng, expected)

    def test_dti_constructor_small_int(self, any_int_dtype):
        # see gh-13721
        exp = DatetimeIndex(['1970-01-01 00:00:00.00000000',
                             '1970-01-01 00:00:00.00000001',
                             '1970-01-01 00:00:00.00000002'])

        arr = np.array([0, 10, 20], dtype=any_int_dtype)
        tm.assert_index_equal(DatetimeIndex(arr), exp)

    def test_ctor_str_intraday(self):
        rng = DatetimeIndex(['1-1-2000 00:00:01'])
        assert rng[0].second == 1

    def test_is_(self):
        dti = date_range(start='1/1/2005', end='12/1/2005', freq='M')
        assert dti.is_(dti)
        assert dti.is_(dti.view())
        assert not dti.is_(dti.copy())

    def test_index_cast_datetime64_other_units(self):
        arr = np.arange(0, 100, 10, dtype=np.int64).view('M8[D]')
        idx = Index(arr)

        assert (idx.values == conversion.ensure_datetime64ns(arr)).all()

    def test_constructor_int64_nocopy(self):
        # GH#1624
        arr = np.arange(1000, dtype=np.int64)
        index = DatetimeIndex(arr)

        arr[50:100] = -1
        assert (index.asi8[50:100] == -1).all()

        arr = np.arange(1000, dtype=np.int64)
        index = DatetimeIndex(arr, copy=True)

        arr[50:100] = -1
        assert (index.asi8[50:100] != -1).all()

    @pytest.mark.parametrize('freq', ['M', 'Q', 'A', 'D', 'B', 'BH',
                                      'T', 'S', 'L', 'U', 'H', 'N', 'C'])
    def test_from_freq_recreate_from_data(self, freq):
        org = date_range(start='2001/02/01 09:00', freq=freq, periods=1)
        idx = DatetimeIndex(org, freq=freq)
        tm.assert_index_equal(idx, org)

        org = date_range(start='2001/02/01 09:00', freq=freq,
                         tz='US/Pacific', periods=1)
        idx = DatetimeIndex(org, freq=freq, tz='US/Pacific')
        tm.assert_index_equal(idx, org)

    def test_datetimeindex_constructor_misc(self):
        arr = ['1/1/2005', '1/2/2005', 'Jn 3, 2005', '2005-01-04']
        pytest.raises(Exception, DatetimeIndex, arr)

        arr = ['1/1/2005', '1/2/2005', '1/3/2005', '2005-01-04']
        idx1 = DatetimeIndex(arr)

        arr = [datetime(2005, 1, 1), '1/2/2005', '1/3/2005', '2005-01-04']
        idx2 = DatetimeIndex(arr)

        arr = [Timestamp(datetime(2005, 1, 1)), '1/2/2005', '1/3/2005',
               '2005-01-04']
        idx3 = DatetimeIndex(arr)

        arr = np.array(['1/1/2005', '1/2/2005', '1/3/2005',
                        '2005-01-04'], dtype='O')
        idx4 = DatetimeIndex(arr)

        arr = to_datetime(['1/1/2005', '1/2/2005', '1/3/2005', '2005-01-04'])
        idx5 = DatetimeIndex(arr)

        arr = to_datetime(['1/1/2005', '1/2/2005', 'Jan 3, 2005', '2005-01-04'
                           ])
        idx6 = DatetimeIndex(arr)

        idx7 = DatetimeIndex(['12/05/2007', '25/01/2008'], dayfirst=True)
        idx8 = DatetimeIndex(['2007/05/12', '2008/01/25'], dayfirst=False,
                             yearfirst=True)
        tm.assert_index_equal(idx7, idx8)

        for other in [idx2, idx3, idx4, idx5, idx6]:
            assert (idx1.values == other.values).all()

        sdate = datetime(1999, 12, 25)
        edate = datetime(2000, 1, 1)
        idx = date_range(start=sdate, freq='1B', periods=20)
        assert len(idx) == 20
        assert idx[0] == sdate + 0 * offsets.BDay()
        assert idx.freq == 'B'

        idx = date_range(end=edate, freq=('D', 5), periods=20)
        assert len(idx) == 20
        assert idx[-1] == edate
        assert idx.freq == '5D'

        idx1 = date_range(start=sdate, end=edate, freq='W-SUN')
        idx2 = date_range(start=sdate, end=edate,
                          freq=offsets.Week(weekday=6))
        assert len(idx1) == len(idx2)
        assert idx1.freq == idx2.freq

        idx1 = date_range(start=sdate, end=edate, freq='QS')
        idx2 = date_range(start=sdate, end=edate,
                          freq=offsets.QuarterBegin(startingMonth=1))
        assert len(idx1) == len(idx2)
        assert idx1.freq == idx2.freq

        idx1 = date_range(start=sdate, end=edate, freq='BQ')
        idx2 = date_range(start=sdate, end=edate,
                          freq=offsets.BQuarterEnd(startingMonth=12))
        assert len(idx1) == len(idx2)
        assert idx1.freq == idx2.freq
