import numpy as np
import pytest

import pandas as pd
from pandas import Index, PeriodIndex, date_range, period_range
import pandas.core.indexes.period as period
import pandas.util.testing as tm


def _permute(obj):
    return obj.take(np.random.permutation(len(obj)))


class TestPeriodIndex(object):

    def test_joins(self, join_type):
        index = period_range('1/1/2000', '1/20/2000', freq='D')

        joined = index.join(index[:-5], how=join_type)

        assert isinstance(joined, PeriodIndex)
        assert joined.freq == index.freq

    def test_join_self(self, join_type):
        index = period_range('1/1/2000', '1/20/2000', freq='D')

        res = index.join(index, how=join_type)
        assert index is res

    def test_join_does_not_recur(self):
        df = tm.makeCustomDataframe(
            3, 2, data_gen_f=lambda *args: np.random.randint(2),
            c_idx_type='p', r_idx_type='dt')
        s = df.iloc[:2, 0]

        res = s.index.join(df.columns, how='outer')
        expected = Index([s.index[0], s.index[1],
                          df.columns[0], df.columns[1]], object)
        tm.assert_index_equal(res, expected)

    @pytest.mark.parametrize("sort", [None, False])
    def test_union(self, sort):
        # union
        other1 = pd.period_range('1/1/2000', freq='D', periods=5)
        rng1 = pd.period_range('1/6/2000', freq='D', periods=5)
        expected1 = pd.period_range('1/1/2000', freq='D', periods=10)

        rng2 = pd.period_range('1/1/2000', freq='D', periods=5)
        other2 = pd.period_range('1/4/2000', freq='D', periods=5)
        expected2 = pd.period_range('1/1/2000', freq='D', periods=8)

        rng3 = pd.period_range('1/1/2000', freq='D', periods=5)
        other3 = pd.PeriodIndex([], freq='D')
        expected3 = pd.period_range('1/1/2000', freq='D', periods=5)

        rng4 = pd.period_range('2000-01-01 09:00', freq='H', periods=5)
        other4 = pd.period_range('2000-01-02 09:00', freq='H', periods=5)
        expected4 = pd.PeriodIndex(['2000-01-01 09:00', '2000-01-01 10:00',
                                    '2000-01-01 11:00', '2000-01-01 12:00',
                                    '2000-01-01 13:00', '2000-01-02 09:00',
                                    '2000-01-02 10:00', '2000-01-02 11:00',
                                    '2000-01-02 12:00', '2000-01-02 13:00'],
                                   freq='H')

        rng5 = pd.PeriodIndex(['2000-01-01 09:01', '2000-01-01 09:03',
                               '2000-01-01 09:05'], freq='T')
        other5 = pd.PeriodIndex(['2000-01-01 09:01', '2000-01-01 09:05'
                                                     '2000-01-01 09:08'],
                                freq='T')
        expected5 = pd.PeriodIndex(['2000-01-01 09:01', '2000-01-01 09:03',
                                    '2000-01-01 09:05', '2000-01-01 09:08'],
                                   freq='T')

        rng6 = pd.period_range('2000-01-01', freq='M', periods=7)
        other6 = pd.period_range('2000-04-01', freq='M', periods=7)
        expected6 = pd.period_range('2000-01-01', freq='M', periods=10)

        rng7 = pd.period_range('2003-01-01', freq='A', periods=5)
        other7 = pd.period_range('1998-01-01', freq='A', periods=8)
        expected7 = pd.period_range('1998-01-01', freq='A', periods=10)

        rng8 = pd.PeriodIndex(['1/3/2000', '1/2/2000', '1/1/2000',
                               '1/5/2000', '1/4/2000'], freq='D')
        other8 = pd.period_range('1/6/2000', freq='D', periods=5)
        expected8 = pd.PeriodIndex(['1/3/2000', '1/2/2000', '1/1/2000',
                                    '1/5/2000', '1/4/2000', '1/6/2000',
                                    '1/7/2000', '1/8/2000', '1/9/2000',
                                    '1/10/2000'], freq='D')

        for rng, other, expected in [(rng1, other1, expected1),
                                     (rng2, other2, expected2),
                                     (rng3, other3, expected3),
                                     (rng4, other4, expected4),
                                     (rng5, other5, expected5),
                                     (rng6, other6, expected6),
                                     (rng7, other7, expected7),
                                     (rng8, other8, expected8)]:

            result_union = rng.union(other, sort=sort)
            if sort is None:
                expected = expected.sort_values()
            tm.assert_index_equal(result_union, expected)

    @pytest.mark.parametrize("sort", [None, False])
    def test_union_misc(self, sort):
        index = period_range('1/1/2000', '1/20/2000', freq='D')

        result = index[:-5].union(index[10:], sort=sort)
        tm.assert_index_equal(result, index)

        # not in order
        result = _permute(index[:-5]).union(_permute(index[10:]), sort=sort)
        if sort is None:
            tm.assert_index_equal(result, index)
        assert tm.equalContents(result, index)

        # raise if different frequencies
        index = period_range('1/1/2000', '1/20/2000', freq='D')
        index2 = period_range('1/1/2000', '1/20/2000', freq='W-WED')
        with pytest.raises(period.IncompatibleFrequency):
            index.union(index2, sort=sort)

        msg = 'can only call with other PeriodIndex-ed objects'
        with pytest.raises(ValueError, match=msg):
            index.join(index.to_timestamp())

        index3 = period_range('1/1/2000', '1/20/2000', freq='2D')
        with pytest.raises(period.IncompatibleFrequency):
            index.join(index3)

    def test_union_dataframe_index(self):
        rng1 = pd.period_range('1/1/1999', '1/1/2012', freq='M')
        s1 = pd.Series(np.random.randn(len(rng1)), rng1)

        rng2 = pd.period_range('1/1/1980', '12/1/2001', freq='M')
        s2 = pd.Series(np.random.randn(len(rng2)), rng2)
        df = pd.DataFrame({'s1': s1, 's2': s2})

        exp = pd.period_range('1/1/1980', '1/1/2012', freq='M')
        tm.assert_index_equal(df.index, exp)

    @pytest.mark.parametrize("sort", [None, False])
    def test_intersection(self, sort):
        index = period_range('1/1/2000', '1/20/2000', freq='D')

        result = index[:-5].intersection(index[10:], sort=sort)
        tm.assert_index_equal(result, index[10:-5])

        # not in order
        left = _permute(index[:-5])
        right = _permute(index[10:])
        result = left.intersection(right, sort=sort)
        if sort is None:
            tm.assert_index_equal(result, index[10:-5])
        assert tm.equalContents(result, index[10:-5])

        # raise if different frequencies
        index = period_range('1/1/2000', '1/20/2000', freq='D')
        index2 = period_range('1/1/2000', '1/20/2000', freq='W-WED')
        with pytest.raises(period.IncompatibleFrequency):
            index.intersection(index2, sort=sort)

        index3 = period_range('1/1/2000', '1/20/2000', freq='2D')
        with pytest.raises(period.IncompatibleFrequency):
            index.intersection(index3, sort=sort)

    @pytest.mark.parametrize("sort", [None, False])
    def test_intersection_cases(self, sort):
        base = period_range('6/1/2000', '6/30/2000', freq='D', name='idx')

        # if target has the same name, it is preserved
        rng2 = period_range('5/15/2000', '6/20/2000', freq='D', name='idx')
        expected2 = period_range('6/1/2000', '6/20/2000', freq='D',
                                 name='idx')

        # if target name is different, it will be reset
        rng3 = period_range('5/15/2000', '6/20/2000', freq='D', name='other')
        expected3 = period_range('6/1/2000', '6/20/2000', freq='D',
                                 name=None)

        rng4 = period_range('7/1/2000', '7/31/2000', freq='D', name='idx')
        expected4 = PeriodIndex([], name='idx', freq='D')

        for (rng, expected) in [(rng2, expected2), (rng3, expected3),
                                (rng4, expected4)]:
            result = base.intersection(rng, sort=sort)
            tm.assert_index_equal(result, expected)
            assert result.name == expected.name
            assert result.freq == expected.freq

        # non-monotonic
        base = PeriodIndex(['2011-01-05', '2011-01-04', '2011-01-02',
                            '2011-01-03'], freq='D', name='idx')

        rng2 = PeriodIndex(['2011-01-04', '2011-01-02',
                            '2011-02-02', '2011-02-03'],
                           freq='D', name='idx')
        expected2 = PeriodIndex(['2011-01-04', '2011-01-02'], freq='D',
                                name='idx')

        rng3 = PeriodIndex(['2011-01-04', '2011-01-02', '2011-02-02',
                            '2011-02-03'],
                           freq='D', name='other')
        expected3 = PeriodIndex(['2011-01-04', '2011-01-02'], freq='D',
                                name=None)

        rng4 = period_range('7/1/2000', '7/31/2000', freq='D', name='idx')
        expected4 = PeriodIndex([], freq='D', name='idx')

        for (rng, expected) in [(rng2, expected2), (rng3, expected3),
                                (rng4, expected4)]:
            result = base.intersection(rng, sort=sort)
            if sort is None:
                expected = expected.sort_values()
            tm.assert_index_equal(result, expected)
            assert result.name == expected.name
            assert result.freq == 'D'

        # empty same freq
        rng = date_range('6/1/2000', '6/15/2000', freq='T')
        result = rng[0:0].intersection(rng)
        assert len(result) == 0

        result = rng.intersection(rng[0:0])
        assert len(result) == 0

    @pytest.mark.parametrize("sort", [None, False])
    def test_difference(self, sort):
        # diff
        period_rng = ['1/3/2000', '1/2/2000', '1/1/2000', '1/5/2000',
                      '1/4/2000']
        rng1 = pd.PeriodIndex(period_rng, freq='D')
        other1 = pd.period_range('1/6/2000', freq='D', periods=5)
        expected1 = rng1

        rng2 = pd.PeriodIndex(period_rng, freq='D')
        other2 = pd.period_range('1/4/2000', freq='D', periods=5)
        expected2 = pd.PeriodIndex(['1/3/2000', '1/2/2000', '1/1/2000'],
                                   freq='D')

        rng3 = pd.PeriodIndex(period_rng, freq='D')
        other3 = pd.PeriodIndex([], freq='D')
        expected3 = rng3

        period_rng = ['2000-01-01 10:00', '2000-01-01 09:00',
                      '2000-01-01 12:00', '2000-01-01 11:00',
                      '2000-01-01 13:00']
        rng4 = pd.PeriodIndex(period_rng, freq='H')
        other4 = pd.period_range('2000-01-02 09:00', freq='H', periods=5)
        expected4 = rng4

        rng5 = pd.PeriodIndex(['2000-01-01 09:03', '2000-01-01 09:01',
                               '2000-01-01 09:05'], freq='T')
        other5 = pd.PeriodIndex(
            ['2000-01-01 09:01', '2000-01-01 09:05'], freq='T')
        expected5 = pd.PeriodIndex(['2000-01-01 09:03'], freq='T')

        period_rng = ['2000-02-01', '2000-01-01', '2000-06-01',
                      '2000-07-01', '2000-05-01', '2000-03-01',
                      '2000-04-01']
        rng6 = pd.PeriodIndex(period_rng, freq='M')
        other6 = pd.period_range('2000-04-01', freq='M', periods=7)
        expected6 = pd.PeriodIndex(['2000-02-01', '2000-01-01', '2000-03-01'],
                                   freq='M')

        period_rng = ['2003', '2007', '2006', '2005', '2004']
        rng7 = pd.PeriodIndex(period_rng, freq='A')
        other7 = pd.period_range('1998-01-01', freq='A', periods=8)
        expected7 = pd.PeriodIndex(['2007', '2006'], freq='A')

        for rng, other, expected in [(rng1, other1, expected1),
                                     (rng2, other2, expected2),
                                     (rng3, other3, expected3),
                                     (rng4, other4, expected4),
                                     (rng5, other5, expected5),
                                     (rng6, other6, expected6),
                                     (rng7, other7, expected7), ]:
            result_difference = rng.difference(other, sort=sort)
            if sort is None:
                expected = expected.sort_values()
            tm.assert_index_equal(result_difference, expected)
