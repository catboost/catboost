# -*- coding: utf-8 -*-
# pylint: disable=E1101

from collections import OrderedDict
import datetime as dt
from datetime import datetime
import gzip
import io
import os
import struct
import warnings

import numpy as np
import pytest

import pandas.compat as compat
from pandas.compat import PY3, ResourceWarning, iterkeys

from pandas.core.dtypes.common import is_categorical_dtype

import pandas as pd
from pandas.core.frame import DataFrame, Series
import pandas.util.testing as tm

from pandas.io.parsers import read_csv
from pandas.io.stata import (
    InvalidColumnName, PossiblePrecisionLoss, StataMissingValue, StataReader,
    read_stata)


@pytest.fixture
def dirpath(datapath):
    return datapath("io", "data")


@pytest.fixture
def parsed_114(dirpath):
    dta14_114 = os.path.join(dirpath, 'stata5_114.dta')
    parsed_114 = read_stata(dta14_114, convert_dates=True)
    parsed_114.index.name = 'index'
    return parsed_114


class TestStata(object):

    @pytest.fixture(autouse=True)
    def setup_method(self, datapath):
        self.dirpath = datapath("io", "data")
        self.dta1_114 = os.path.join(self.dirpath, 'stata1_114.dta')
        self.dta1_117 = os.path.join(self.dirpath, 'stata1_117.dta')

        self.dta2_113 = os.path.join(self.dirpath, 'stata2_113.dta')
        self.dta2_114 = os.path.join(self.dirpath, 'stata2_114.dta')
        self.dta2_115 = os.path.join(self.dirpath, 'stata2_115.dta')
        self.dta2_117 = os.path.join(self.dirpath, 'stata2_117.dta')

        self.dta3_113 = os.path.join(self.dirpath, 'stata3_113.dta')
        self.dta3_114 = os.path.join(self.dirpath, 'stata3_114.dta')
        self.dta3_115 = os.path.join(self.dirpath, 'stata3_115.dta')
        self.dta3_117 = os.path.join(self.dirpath, 'stata3_117.dta')
        self.csv3 = os.path.join(self.dirpath, 'stata3.csv')

        self.dta4_113 = os.path.join(self.dirpath, 'stata4_113.dta')
        self.dta4_114 = os.path.join(self.dirpath, 'stata4_114.dta')
        self.dta4_115 = os.path.join(self.dirpath, 'stata4_115.dta')
        self.dta4_117 = os.path.join(self.dirpath, 'stata4_117.dta')

        self.dta_encoding = os.path.join(self.dirpath, 'stata1_encoding.dta')

        self.csv14 = os.path.join(self.dirpath, 'stata5.csv')
        self.dta14_113 = os.path.join(self.dirpath, 'stata5_113.dta')
        self.dta14_114 = os.path.join(self.dirpath, 'stata5_114.dta')
        self.dta14_115 = os.path.join(self.dirpath, 'stata5_115.dta')
        self.dta14_117 = os.path.join(self.dirpath, 'stata5_117.dta')

        self.csv15 = os.path.join(self.dirpath, 'stata6.csv')
        self.dta15_113 = os.path.join(self.dirpath, 'stata6_113.dta')
        self.dta15_114 = os.path.join(self.dirpath, 'stata6_114.dta')
        self.dta15_115 = os.path.join(self.dirpath, 'stata6_115.dta')
        self.dta15_117 = os.path.join(self.dirpath, 'stata6_117.dta')

        self.dta16_115 = os.path.join(self.dirpath, 'stata7_115.dta')
        self.dta16_117 = os.path.join(self.dirpath, 'stata7_117.dta')

        self.dta17_113 = os.path.join(self.dirpath, 'stata8_113.dta')
        self.dta17_115 = os.path.join(self.dirpath, 'stata8_115.dta')
        self.dta17_117 = os.path.join(self.dirpath, 'stata8_117.dta')

        self.dta18_115 = os.path.join(self.dirpath, 'stata9_115.dta')
        self.dta18_117 = os.path.join(self.dirpath, 'stata9_117.dta')

        self.dta19_115 = os.path.join(self.dirpath, 'stata10_115.dta')
        self.dta19_117 = os.path.join(self.dirpath, 'stata10_117.dta')

        self.dta20_115 = os.path.join(self.dirpath, 'stata11_115.dta')
        self.dta20_117 = os.path.join(self.dirpath, 'stata11_117.dta')

        self.dta21_117 = os.path.join(self.dirpath, 'stata12_117.dta')

        self.dta22_118 = os.path.join(self.dirpath, 'stata14_118.dta')
        self.dta23 = os.path.join(self.dirpath, 'stata15.dta')

        self.dta24_111 = os.path.join(self.dirpath, 'stata7_111.dta')
        self.dta25_118 = os.path.join(self.dirpath, 'stata16_118.dta')

        self.stata_dates = os.path.join(self.dirpath, 'stata13_dates.dta')

    def read_dta(self, file):
        # Legacy default reader configuration
        return read_stata(file, convert_dates=True)

    def read_csv(self, file):
        return read_csv(file, parse_dates=True)

    @pytest.mark.parametrize('version', [114, 117])
    def test_read_empty_dta(self, version):
        empty_ds = DataFrame(columns=['unit'])
        # GH 7369, make sure can read a 0-obs dta file
        with tm.ensure_clean() as path:
            empty_ds.to_stata(path, write_index=False, version=version)
            empty_ds2 = read_stata(path)
            tm.assert_frame_equal(empty_ds, empty_ds2)

    def test_data_method(self):
        # Minimal testing of legacy data method
        with StataReader(self.dta1_114) as rdr:
            with tm.assert_produces_warning(UserWarning):
                parsed_114_data = rdr.data()

        with StataReader(self.dta1_114) as rdr:
            parsed_114_read = rdr.read()
        tm.assert_frame_equal(parsed_114_data, parsed_114_read)

    @pytest.mark.parametrize(
        'file', ['dta1_114', 'dta1_117'])
    def test_read_dta1(self, file):

        file = getattr(self, file)
        parsed = self.read_dta(file)

        # Pandas uses np.nan as missing value.
        # Thus, all columns will be of type float, regardless of their name.
        expected = DataFrame([(np.nan, np.nan, np.nan, np.nan, np.nan)],
                             columns=['float_miss', 'double_miss', 'byte_miss',
                                      'int_miss', 'long_miss'])

        # this is an oddity as really the nan should be float64, but
        # the casting doesn't fail so need to match stata here
        expected['float_miss'] = expected['float_miss'].astype(np.float32)

        tm.assert_frame_equal(parsed, expected)

    def test_read_dta2(self):

        expected = DataFrame.from_records(
            [
                (
                    datetime(2006, 11, 19, 23, 13, 20),
                    1479596223000,
                    datetime(2010, 1, 20),
                    datetime(2010, 1, 8),
                    datetime(2010, 1, 1),
                    datetime(1974, 7, 1),
                    datetime(2010, 1, 1),
                    datetime(2010, 1, 1)
                ),
                (
                    datetime(1959, 12, 31, 20, 3, 20),
                    -1479590,
                    datetime(1953, 10, 2),
                    datetime(1948, 6, 10),
                    datetime(1955, 1, 1),
                    datetime(1955, 7, 1),
                    datetime(1955, 1, 1),
                    datetime(2, 1, 1)
                ),
                (
                    pd.NaT,
                    pd.NaT,
                    pd.NaT,
                    pd.NaT,
                    pd.NaT,
                    pd.NaT,
                    pd.NaT,
                    pd.NaT,
                )
            ],
            columns=['datetime_c', 'datetime_big_c', 'date', 'weekly_date',
                     'monthly_date', 'quarterly_date', 'half_yearly_date',
                     'yearly_date']
        )
        expected['yearly_date'] = expected['yearly_date'].astype('O')

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            parsed_114 = self.read_dta(self.dta2_114)
            parsed_115 = self.read_dta(self.dta2_115)
            parsed_117 = self.read_dta(self.dta2_117)
            # 113 is buggy due to limits of date format support in Stata
            # parsed_113 = self.read_dta(self.dta2_113)

            # Remove resource warnings
            w = [x for x in w if x.category is UserWarning]

            # should get warning for each call to read_dta
            assert len(w) == 3

        # buggy test because of the NaT comparison on certain platforms
        # Format 113 test fails since it does not support tc and tC formats
        # tm.assert_frame_equal(parsed_113, expected)
        tm.assert_frame_equal(parsed_114, expected,
                              check_datetimelike_compat=True)
        tm.assert_frame_equal(parsed_115, expected,
                              check_datetimelike_compat=True)
        tm.assert_frame_equal(parsed_117, expected,
                              check_datetimelike_compat=True)

    @pytest.mark.parametrize(
        'file', ['dta3_113', 'dta3_114', 'dta3_115', 'dta3_117'])
    def test_read_dta3(self, file):

        file = getattr(self, file)
        parsed = self.read_dta(file)

        # match stata here
        expected = self.read_csv(self.csv3)
        expected = expected.astype(np.float32)
        expected['year'] = expected['year'].astype(np.int16)
        expected['quarter'] = expected['quarter'].astype(np.int8)

        tm.assert_frame_equal(parsed, expected)

    @pytest.mark.parametrize(
        'file', ['dta4_113', 'dta4_114', 'dta4_115', 'dta4_117'])
    def test_read_dta4(self, file):

        file = getattr(self, file)
        parsed = self.read_dta(file)

        expected = DataFrame.from_records(
            [
                ["one", "ten", "one", "one", "one"],
                ["two", "nine", "two", "two", "two"],
                ["three", "eight", "three", "three", "three"],
                ["four", "seven", 4, "four", "four"],
                ["five", "six", 5, np.nan, "five"],
                ["six", "five", 6, np.nan, "six"],
                ["seven", "four", 7, np.nan, "seven"],
                ["eight", "three", 8, np.nan, "eight"],
                ["nine", "two", 9, np.nan, "nine"],
                ["ten", "one", "ten", np.nan, "ten"]
            ],
            columns=['fully_labeled', 'fully_labeled2', 'incompletely_labeled',
                     'labeled_with_missings', 'float_labelled'])

        # these are all categoricals
        expected = pd.concat([expected[col].astype('category')
                              for col in expected], axis=1)

        # stata doesn't save .category metadata
        tm.assert_frame_equal(parsed, expected, check_categorical=False)

    # File containing strls
    def test_read_dta12(self):
        parsed_117 = self.read_dta(self.dta21_117)
        expected = DataFrame.from_records(
            [
                [1, "abc", "abcdefghi"],
                [3, "cba", "qwertywertyqwerty"],
                [93, "", "strl"],
            ],
            columns=['x', 'y', 'z'])

        tm.assert_frame_equal(parsed_117, expected, check_dtype=False)

    def test_read_dta18(self):
        parsed_118 = self.read_dta(self.dta22_118)
        parsed_118["Bytes"] = parsed_118["Bytes"].astype('O')
        expected = DataFrame.from_records(
            [['Cat', 'Bogota', u'Bogotá', 1, 1.0, u'option b Ünicode', 1.0],
             ['Dog', 'Boston', u'Uzunköprü', np.nan, np.nan, np.nan, np.nan],
             ['Plane', 'Rome', u'Tromsø', 0, 0.0, 'option a', 0.0],
             ['Potato', 'Tokyo', u'Elâzığ', -4, 4.0, 4, 4],
             ['', '', '', 0, 0.3332999, 'option a', 1 / 3.]
             ],
            columns=['Things', 'Cities', 'Unicode_Cities_Strl',
                     'Ints', 'Floats', 'Bytes', 'Longs'])
        expected["Floats"] = expected["Floats"].astype(np.float32)
        for col in parsed_118.columns:
            tm.assert_almost_equal(parsed_118[col], expected[col])

        with StataReader(self.dta22_118) as rdr:
            vl = rdr.variable_labels()
            vl_expected = {u'Unicode_Cities_Strl':
                           u'Here are some strls with Ünicode chars',
                           u'Longs': u'long data',
                           u'Things': u'Here are some things',
                           u'Bytes': u'byte data',
                           u'Ints': u'int data',
                           u'Cities': u'Here are some cities',
                           u'Floats': u'float data'}
            tm.assert_dict_equal(vl, vl_expected)

            assert rdr.data_label == u'This is a  Ünicode data label'

    def test_read_write_dta5(self):
        original = DataFrame([(np.nan, np.nan, np.nan, np.nan, np.nan)],
                             columns=['float_miss', 'double_miss', 'byte_miss',
                                      'int_miss', 'long_miss'])
        original.index.name = 'index'

        with tm.ensure_clean() as path:
            original.to_stata(path, None)
            written_and_read_again = self.read_dta(path)
            tm.assert_frame_equal(written_and_read_again.set_index('index'),
                                  original)

    def test_write_dta6(self):
        original = self.read_csv(self.csv3)
        original.index.name = 'index'
        original.index = original.index.astype(np.int32)
        original['year'] = original['year'].astype(np.int32)
        original['quarter'] = original['quarter'].astype(np.int32)

        with tm.ensure_clean() as path:
            original.to_stata(path, None)
            written_and_read_again = self.read_dta(path)
            tm.assert_frame_equal(written_and_read_again.set_index('index'),
                                  original, check_index_type=False)

    @pytest.mark.parametrize('version', [114, 117])
    def test_read_write_dta10(self, version):
        original = DataFrame(data=[["string", "object", 1, 1.1,
                                    np.datetime64('2003-12-25')]],
                             columns=['string', 'object', 'integer',
                                      'floating', 'datetime'])
        original["object"] = Series(original["object"], dtype=object)
        original.index.name = 'index'
        original.index = original.index.astype(np.int32)
        original['integer'] = original['integer'].astype(np.int32)

        with tm.ensure_clean() as path:
            original.to_stata(path, {'datetime': 'tc'}, version=version)
            written_and_read_again = self.read_dta(path)
            # original.index is np.int32, read index is np.int64
            tm.assert_frame_equal(written_and_read_again.set_index('index'),
                                  original, check_index_type=False)

    def test_stata_doc_examples(self):
        with tm.ensure_clean() as path:
            df = DataFrame(np.random.randn(10, 2), columns=list('AB'))
            df.to_stata(path)

    def test_write_preserves_original(self):
        # 9795
        np.random.seed(423)
        df = pd.DataFrame(np.random.randn(5, 4), columns=list('abcd'))
        df.loc[2, 'a':'c'] = np.nan
        df_copy = df.copy()
        with tm.ensure_clean() as path:
            df.to_stata(path, write_index=False)
        tm.assert_frame_equal(df, df_copy)

    @pytest.mark.parametrize('version', [114, 117])
    def test_encoding(self, version):

        # GH 4626, proper encoding handling
        raw = read_stata(self.dta_encoding)
        with tm.assert_produces_warning(FutureWarning):
            encoded = read_stata(self.dta_encoding, encoding='latin-1')
        result = encoded.kreis1849[0]

        expected = raw.kreis1849[0]
        assert result == expected
        assert isinstance(result, compat.string_types)

        with tm.ensure_clean() as path:
            with tm.assert_produces_warning(FutureWarning):
                encoded.to_stata(path, write_index=False, version=version,
                                 encoding='latin-1')
            reread_encoded = read_stata(path)
            tm.assert_frame_equal(encoded, reread_encoded)

    def test_read_write_dta11(self):
        original = DataFrame([(1, 2, 3, 4)],
                             columns=['good', compat.u('b\u00E4d'), '8number',
                                      'astringwithmorethan32characters______'])
        formatted = DataFrame([(1, 2, 3, 4)],
                              columns=['good', 'b_d', '_8number',
                                       'astringwithmorethan32characters_'])
        formatted.index.name = 'index'
        formatted = formatted.astype(np.int32)

        with tm.ensure_clean() as path:
            with tm.assert_produces_warning(pd.io.stata.InvalidColumnName):
                original.to_stata(path, None)

            written_and_read_again = self.read_dta(path)
            tm.assert_frame_equal(
                written_and_read_again.set_index('index'), formatted)

    @pytest.mark.parametrize('version', [114, 117])
    def test_read_write_dta12(self, version):
        original = DataFrame([(1, 2, 3, 4, 5, 6)],
                             columns=['astringwithmorethan32characters_1',
                                      'astringwithmorethan32characters_2',
                                      '+',
                                      '-',
                                      'short',
                                      'delete'])
        formatted = DataFrame([(1, 2, 3, 4, 5, 6)],
                              columns=['astringwithmorethan32characters_',
                                       '_0astringwithmorethan32character',
                                       '_',
                                       '_1_',
                                       '_short',
                                       '_delete'])
        formatted.index.name = 'index'
        formatted = formatted.astype(np.int32)

        with tm.ensure_clean() as path:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always', InvalidColumnName)
                original.to_stata(path, None, version=version)
                # should get a warning for that format.
                assert len(w) == 1

            written_and_read_again = self.read_dta(path)
            tm.assert_frame_equal(
                written_and_read_again.set_index('index'), formatted)

    def test_read_write_dta13(self):
        s1 = Series(2 ** 9, dtype=np.int16)
        s2 = Series(2 ** 17, dtype=np.int32)
        s3 = Series(2 ** 33, dtype=np.int64)
        original = DataFrame({'int16': s1, 'int32': s2, 'int64': s3})
        original.index.name = 'index'

        formatted = original
        formatted['int64'] = formatted['int64'].astype(np.float64)

        with tm.ensure_clean() as path:
            original.to_stata(path)
            written_and_read_again = self.read_dta(path)
            tm.assert_frame_equal(written_and_read_again.set_index('index'),
                                  formatted)

    @pytest.mark.parametrize('version', [114, 117])
    @pytest.mark.parametrize(
        'file', ['dta14_113', 'dta14_114', 'dta14_115', 'dta14_117'])
    def test_read_write_reread_dta14(self, file, parsed_114, version):
        file = getattr(self, file)
        parsed = self.read_dta(file)
        parsed.index.name = 'index'

        expected = self.read_csv(self.csv14)
        cols = ['byte_', 'int_', 'long_', 'float_', 'double_']
        for col in cols:
            expected[col] = expected[col]._convert(datetime=True, numeric=True)
        expected['float_'] = expected['float_'].astype(np.float32)
        expected['date_td'] = pd.to_datetime(
            expected['date_td'], errors='coerce')

        tm.assert_frame_equal(parsed_114, parsed)

        with tm.ensure_clean() as path:
            parsed_114.to_stata(path, {'date_td': 'td'}, version=version)
            written_and_read_again = self.read_dta(path)
            tm.assert_frame_equal(
                written_and_read_again.set_index('index'), parsed_114)

    @pytest.mark.parametrize(
        'file', ['dta15_113', 'dta15_114', 'dta15_115', 'dta15_117'])
    def test_read_write_reread_dta15(self, file):

        expected = self.read_csv(self.csv15)
        expected['byte_'] = expected['byte_'].astype(np.int8)
        expected['int_'] = expected['int_'].astype(np.int16)
        expected['long_'] = expected['long_'].astype(np.int32)
        expected['float_'] = expected['float_'].astype(np.float32)
        expected['double_'] = expected['double_'].astype(np.float64)
        expected['date_td'] = expected['date_td'].apply(
            datetime.strptime, args=('%Y-%m-%d',))

        file = getattr(self, file)
        parsed = self.read_dta(file)

        tm.assert_frame_equal(expected, parsed)

    @pytest.mark.parametrize('version', [114, 117])
    def test_timestamp_and_label(self, version):
        original = DataFrame([(1,)], columns=['variable'])
        time_stamp = datetime(2000, 2, 29, 14, 21)
        data_label = 'This is a data file.'
        with tm.ensure_clean() as path:
            original.to_stata(path, time_stamp=time_stamp,
                              data_label=data_label,
                              version=version)

            with StataReader(path) as reader:
                assert reader.time_stamp == '29 Feb 2000 14:21'
                assert reader.data_label == data_label

    @pytest.mark.parametrize('version', [114, 117])
    def test_invalid_timestamp(self, version):
        original = DataFrame([(1,)], columns=['variable'])
        time_stamp = '01 Jan 2000, 00:00:00'
        with tm.ensure_clean() as path:
            msg = "time_stamp should be datetime type"
            with pytest.raises(ValueError, match=msg):
                original.to_stata(path, time_stamp=time_stamp,
                                  version=version)

    def test_numeric_column_names(self):
        original = DataFrame(np.reshape(np.arange(25.0), (5, 5)))
        original.index.name = 'index'
        with tm.ensure_clean() as path:
            # should get a warning for that format.
            with tm.assert_produces_warning(InvalidColumnName):
                original.to_stata(path)

            written_and_read_again = self.read_dta(path)
            written_and_read_again = written_and_read_again.set_index('index')
            columns = list(written_and_read_again.columns)
            convert_col_name = lambda x: int(x[1])
            written_and_read_again.columns = map(convert_col_name, columns)
            tm.assert_frame_equal(original, written_and_read_again)

    @pytest.mark.parametrize('version', [114, 117])
    def test_nan_to_missing_value(self, version):
        s1 = Series(np.arange(4.0), dtype=np.float32)
        s2 = Series(np.arange(4.0), dtype=np.float64)
        s1[::2] = np.nan
        s2[1::2] = np.nan
        original = DataFrame({'s1': s1, 's2': s2})
        original.index.name = 'index'
        with tm.ensure_clean() as path:
            original.to_stata(path, version=version)
            written_and_read_again = self.read_dta(path)
            written_and_read_again = written_and_read_again.set_index('index')
            tm.assert_frame_equal(written_and_read_again, original)

    def test_no_index(self):
        columns = ['x', 'y']
        original = DataFrame(np.reshape(np.arange(10.0), (5, 2)),
                             columns=columns)
        original.index.name = 'index_not_written'
        with tm.ensure_clean() as path:
            original.to_stata(path, write_index=False)
            written_and_read_again = self.read_dta(path)
            with pytest.raises(KeyError, match=original.index.name):
                written_and_read_again['index_not_written']

    def test_string_no_dates(self):
        s1 = Series(['a', 'A longer string'])
        s2 = Series([1.0, 2.0], dtype=np.float64)
        original = DataFrame({'s1': s1, 's2': s2})
        original.index.name = 'index'
        with tm.ensure_clean() as path:
            original.to_stata(path)
            written_and_read_again = self.read_dta(path)
            tm.assert_frame_equal(written_and_read_again.set_index('index'),
                                  original)

    def test_large_value_conversion(self):
        s0 = Series([1, 99], dtype=np.int8)
        s1 = Series([1, 127], dtype=np.int8)
        s2 = Series([1, 2 ** 15 - 1], dtype=np.int16)
        s3 = Series([1, 2 ** 63 - 1], dtype=np.int64)
        original = DataFrame({'s0': s0, 's1': s1, 's2': s2, 's3': s3})
        original.index.name = 'index'
        with tm.ensure_clean() as path:
            with tm.assert_produces_warning(PossiblePrecisionLoss):
                original.to_stata(path)

            written_and_read_again = self.read_dta(path)
            modified = original.copy()
            modified['s1'] = Series(modified['s1'], dtype=np.int16)
            modified['s2'] = Series(modified['s2'], dtype=np.int32)
            modified['s3'] = Series(modified['s3'], dtype=np.float64)
            tm.assert_frame_equal(written_and_read_again.set_index('index'),
                                  modified)

    def test_dates_invalid_column(self):
        original = DataFrame([datetime(2006, 11, 19, 23, 13, 20)])
        original.index.name = 'index'
        with tm.ensure_clean() as path:
            with tm.assert_produces_warning(InvalidColumnName):
                original.to_stata(path, {0: 'tc'})

            written_and_read_again = self.read_dta(path)
            modified = original.copy()
            modified.columns = ['_0']
            tm.assert_frame_equal(written_and_read_again.set_index('index'),
                                  modified)

    def test_105(self):
        # Data obtained from:
        # http://go.worldbank.org/ZXY29PVJ21
        dpath = os.path.join(self.dirpath, 'S4_EDUC1.dta')
        df = pd.read_stata(dpath)
        df0 = [[1, 1, 3, -2], [2, 1, 2, -2], [4, 1, 1, -2]]
        df0 = pd.DataFrame(df0)
        df0.columns = ["clustnum", "pri_schl", "psch_num", "psch_dis"]
        df0['clustnum'] = df0["clustnum"].astype(np.int16)
        df0['pri_schl'] = df0["pri_schl"].astype(np.int8)
        df0['psch_num'] = df0["psch_num"].astype(np.int8)
        df0['psch_dis'] = df0["psch_dis"].astype(np.float32)
        tm.assert_frame_equal(df.head(3), df0)

    def test_value_labels_old_format(self):
        # GH 19417
        #
        # Test that value_labels() returns an empty dict if the file format
        # predates supporting value labels.
        dpath = os.path.join(self.dirpath, 'S4_EDUC1.dta')
        reader = StataReader(dpath)
        assert reader.value_labels() == {}
        reader.close()

    def test_date_export_formats(self):
        columns = ['tc', 'td', 'tw', 'tm', 'tq', 'th', 'ty']
        conversions = {c: c for c in columns}
        data = [datetime(2006, 11, 20, 23, 13, 20)] * len(columns)
        original = DataFrame([data], columns=columns)
        original.index.name = 'index'
        expected_values = [datetime(2006, 11, 20, 23, 13, 20),  # Time
                           datetime(2006, 11, 20),  # Day
                           datetime(2006, 11, 19),  # Week
                           datetime(2006, 11, 1),  # Month
                           datetime(2006, 10, 1),  # Quarter year
                           datetime(2006, 7, 1),  # Half year
                           datetime(2006, 1, 1)]  # Year

        expected = DataFrame([expected_values], columns=columns)
        expected.index.name = 'index'
        with tm.ensure_clean() as path:
            original.to_stata(path, conversions)
            written_and_read_again = self.read_dta(path)
            tm.assert_frame_equal(written_and_read_again.set_index('index'),
                                  expected)

    def test_write_missing_strings(self):
        original = DataFrame([["1"], [None]], columns=["foo"])
        expected = DataFrame([["1"], [""]], columns=["foo"])
        expected.index.name = 'index'
        with tm.ensure_clean() as path:
            original.to_stata(path)
            written_and_read_again = self.read_dta(path)
            tm.assert_frame_equal(written_and_read_again.set_index('index'),
                                  expected)

    @pytest.mark.parametrize('version', [114, 117])
    @pytest.mark.parametrize('byteorder', ['>', '<'])
    def test_bool_uint(self, byteorder, version):
        s0 = Series([0, 1, True], dtype=np.bool)
        s1 = Series([0, 1, 100], dtype=np.uint8)
        s2 = Series([0, 1, 255], dtype=np.uint8)
        s3 = Series([0, 1, 2 ** 15 - 100], dtype=np.uint16)
        s4 = Series([0, 1, 2 ** 16 - 1], dtype=np.uint16)
        s5 = Series([0, 1, 2 ** 31 - 100], dtype=np.uint32)
        s6 = Series([0, 1, 2 ** 32 - 1], dtype=np.uint32)

        original = DataFrame({'s0': s0, 's1': s1, 's2': s2, 's3': s3,
                              's4': s4, 's5': s5, 's6': s6})
        original.index.name = 'index'
        expected = original.copy()
        expected_types = (np.int8, np.int8, np.int16, np.int16, np.int32,
                          np.int32, np.float64)
        for c, t in zip(expected.columns, expected_types):
            expected[c] = expected[c].astype(t)

        with tm.ensure_clean() as path:
            original.to_stata(path, byteorder=byteorder, version=version)
            written_and_read_again = self.read_dta(path)
            written_and_read_again = written_and_read_again.set_index('index')
            tm.assert_frame_equal(written_and_read_again, expected)

    def test_variable_labels(self):
        with StataReader(self.dta16_115) as rdr:
            sr_115 = rdr.variable_labels()
        with StataReader(self.dta16_117) as rdr:
            sr_117 = rdr.variable_labels()
        keys = ('var1', 'var2', 'var3')
        labels = ('label1', 'label2', 'label3')
        for k, v in compat.iteritems(sr_115):
            assert k in sr_117
            assert v == sr_117[k]
            assert k in keys
            assert v in labels

    def test_minimal_size_col(self):
        str_lens = (1, 100, 244)
        s = {}
        for str_len in str_lens:
            s['s' + str(str_len)] = Series(['a' * str_len,
                                            'b' * str_len, 'c' * str_len])
        original = DataFrame(s)
        with tm.ensure_clean() as path:
            original.to_stata(path, write_index=False)

            with StataReader(path) as sr:
                typlist = sr.typlist
                variables = sr.varlist
                formats = sr.fmtlist
                for variable, fmt, typ in zip(variables, formats, typlist):
                    assert int(variable[1:]) == int(fmt[1:-1])
                    assert int(variable[1:]) == typ

    def test_excessively_long_string(self):
        str_lens = (1, 244, 500)
        s = {}
        for str_len in str_lens:
            s['s' + str(str_len)] = Series(['a' * str_len,
                                            'b' * str_len, 'c' * str_len])
        original = DataFrame(s)
        msg = (r"Fixed width strings in Stata \.dta files are limited to 244"
               r" \(or fewer\)\ncharacters\.  Column 's500' does not satisfy"
               r" this restriction\. Use the\n'version=117' parameter to write"
               r" the newer \(Stata 13 and later\) format\.")
        with pytest.raises(ValueError, match=msg):
            with tm.ensure_clean() as path:
                original.to_stata(path)

    def test_missing_value_generator(self):
        types = ('b', 'h', 'l')
        df = DataFrame([[0.0]], columns=['float_'])
        with tm.ensure_clean() as path:
            df.to_stata(path)
            with StataReader(path) as rdr:
                valid_range = rdr.VALID_RANGE
        expected_values = ['.' + chr(97 + i) for i in range(26)]
        expected_values.insert(0, '.')
        for t in types:
            offset = valid_range[t][1]
            for i in range(0, 27):
                val = StataMissingValue(offset + 1 + i)
                assert val.string == expected_values[i]

        # Test extremes for floats
        val = StataMissingValue(struct.unpack('<f', b'\x00\x00\x00\x7f')[0])
        assert val.string == '.'
        val = StataMissingValue(struct.unpack('<f', b'\x00\xd0\x00\x7f')[0])
        assert val.string == '.z'

        # Test extremes for floats
        val = StataMissingValue(struct.unpack(
            '<d', b'\x00\x00\x00\x00\x00\x00\xe0\x7f')[0])
        assert val.string == '.'
        val = StataMissingValue(struct.unpack(
            '<d', b'\x00\x00\x00\x00\x00\x1a\xe0\x7f')[0])
        assert val.string == '.z'

    @pytest.mark.parametrize(
        'file', ['dta17_113', 'dta17_115', 'dta17_117'])
    def test_missing_value_conversion(self, file):
        columns = ['int8_', 'int16_', 'int32_', 'float32_', 'float64_']
        smv = StataMissingValue(101)
        keys = [key for key in iterkeys(smv.MISSING_VALUES)]
        keys.sort()
        data = []
        for i in range(27):
            row = [StataMissingValue(keys[i + (j * 27)]) for j in range(5)]
            data.append(row)
        expected = DataFrame(data, columns=columns)

        parsed = read_stata(getattr(self, file), convert_missing=True)
        tm.assert_frame_equal(parsed, expected)

    def test_big_dates(self):
        yr = [1960, 2000, 9999, 100, 2262, 1677]
        mo = [1, 1, 12, 1, 4, 9]
        dd = [1, 1, 31, 1, 22, 23]
        hr = [0, 0, 23, 0, 0, 0]
        mm = [0, 0, 59, 0, 0, 0]
        ss = [0, 0, 59, 0, 0, 0]
        expected = []
        for i in range(len(yr)):
            row = []
            for j in range(7):
                if j == 0:
                    row.append(
                        datetime(yr[i], mo[i], dd[i], hr[i], mm[i], ss[i]))
                elif j == 6:
                    row.append(datetime(yr[i], 1, 1))
                else:
                    row.append(datetime(yr[i], mo[i], dd[i]))
            expected.append(row)
        expected.append([pd.NaT] * 7)
        columns = ['date_tc', 'date_td', 'date_tw', 'date_tm', 'date_tq',
                   'date_th', 'date_ty']

        # Fixes for weekly, quarterly,half,year
        expected[2][2] = datetime(9999, 12, 24)
        expected[2][3] = datetime(9999, 12, 1)
        expected[2][4] = datetime(9999, 10, 1)
        expected[2][5] = datetime(9999, 7, 1)
        expected[4][2] = datetime(2262, 4, 16)
        expected[4][3] = expected[4][4] = datetime(2262, 4, 1)
        expected[4][5] = expected[4][6] = datetime(2262, 1, 1)
        expected[5][2] = expected[5][3] = expected[
            5][4] = datetime(1677, 10, 1)
        expected[5][5] = expected[5][6] = datetime(1678, 1, 1)

        expected = DataFrame(expected, columns=columns, dtype=np.object)
        parsed_115 = read_stata(self.dta18_115)
        parsed_117 = read_stata(self.dta18_117)
        tm.assert_frame_equal(expected, parsed_115,
                              check_datetimelike_compat=True)
        tm.assert_frame_equal(expected, parsed_117,
                              check_datetimelike_compat=True)

        date_conversion = {c: c[-2:] for c in columns}
        # {c : c[-2:] for c in columns}
        with tm.ensure_clean() as path:
            expected.index.name = 'index'
            expected.to_stata(path, date_conversion)
            written_and_read_again = self.read_dta(path)
            tm.assert_frame_equal(written_and_read_again.set_index('index'),
                                  expected,
                                  check_datetimelike_compat=True)

    def test_dtype_conversion(self):
        expected = self.read_csv(self.csv15)
        expected['byte_'] = expected['byte_'].astype(np.int8)
        expected['int_'] = expected['int_'].astype(np.int16)
        expected['long_'] = expected['long_'].astype(np.int32)
        expected['float_'] = expected['float_'].astype(np.float32)
        expected['double_'] = expected['double_'].astype(np.float64)
        expected['date_td'] = expected['date_td'].apply(datetime.strptime,
                                                        args=('%Y-%m-%d',))

        no_conversion = read_stata(self.dta15_117,
                                   convert_dates=True)
        tm.assert_frame_equal(expected, no_conversion)

        conversion = read_stata(self.dta15_117,
                                convert_dates=True,
                                preserve_dtypes=False)

        # read_csv types are the same
        expected = self.read_csv(self.csv15)
        expected['date_td'] = expected['date_td'].apply(datetime.strptime,
                                                        args=('%Y-%m-%d',))

        tm.assert_frame_equal(expected, conversion)

    def test_drop_column(self):
        expected = self.read_csv(self.csv15)
        expected['byte_'] = expected['byte_'].astype(np.int8)
        expected['int_'] = expected['int_'].astype(np.int16)
        expected['long_'] = expected['long_'].astype(np.int32)
        expected['float_'] = expected['float_'].astype(np.float32)
        expected['double_'] = expected['double_'].astype(np.float64)
        expected['date_td'] = expected['date_td'].apply(datetime.strptime,
                                                        args=('%Y-%m-%d',))

        columns = ['byte_', 'int_', 'long_']
        expected = expected[columns]
        dropped = read_stata(self.dta15_117, convert_dates=True,
                             columns=columns)

        tm.assert_frame_equal(expected, dropped)

        # See PR 10757
        columns = ['int_', 'long_', 'byte_']
        expected = expected[columns]
        reordered = read_stata(self.dta15_117, convert_dates=True,
                               columns=columns)
        tm.assert_frame_equal(expected, reordered)

        msg = "columns contains duplicate entries"
        with pytest.raises(ValueError, match=msg):
            columns = ['byte_', 'byte_']
            read_stata(self.dta15_117, convert_dates=True, columns=columns)

        msg = ("The following columns were not found in the Stata data set:"
               " not_found")
        with pytest.raises(ValueError, match=msg):
            columns = ['byte_', 'int_', 'long_', 'not_found']
            read_stata(self.dta15_117, convert_dates=True, columns=columns)

    @pytest.mark.parametrize('version', [114, 117])
    @pytest.mark.filterwarnings(
        "ignore:\\nStata value:pandas.io.stata.ValueLabelTypeMismatch"
    )
    def test_categorical_writing(self, version):
        original = DataFrame.from_records(
            [
                ["one", "ten", "one", "one", "one", 1],
                ["two", "nine", "two", "two", "two", 2],
                ["three", "eight", "three", "three", "three", 3],
                ["four", "seven", 4, "four", "four", 4],
                ["five", "six", 5, np.nan, "five", 5],
                ["six", "five", 6, np.nan, "six", 6],
                ["seven", "four", 7, np.nan, "seven", 7],
                ["eight", "three", 8, np.nan, "eight", 8],
                ["nine", "two", 9, np.nan, "nine", 9],
                ["ten", "one", "ten", np.nan, "ten", 10]
            ],
            columns=['fully_labeled', 'fully_labeled2', 'incompletely_labeled',
                     'labeled_with_missings', 'float_labelled', 'unlabeled'])
        expected = original.copy()

        # these are all categoricals
        original = pd.concat([original[col].astype('category')
                              for col in original], axis=1)

        expected['incompletely_labeled'] = expected[
            'incompletely_labeled'].apply(str)
        expected['unlabeled'] = expected['unlabeled'].apply(str)
        expected = pd.concat([expected[col].astype('category')
                              for col in expected], axis=1)
        expected.index.name = 'index'

        with tm.ensure_clean() as path:
            original.to_stata(path, version=version)
            written_and_read_again = self.read_dta(path)
            res = written_and_read_again.set_index('index')
            tm.assert_frame_equal(res, expected, check_categorical=False)

    def test_categorical_warnings_and_errors(self):
        # Warning for non-string labels
        # Error for labels too long
        original = pd.DataFrame.from_records(
            [['a' * 10000],
             ['b' * 10000],
             ['c' * 10000],
             ['d' * 10000]],
            columns=['Too_long'])

        original = pd.concat([original[col].astype('category')
                              for col in original], axis=1)
        with tm.ensure_clean() as path:
            msg = ("Stata value labels for a single variable must have"
                   r" a combined length less than 32,000 characters\.")
            with pytest.raises(ValueError, match=msg):
                original.to_stata(path)

        original = pd.DataFrame.from_records(
            [['a'],
             ['b'],
             ['c'],
             ['d'],
             [1]],
            columns=['Too_long'])
        original = pd.concat([original[col].astype('category')
                              for col in original], axis=1)

        with tm.assert_produces_warning(pd.io.stata.ValueLabelTypeMismatch):
            original.to_stata(path)
            # should get a warning for mixed content

    @pytest.mark.parametrize('version', [114, 117])
    def test_categorical_with_stata_missing_values(self, version):
        values = [['a' + str(i)] for i in range(120)]
        values.append([np.nan])
        original = pd.DataFrame.from_records(values, columns=['many_labels'])
        original = pd.concat([original[col].astype('category')
                              for col in original], axis=1)
        original.index.name = 'index'
        with tm.ensure_clean() as path:
            original.to_stata(path, version=version)
            written_and_read_again = self.read_dta(path)
            res = written_and_read_again.set_index('index')
            tm.assert_frame_equal(res, original, check_categorical=False)

    @pytest.mark.parametrize(
        'file', ['dta19_115', 'dta19_117'])
    def test_categorical_order(self, file):
        # Directly construct using expected codes
        # Format is is_cat, col_name, labels (in order), underlying data
        expected = [(True, 'ordered', ['a', 'b', 'c', 'd', 'e'], np.arange(5)),
                    (True, 'reverse', ['a', 'b', 'c',
                                       'd', 'e'], np.arange(5)[::-1]),
                    (True, 'noorder', ['a', 'b', 'c', 'd',
                                       'e'], np.array([2, 1, 4, 0, 3])),
                    (True, 'floating', [
                     'a', 'b', 'c', 'd', 'e'], np.arange(0, 5)),
                    (True, 'float_missing', [
                     'a', 'd', 'e'], np.array([0, 1, 2, -1, -1])),
                    (False, 'nolabel', [
                     1.0, 2.0, 3.0, 4.0, 5.0], np.arange(5)),
                    (True, 'int32_mixed', ['d', 2, 'e', 'b', 'a'],
                     np.arange(5))]
        cols = []
        for is_cat, col, labels, codes in expected:
            if is_cat:
                cols.append((col, pd.Categorical.from_codes(codes, labels)))
            else:
                cols.append((col, pd.Series(labels, dtype=np.float32)))
        expected = DataFrame.from_dict(OrderedDict(cols))

        # Read with and with out categoricals, ensure order is identical
        file = getattr(self, file)
        parsed = read_stata(file)
        tm.assert_frame_equal(expected, parsed, check_categorical=False)

        # Check identity of codes
        for col in expected:
            if is_categorical_dtype(expected[col]):
                tm.assert_series_equal(expected[col].cat.codes,
                                       parsed[col].cat.codes)
                tm.assert_index_equal(expected[col].cat.categories,
                                      parsed[col].cat.categories)

    @pytest.mark.parametrize(
        'file', ['dta20_115', 'dta20_117'])
    def test_categorical_sorting(self, file):
        parsed = read_stata(getattr(self, file))

        # Sort based on codes, not strings
        parsed = parsed.sort_values("srh", na_position='first')

        # Don't sort index
        parsed.index = np.arange(parsed.shape[0])
        codes = [-1, -1, 0, 1, 1, 1, 2, 2, 3, 4]
        categories = ["Poor", "Fair", "Good", "Very good", "Excellent"]
        cat = pd.Categorical.from_codes(codes=codes, categories=categories)
        expected = pd.Series(cat, name='srh')
        tm.assert_series_equal(expected, parsed["srh"],
                               check_categorical=False)

    @pytest.mark.parametrize(
        'file', ['dta19_115', 'dta19_117'])
    def test_categorical_ordering(self, file):
        file = getattr(self, file)
        parsed = read_stata(file)

        parsed_unordered = read_stata(file,
                                      order_categoricals=False)
        for col in parsed:
            if not is_categorical_dtype(parsed[col]):
                continue
            assert parsed[col].cat.ordered
            assert not parsed_unordered[col].cat.ordered

    @pytest.mark.parametrize(
        'file', ['dta1_117', 'dta2_117', 'dta3_117',
                 'dta4_117', 'dta14_117', 'dta15_117',
                 'dta16_117', 'dta17_117', 'dta18_117',
                 'dta19_117', 'dta20_117'])
    @pytest.mark.parametrize(
        'chunksize', [1, 2])
    @pytest.mark.parametrize(
        'convert_categoricals', [False, True])
    @pytest.mark.parametrize(
        'convert_dates', [False, True])
    def test_read_chunks_117(self, file, chunksize,
                             convert_categoricals, convert_dates):
        fname = getattr(self, file)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            parsed = read_stata(
                fname,
                convert_categoricals=convert_categoricals,
                convert_dates=convert_dates)
        itr = read_stata(
            fname, iterator=True,
            convert_categoricals=convert_categoricals,
            convert_dates=convert_dates)

        pos = 0
        for j in range(5):
            with warnings.catch_warnings(record=True) as w:  # noqa
                warnings.simplefilter("always")
                try:
                    chunk = itr.read(chunksize)
                except StopIteration:
                    break
            from_frame = parsed.iloc[pos:pos + chunksize, :]
            tm.assert_frame_equal(
                from_frame, chunk, check_dtype=False,
                check_datetimelike_compat=True,
                check_categorical=False)

            pos += chunksize
        itr.close()

    def test_iterator(self):

        fname = self.dta3_117

        parsed = read_stata(fname)

        with read_stata(fname, iterator=True) as itr:
            chunk = itr.read(5)
            tm.assert_frame_equal(parsed.iloc[0:5, :], chunk)

        with read_stata(fname, chunksize=5) as itr:
            chunk = list(itr)
            tm.assert_frame_equal(parsed.iloc[0:5, :], chunk[0])

        with read_stata(fname, iterator=True) as itr:
            chunk = itr.get_chunk(5)
            tm.assert_frame_equal(parsed.iloc[0:5, :], chunk)

        with read_stata(fname, chunksize=5) as itr:
            chunk = itr.get_chunk()
            tm.assert_frame_equal(parsed.iloc[0:5, :], chunk)

        # GH12153
        with read_stata(fname, chunksize=4) as itr:
            from_chunks = pd.concat(itr)
        tm.assert_frame_equal(parsed, from_chunks)

    @pytest.mark.parametrize(
        'file', ['dta2_115', 'dta3_115', 'dta4_115',
                 'dta14_115', 'dta15_115', 'dta16_115',
                 'dta17_115', 'dta18_115', 'dta19_115',
                 'dta20_115'])
    @pytest.mark.parametrize(
        'chunksize', [1, 2])
    @pytest.mark.parametrize(
        'convert_categoricals', [False, True])
    @pytest.mark.parametrize(
        'convert_dates', [False, True])
    def test_read_chunks_115(self, file, chunksize,
                             convert_categoricals, convert_dates):
        fname = getattr(self, file)

        # Read the whole file
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            parsed = read_stata(
                fname,
                convert_categoricals=convert_categoricals,
                convert_dates=convert_dates)

        # Compare to what we get when reading by chunk
        itr = read_stata(
            fname, iterator=True,
            convert_dates=convert_dates,
            convert_categoricals=convert_categoricals)
        pos = 0
        for j in range(5):
            with warnings.catch_warnings(record=True) as w:  # noqa
                warnings.simplefilter("always")
                try:
                    chunk = itr.read(chunksize)
                except StopIteration:
                    break
            from_frame = parsed.iloc[pos:pos + chunksize, :]
            tm.assert_frame_equal(
                from_frame, chunk, check_dtype=False,
                check_datetimelike_compat=True,
                check_categorical=False)

            pos += chunksize
        itr.close()

    def test_read_chunks_columns(self):
        fname = self.dta3_117
        columns = ['quarter', 'cpi', 'm1']
        chunksize = 2

        parsed = read_stata(fname, columns=columns)
        with read_stata(fname, iterator=True) as itr:
            pos = 0
            for j in range(5):
                chunk = itr.read(chunksize, columns=columns)
                if chunk is None:
                    break
                from_frame = parsed.iloc[pos:pos + chunksize, :]
                tm.assert_frame_equal(from_frame, chunk, check_dtype=False)
                pos += chunksize

    @pytest.mark.parametrize('version', [114, 117])
    def test_write_variable_labels(self, version):
        # GH 13631, add support for writing variable labels
        original = pd.DataFrame({'a': [1, 2, 3, 4],
                                 'b': [1.0, 3.0, 27.0, 81.0],
                                 'c': ['Atlanta', 'Birmingham',
                                       'Cincinnati', 'Detroit']})
        original.index.name = 'index'
        variable_labels = {'a': 'City Rank', 'b': 'City Exponent', 'c': 'City'}
        with tm.ensure_clean() as path:
            original.to_stata(path,
                              variable_labels=variable_labels,
                              version=version)
            with StataReader(path) as sr:
                read_labels = sr.variable_labels()
            expected_labels = {'index': '',
                               'a': 'City Rank',
                               'b': 'City Exponent',
                               'c': 'City'}
            assert read_labels == expected_labels

        variable_labels['index'] = 'The Index'
        with tm.ensure_clean() as path:
            original.to_stata(path,
                              variable_labels=variable_labels,
                              version=version)
            with StataReader(path) as sr:
                read_labels = sr.variable_labels()
            assert read_labels == variable_labels

    @pytest.mark.parametrize('version', [114, 117])
    def test_invalid_variable_labels(self, version):
        original = pd.DataFrame({'a': [1, 2, 3, 4],
                                 'b': [1.0, 3.0, 27.0, 81.0],
                                 'c': ['Atlanta', 'Birmingham',
                                       'Cincinnati', 'Detroit']})
        original.index.name = 'index'
        variable_labels = {'a': 'very long' * 10,
                           'b': 'City Exponent',
                           'c': 'City'}
        with tm.ensure_clean() as path:
            msg = "Variable labels must be 80 characters or fewer"
            with pytest.raises(ValueError, match=msg):
                original.to_stata(path,
                                  variable_labels=variable_labels,
                                  version=version)

        variable_labels['a'] = u'invalid character Œ'
        with tm.ensure_clean() as path:
            msg = ("Variable labels must contain only characters that can be"
                   " encoded in Latin-1")
            with pytest.raises(ValueError, match=msg):
                original.to_stata(path,
                                  variable_labels=variable_labels,
                                  version=version)

    def test_write_variable_label_errors(self):
        original = pd.DataFrame({'a': [1, 2, 3, 4],
                                 'b': [1.0, 3.0, 27.0, 81.0],
                                 'c': ['Atlanta', 'Birmingham',
                                       'Cincinnati', 'Detroit']})
        values = [u'\u03A1', u'\u0391',
                  u'\u039D', u'\u0394',
                  u'\u0391', u'\u03A3']

        variable_labels_utf8 = {'a': 'City Rank',
                                'b': 'City Exponent',
                                'c': u''.join(values)}

        msg = ("Variable labels must contain only characters that can be"
               " encoded in Latin-1")
        with pytest.raises(ValueError, match=msg):
            with tm.ensure_clean() as path:
                original.to_stata(path, variable_labels=variable_labels_utf8)

        variable_labels_long = {'a': 'City Rank',
                                'b': 'City Exponent',
                                'c': 'A very, very, very long variable label '
                                     'that is too long for Stata which means '
                                     'that it has more than 80 characters'}

        msg = "Variable labels must be 80 characters or fewer"
        with pytest.raises(ValueError, match=msg):
            with tm.ensure_clean() as path:
                original.to_stata(path, variable_labels=variable_labels_long)

    def test_default_date_conversion(self):
        # GH 12259
        dates = [dt.datetime(1999, 12, 31, 12, 12, 12, 12000),
                 dt.datetime(2012, 12, 21, 12, 21, 12, 21000),
                 dt.datetime(1776, 7, 4, 7, 4, 7, 4000)]
        original = pd.DataFrame({'nums': [1.0, 2.0, 3.0],
                                 'strs': ['apple', 'banana', 'cherry'],
                                 'dates': dates})

        with tm.ensure_clean() as path:
            original.to_stata(path, write_index=False)
            reread = read_stata(path, convert_dates=True)
            tm.assert_frame_equal(original, reread)

            original.to_stata(path,
                              write_index=False,
                              convert_dates={'dates': 'tc'})
            direct = read_stata(path, convert_dates=True)
            tm.assert_frame_equal(reread, direct)

            dates_idx = original.columns.tolist().index('dates')
            original.to_stata(path,
                              write_index=False,
                              convert_dates={dates_idx: 'tc'})
            direct = read_stata(path, convert_dates=True)
            tm.assert_frame_equal(reread, direct)

    def test_unsupported_type(self):
        original = pd.DataFrame({'a': [1 + 2j, 2 + 4j]})

        msg = "Data type complex128 not supported"
        with pytest.raises(NotImplementedError, match=msg):
            with tm.ensure_clean() as path:
                original.to_stata(path)

    def test_unsupported_datetype(self):
        dates = [dt.datetime(1999, 12, 31, 12, 12, 12, 12000),
                 dt.datetime(2012, 12, 21, 12, 21, 12, 21000),
                 dt.datetime(1776, 7, 4, 7, 4, 7, 4000)]
        original = pd.DataFrame({'nums': [1.0, 2.0, 3.0],
                                 'strs': ['apple', 'banana', 'cherry'],
                                 'dates': dates})

        msg = "Format %tC not implemented"
        with pytest.raises(NotImplementedError, match=msg):
            with tm.ensure_clean() as path:
                original.to_stata(path, convert_dates={'dates': 'tC'})

        dates = pd.date_range('1-1-1990', periods=3, tz='Asia/Hong_Kong')
        original = pd.DataFrame({'nums': [1.0, 2.0, 3.0],
                                 'strs': ['apple', 'banana', 'cherry'],
                                 'dates': dates})
        with pytest.raises(NotImplementedError):
            with tm.ensure_clean() as path:
                original.to_stata(path)

    def test_repeated_column_labels(self):
        # GH 13923
        msg = (r"Value labels for column ethnicsn are not unique\. The"
               r" repeated labels are:\n\n-+wolof")
        with pytest.raises(ValueError, match=msg):
            read_stata(self.dta23, convert_categoricals=True)

    def test_stata_111(self):
        # 111 is an old version but still used by current versions of
        # SAS when exporting to Stata format. We do not know of any
        # on-line documentation for this version.
        df = read_stata(self.dta24_111)
        original = pd.DataFrame({'y': [1, 1, 1, 1, 1, 0, 0, np.NaN, 0, 0],
                                 'x': [1, 2, 1, 3, np.NaN, 4, 3, 5, 1, 6],
                                 'w': [2, np.NaN, 5, 2, 4, 4, 3, 1, 2, 3],
                                 'z': ['a', 'b', 'c', 'd', 'e', '', 'g', 'h',
                                       'i', 'j']})
        original = original[['y', 'x', 'w', 'z']]
        tm.assert_frame_equal(original, df)

    def test_out_of_range_double(self):
        # GH 14618
        df = DataFrame({'ColumnOk': [0.0,
                                     np.finfo(np.double).eps,
                                     4.49423283715579e+307],
                        'ColumnTooBig': [0.0,
                                         np.finfo(np.double).eps,
                                         np.finfo(np.double).max]})
        msg = (r"Column ColumnTooBig has a maximum value \(.+\)"
               r" outside the range supported by Stata \(.+\)")
        with pytest.raises(ValueError, match=msg):
            with tm.ensure_clean() as path:
                df.to_stata(path)

        df.loc[2, 'ColumnTooBig'] = np.inf
        msg = ("Column ColumnTooBig has a maximum value of infinity which"
               " is outside the range supported by Stata")
        with pytest.raises(ValueError, match=msg):
            with tm.ensure_clean() as path:
                df.to_stata(path)

    def test_out_of_range_float(self):
        original = DataFrame({'ColumnOk': [0.0,
                                           np.finfo(np.float32).eps,
                                           np.finfo(np.float32).max / 10.0],
                              'ColumnTooBig': [0.0,
                                               np.finfo(np.float32).eps,
                                               np.finfo(np.float32).max]})
        original.index.name = 'index'
        for col in original:
            original[col] = original[col].astype(np.float32)

        with tm.ensure_clean() as path:
            original.to_stata(path)
            reread = read_stata(path)
            original['ColumnTooBig'] = original['ColumnTooBig'].astype(
                np.float64)
            tm.assert_frame_equal(original,
                                  reread.set_index('index'))

        original.loc[2, 'ColumnTooBig'] = np.inf
        msg = ("Column ColumnTooBig has a maximum value of infinity which"
               " is outside the range supported by Stata")
        with pytest.raises(ValueError, match=msg):
            with tm.ensure_clean() as path:
                original.to_stata(path)

    def test_path_pathlib(self):
        df = tm.makeDataFrame()
        df.index.name = 'index'
        reader = lambda x: read_stata(x).set_index('index')
        result = tm.round_trip_pathlib(df.to_stata, reader)
        tm.assert_frame_equal(df, result)

    def test_pickle_path_localpath(self):
        df = tm.makeDataFrame()
        df.index.name = 'index'
        reader = lambda x: read_stata(x).set_index('index')
        result = tm.round_trip_localpath(df.to_stata, reader)
        tm.assert_frame_equal(df, result)

    @pytest.mark.parametrize(
        'write_index', [True, False])
    def test_value_labels_iterator(self, write_index):
        # GH 16923
        d = {'A': ['B', 'E', 'C', 'A', 'E']}
        df = pd.DataFrame(data=d)
        df['A'] = df['A'].astype('category')
        with tm.ensure_clean() as path:
            df.to_stata(path, write_index=write_index)

            with pd.read_stata(path, iterator=True) as dta_iter:
                value_labels = dta_iter.value_labels()
        assert value_labels == {'A': {0: 'A', 1: 'B', 2: 'C', 3: 'E'}}

    def test_set_index(self):
        # GH 17328
        df = tm.makeDataFrame()
        df.index.name = 'index'
        with tm.ensure_clean() as path:
            df.to_stata(path)
            reread = pd.read_stata(path, index_col='index')
        tm.assert_frame_equal(df, reread)

    @pytest.mark.parametrize(
        'column', ['ms', 'day', 'week', 'month', 'qtr', 'half', 'yr'])
    def test_date_parsing_ignores_format_details(self, column):
        # GH 17797
        #
        # Test that display formats are ignored when determining if a numeric
        # column is a date value.
        #
        # All date types are stored as numbers and format associated with the
        # column denotes both the type of the date and the display format.
        #
        # STATA supports 9 date types which each have distinct units. We test 7
        # of the 9 types, ignoring %tC and %tb. %tC is a variant of %tc that
        # accounts for leap seconds and %tb relies on STATAs business calendar.
        df = read_stata(self.stata_dates)
        unformatted = df.loc[0, column]
        formatted = df.loc[0, column + "_fmt"]
        assert unformatted == formatted

    def test_writer_117(self):
        original = DataFrame(data=[['string', 'object', 1, 1, 1, 1.1, 1.1,
                                    np.datetime64('2003-12-25'),
                                    'a', 'a' * 2045, 'a' * 5000, 'a'],
                                   ['string-1', 'object-1', 1, 1, 1, 1.1, 1.1,
                                    np.datetime64('2003-12-26'),
                                    'b', 'b' * 2045, '', '']
                                   ],
                             columns=['string', 'object', 'int8', 'int16',
                                      'int32', 'float32', 'float64',
                                      'datetime',
                                      's1', 's2045', 'srtl', 'forced_strl'])
        original['object'] = Series(original['object'], dtype=object)
        original['int8'] = Series(original['int8'], dtype=np.int8)
        original['int16'] = Series(original['int16'], dtype=np.int16)
        original['int32'] = original['int32'].astype(np.int32)
        original['float32'] = Series(original['float32'], dtype=np.float32)
        original.index.name = 'index'
        original.index = original.index.astype(np.int32)
        copy = original.copy()
        with tm.ensure_clean() as path:
            original.to_stata(path,
                              convert_dates={'datetime': 'tc'},
                              convert_strl=['forced_strl'],
                              version=117)
            written_and_read_again = self.read_dta(path)
            # original.index is np.int32, read index is np.int64
            tm.assert_frame_equal(written_and_read_again.set_index('index'),
                                  original, check_index_type=False)
            tm.assert_frame_equal(original, copy)

    def test_convert_strl_name_swap(self):
        original = DataFrame([['a' * 3000, 'A', 'apple'],
                              ['b' * 1000, 'B', 'banana']],
                             columns=['long1' * 10, 'long', 1])
        original.index.name = 'index'

        with tm.assert_produces_warning(pd.io.stata.InvalidColumnName):
            with tm.ensure_clean() as path:
                original.to_stata(path, convert_strl=['long', 1], version=117)
                reread = self.read_dta(path)
                reread = reread.set_index('index')
                reread.columns = original.columns
                tm.assert_frame_equal(reread, original,
                                      check_index_type=False)

    def test_invalid_date_conversion(self):
        # GH 12259
        dates = [dt.datetime(1999, 12, 31, 12, 12, 12, 12000),
                 dt.datetime(2012, 12, 21, 12, 21, 12, 21000),
                 dt.datetime(1776, 7, 4, 7, 4, 7, 4000)]
        original = pd.DataFrame({'nums': [1.0, 2.0, 3.0],
                                 'strs': ['apple', 'banana', 'cherry'],
                                 'dates': dates})

        with tm.ensure_clean() as path:
            msg = "convert_dates key must be a column or an integer"
            with pytest.raises(ValueError, match=msg):
                original.to_stata(path,
                                  convert_dates={'wrong_name': 'tc'})

    @pytest.mark.parametrize('version', [114, 117])
    def test_nonfile_writing(self, version):
        # GH 21041
        bio = io.BytesIO()
        df = tm.makeDataFrame()
        df.index.name = 'index'
        with tm.ensure_clean() as path:
            df.to_stata(bio, version=version)
            bio.seek(0)
            with open(path, 'wb') as dta:
                dta.write(bio.read())
            reread = pd.read_stata(path, index_col='index')
        tm.assert_frame_equal(df, reread)

    def test_gzip_writing(self):
        # writing version 117 requires seek and cannot be used with gzip
        df = tm.makeDataFrame()
        df.index.name = 'index'
        with tm.ensure_clean() as path:
            with gzip.GzipFile(path, 'wb') as gz:
                df.to_stata(gz, version=114)
            with gzip.GzipFile(path, 'rb') as gz:
                reread = pd.read_stata(gz, index_col='index')
        tm.assert_frame_equal(df, reread)

    def test_unicode_dta_118(self):
        unicode_df = self.read_dta(self.dta25_118)

        columns = ['utf8', 'latin1', 'ascii', 'utf8_strl', 'ascii_strl']
        values = [[u'ραηδας', u'PÄNDÄS', 'p', u'ραηδας', 'p'],
                  [u'ƤĀńĐąŜ', u'Ö', 'a', u'ƤĀńĐąŜ', 'a'],
                  [u'ᴘᴀᴎᴅᴀS', u'Ü', 'n', u'ᴘᴀᴎᴅᴀS', 'n'],
                  ['      ', '      ', 'd', '      ', 'd'],
                  [' ', '', 'a', ' ', 'a'],
                  ['', '', 's', '', 's'],
                  ['', '', ' ', '', ' ']]
        expected = pd.DataFrame(values, columns=columns)

        tm.assert_frame_equal(unicode_df, expected)

    def test_mixed_string_strl(self):
        # GH 23633
        output = [
            {'mixed': 'string' * 500,
             'number': 0},
            {'mixed': None,
             'number': 1}
        ]
        output = pd.DataFrame(output)
        output.number = output.number.astype('int32')

        with tm.ensure_clean() as path:
            output.to_stata(path, write_index=False, version=117)
            reread = read_stata(path)
            expected = output.fillna('')
            tm.assert_frame_equal(reread, expected)

            # Check strl supports all None (null)
            output.loc[:, 'mixed'] = None
            output.to_stata(path, write_index=False, convert_strl=['mixed'],
                            version=117)
            reread = read_stata(path)
            expected = output.fillna('')
            tm.assert_frame_equal(reread, expected)

    @pytest.mark.parametrize('version', [114, 117])
    def test_all_none_exception(self, version):
        output = [
            {'none': 'none',
             'number': 0},
            {'none': None,
             'number': 1}
        ]
        output = pd.DataFrame(output)
        output.loc[:, 'none'] = None
        with tm.ensure_clean() as path:
            msg = (r"Column `none` cannot be exported\.\n\n"
                   "Only string-like object arrays containing all strings or a"
                   r" mix of strings and None can be exported\. Object arrays"
                   r" containing only null values are prohibited\. Other"
                   " object typescannot be exported and must first be"
                   r" converted to one of the supported types\.")
            with pytest.raises(ValueError, match=msg):
                output.to_stata(path, version=version)

    @pytest.mark.parametrize('version', [114, 117])
    def test_invalid_file_not_written(self, version):
        content = 'Here is one __�__ Another one __·__ Another one __½__'
        df = DataFrame([content], columns=['invalid'])
        expected_exc = UnicodeEncodeError if PY3 else UnicodeDecodeError
        with tm.ensure_clean() as path:
            msg1 = (r"'latin-1' codec can't encode character '\\ufffd'"
                    r" in position 14: ordinal not in range\(256\)")
            msg2 = ("'ascii' codec can't decode byte 0xef in position 14:"
                    r" ordinal not in range\(128\)")
            with pytest.raises(expected_exc, match=r'{}|{}'.format(
                    msg1, msg2)):
                with tm.assert_produces_warning(ResourceWarning):
                    df.to_stata(path)

    def test_strl_latin1(self):
        # GH 23573, correct GSO data to reflect correct size
        output = DataFrame([[u'pandas'] * 2, [u'þâÑÐÅ§'] * 2],
                           columns=['var_str', 'var_strl'])

        with tm.ensure_clean() as path:
            output.to_stata(path, version=117, convert_strl=['var_strl'])
            with open(path, 'rb') as reread:
                content = reread.read()
                expected = u'þâÑÐÅ§'
                assert expected.encode('latin-1') in content
                assert expected.encode('utf-8') in content
                gsos = content.split(b'strls')[1][1:-2]
                for gso in gsos.split(b'GSO')[1:]:
                    val = gso.split(b'\x00')[-2]
                    size = gso[gso.find(b'\x82') + 1]
                    if not PY3:
                        size = ord(size)
                    assert len(val) == size - 1
