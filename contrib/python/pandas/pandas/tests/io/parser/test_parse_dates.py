# -*- coding: utf-8 -*-

"""
Tests date parsing functionality for all of the
parsers defined in parsers.py
"""

from datetime import date, datetime

import numpy as np
import pytest
import pytz

from pandas._libs.tslib import Timestamp
from pandas._libs.tslibs import parsing
from pandas.compat import StringIO, lrange, parse_date
from pandas.compat.numpy import np_array_datetime64_compat

import pandas as pd
from pandas import DataFrame, DatetimeIndex, Index, MultiIndex
from pandas.core.indexes.datetimes import date_range
import pandas.util.testing as tm

import pandas.io.date_converters as conv
import pandas.io.parsers as parsers


def test_separator_date_conflict(all_parsers):
    # Regression test for gh-4678
    #
    # Make sure thousands separator and
    # date parsing do not conflict.
    parser = all_parsers
    data = "06-02-2013;13:00;1-000.215"
    expected = DataFrame([[datetime(2013, 6, 2, 13, 0, 0), 1000.215]],
                         columns=["Date", 2])

    df = parser.read_csv(StringIO(data), sep=";", thousands="-",
                         parse_dates={"Date": [0, 1]}, header=None)
    tm.assert_frame_equal(df, expected)


@pytest.mark.parametrize("keep_date_col", [True, False])
def test_multiple_date_col_custom(all_parsers, keep_date_col):
    data = """\
KORD,19990127, 19:00:00, 18:56:00, 0.8100, 2.8100, 7.2000, 0.0000, 280.0000
KORD,19990127, 20:00:00, 19:56:00, 0.0100, 2.2100, 7.2000, 0.0000, 260.0000
KORD,19990127, 21:00:00, 20:56:00, -0.5900, 2.2100, 5.7000, 0.0000, 280.0000
KORD,19990127, 21:00:00, 21:18:00, -0.9900, 2.0100, 3.6000, 0.0000, 270.0000
KORD,19990127, 22:00:00, 21:56:00, -0.5900, 1.7100, 5.1000, 0.0000, 290.0000
KORD,19990127, 23:00:00, 22:56:00, -0.5900, 1.7100, 4.6000, 0.0000, 280.0000
"""
    parser = all_parsers

    def date_parser(*date_cols):
        """
        Test date parser.

        Parameters
        ----------
        date_cols : args
            The list of data columns to parse.

        Returns
        -------
        parsed : Series
        """
        return parsing.try_parse_dates(parsers._concat_date_cols(date_cols))

    result = parser.read_csv(StringIO(data), header=None,
                             date_parser=date_parser, prefix="X",
                             parse_dates={"actual": [1, 2],
                                          "nominal": [1, 3]},
                             keep_date_col=keep_date_col)
    expected = DataFrame([
        [datetime(1999, 1, 27, 19, 0), datetime(1999, 1, 27, 18, 56),
         "KORD", "19990127", " 19:00:00", " 18:56:00",
         0.81, 2.81, 7.2, 0.0, 280.0],
        [datetime(1999, 1, 27, 20, 0), datetime(1999, 1, 27, 19, 56),
         "KORD", "19990127", " 20:00:00", " 19:56:00",
         0.01, 2.21, 7.2, 0.0, 260.0],
        [datetime(1999, 1, 27, 21, 0), datetime(1999, 1, 27, 20, 56),
         "KORD", "19990127", " 21:00:00", " 20:56:00",
         -0.59, 2.21, 5.7, 0.0, 280.0],
        [datetime(1999, 1, 27, 21, 0), datetime(1999, 1, 27, 21, 18),
         "KORD", "19990127", " 21:00:00", " 21:18:00",
         -0.99, 2.01, 3.6, 0.0, 270.0],
        [datetime(1999, 1, 27, 22, 0), datetime(1999, 1, 27, 21, 56),
         "KORD", "19990127", " 22:00:00", " 21:56:00",
         -0.59, 1.71, 5.1, 0.0, 290.0],
        [datetime(1999, 1, 27, 23, 0), datetime(1999, 1, 27, 22, 56),
         "KORD", "19990127", " 23:00:00", " 22:56:00",
         -0.59, 1.71, 4.6, 0.0, 280.0],
    ], columns=["actual", "nominal", "X0", "X1", "X2",
                "X3", "X4", "X5", "X6", "X7", "X8"])

    if not keep_date_col:
        expected = expected.drop(["X1", "X2", "X3"], axis=1)
    elif parser.engine == "python":
        expected["X1"] = expected["X1"].astype(np.int64)

    # Python can sometimes be flaky about how
    # the aggregated columns are entered, so
    # this standardizes the order.
    result = result[expected.columns]
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("keep_date_col", [True, False])
def test_multiple_date_col(all_parsers, keep_date_col):
    data = """\
KORD,19990127, 19:00:00, 18:56:00, 0.8100, 2.8100, 7.2000, 0.0000, 280.0000
KORD,19990127, 20:00:00, 19:56:00, 0.0100, 2.2100, 7.2000, 0.0000, 260.0000
KORD,19990127, 21:00:00, 20:56:00, -0.5900, 2.2100, 5.7000, 0.0000, 280.0000
KORD,19990127, 21:00:00, 21:18:00, -0.9900, 2.0100, 3.6000, 0.0000, 270.0000
KORD,19990127, 22:00:00, 21:56:00, -0.5900, 1.7100, 5.1000, 0.0000, 290.0000
KORD,19990127, 23:00:00, 22:56:00, -0.5900, 1.7100, 4.6000, 0.0000, 280.0000
"""
    parser = all_parsers
    result = parser.read_csv(StringIO(data), header=None,
                             prefix="X", parse_dates=[[1, 2], [1, 3]],
                             keep_date_col=keep_date_col)
    expected = DataFrame([
        [datetime(1999, 1, 27, 19, 0), datetime(1999, 1, 27, 18, 56),
         "KORD", "19990127", " 19:00:00", " 18:56:00",
         0.81, 2.81, 7.2, 0.0, 280.0],
        [datetime(1999, 1, 27, 20, 0), datetime(1999, 1, 27, 19, 56),
         "KORD", "19990127", " 20:00:00", " 19:56:00",
         0.01, 2.21, 7.2, 0.0, 260.0],
        [datetime(1999, 1, 27, 21, 0), datetime(1999, 1, 27, 20, 56),
         "KORD", "19990127", " 21:00:00", " 20:56:00",
         -0.59, 2.21, 5.7, 0.0, 280.0],
        [datetime(1999, 1, 27, 21, 0), datetime(1999, 1, 27, 21, 18),
         "KORD", "19990127", " 21:00:00", " 21:18:00",
         -0.99, 2.01, 3.6, 0.0, 270.0],
        [datetime(1999, 1, 27, 22, 0), datetime(1999, 1, 27, 21, 56),
         "KORD", "19990127", " 22:00:00", " 21:56:00",
         -0.59, 1.71, 5.1, 0.0, 290.0],
        [datetime(1999, 1, 27, 23, 0), datetime(1999, 1, 27, 22, 56),
         "KORD", "19990127", " 23:00:00", " 22:56:00",
         -0.59, 1.71, 4.6, 0.0, 280.0],
    ], columns=["X1_X2", "X1_X3", "X0", "X1", "X2",
                "X3", "X4", "X5", "X6", "X7", "X8"])

    if not keep_date_col:
        expected = expected.drop(["X1", "X2", "X3"], axis=1)
    elif parser.engine == "python":
        expected["X1"] = expected["X1"].astype(np.int64)

    tm.assert_frame_equal(result, expected)


def test_date_col_as_index_col(all_parsers):
    data = """\
KORD,19990127 19:00:00, 18:56:00, 0.8100, 2.8100, 7.2000, 0.0000, 280.0000
KORD,19990127 20:00:00, 19:56:00, 0.0100, 2.2100, 7.2000, 0.0000, 260.0000
KORD,19990127 21:00:00, 20:56:00, -0.5900, 2.2100, 5.7000, 0.0000, 280.0000
KORD,19990127 21:00:00, 21:18:00, -0.9900, 2.0100, 3.6000, 0.0000, 270.0000
KORD,19990127 22:00:00, 21:56:00, -0.5900, 1.7100, 5.1000, 0.0000, 290.0000
"""
    parser = all_parsers
    result = parser.read_csv(StringIO(data), header=None, prefix="X",
                             parse_dates=[1], index_col=1)

    index = Index([datetime(1999, 1, 27, 19, 0), datetime(1999, 1, 27, 20, 0),
                   datetime(1999, 1, 27, 21, 0), datetime(1999, 1, 27, 21, 0),
                   datetime(1999, 1, 27, 22, 0)], name="X1")
    expected = DataFrame([
        ["KORD", " 18:56:00", 0.81, 2.81, 7.2, 0.0, 280.0],
        ["KORD", " 19:56:00", 0.01, 2.21, 7.2, 0.0, 260.0],
        ["KORD", " 20:56:00", -0.59, 2.21, 5.7, 0.0, 280.0],
        ["KORD", " 21:18:00", -0.99, 2.01, 3.6, 0.0, 270.0],
        ["KORD", " 21:56:00", -0.59, 1.71, 5.1, 0.0, 290.0],
    ], columns=["X0", "X2", "X3", "X4", "X5", "X6", "X7"], index=index)
    tm.assert_frame_equal(result, expected)


def test_multiple_date_cols_int_cast(all_parsers):
    data = ("KORD,19990127, 19:00:00, 18:56:00, 0.8100\n"
            "KORD,19990127, 20:00:00, 19:56:00, 0.0100\n"
            "KORD,19990127, 21:00:00, 20:56:00, -0.5900\n"
            "KORD,19990127, 21:00:00, 21:18:00, -0.9900\n"
            "KORD,19990127, 22:00:00, 21:56:00, -0.5900\n"
            "KORD,19990127, 23:00:00, 22:56:00, -0.5900")
    parse_dates = {"actual": [1, 2], "nominal": [1, 3]}
    parser = all_parsers

    result = parser.read_csv(StringIO(data), header=None,
                             date_parser=conv.parse_date_time,
                             parse_dates=parse_dates, prefix="X")
    expected = DataFrame([
        [datetime(1999, 1, 27, 19, 0), datetime(1999, 1, 27, 18, 56),
         "KORD", 0.81],
        [datetime(1999, 1, 27, 20, 0), datetime(1999, 1, 27, 19, 56),
         "KORD", 0.01],
        [datetime(1999, 1, 27, 21, 0), datetime(1999, 1, 27, 20, 56),
         "KORD", -0.59],
        [datetime(1999, 1, 27, 21, 0), datetime(1999, 1, 27, 21, 18),
         "KORD", -0.99],
        [datetime(1999, 1, 27, 22, 0), datetime(1999, 1, 27, 21, 56),
         "KORD", -0.59],
        [datetime(1999, 1, 27, 23, 0), datetime(1999, 1, 27, 22, 56),
         "KORD", -0.59],
    ], columns=["actual", "nominal", "X0", "X4"])

    # Python can sometimes be flaky about how
    # the aggregated columns are entered, so
    # this standardizes the order.
    result = result[expected.columns]
    tm.assert_frame_equal(result, expected)


def test_multiple_date_col_timestamp_parse(all_parsers):
    parser = all_parsers
    data = """05/31/2012,15:30:00.029,1306.25,1,E,0,,1306.25
05/31/2012,15:30:00.029,1306.25,8,E,0,,1306.25"""

    result = parser.read_csv(StringIO(data), parse_dates=[[0, 1]],
                             header=None, date_parser=Timestamp)
    expected = DataFrame([
        [Timestamp("05/31/2012, 15:30:00.029"),
         1306.25, 1, "E", 0, np.nan, 1306.25],
        [Timestamp("05/31/2012, 15:30:00.029"),
         1306.25, 8, "E", 0, np.nan, 1306.25]
    ], columns=["0_1", 2, 3, 4, 5, 6, 7])
    tm.assert_frame_equal(result, expected)


def test_multiple_date_cols_with_header(all_parsers):
    parser = all_parsers
    data = """\
ID,date,NominalTime,ActualTime,TDew,TAir,Windspeed,Precip,WindDir
KORD,19990127, 19:00:00, 18:56:00, 0.8100, 2.8100, 7.2000, 0.0000, 280.0000
KORD,19990127, 20:00:00, 19:56:00, 0.0100, 2.2100, 7.2000, 0.0000, 260.0000
KORD,19990127, 21:00:00, 20:56:00, -0.5900, 2.2100, 5.7000, 0.0000, 280.0000
KORD,19990127, 21:00:00, 21:18:00, -0.9900, 2.0100, 3.6000, 0.0000, 270.0000
KORD,19990127, 22:00:00, 21:56:00, -0.5900, 1.7100, 5.1000, 0.0000, 290.0000
KORD,19990127, 23:00:00, 22:56:00, -0.5900, 1.7100, 4.6000, 0.0000, 280.0000"""

    result = parser.read_csv(StringIO(data), parse_dates={"nominal": [1, 2]})
    expected = DataFrame([
        [datetime(1999, 1, 27, 19, 0), "KORD", " 18:56:00",
         0.81, 2.81, 7.2, 0.0, 280.0],
        [datetime(1999, 1, 27, 20, 0), "KORD", " 19:56:00",
         0.01, 2.21, 7.2, 0.0, 260.0],
        [datetime(1999, 1, 27, 21, 0), "KORD", " 20:56:00",
         -0.59, 2.21, 5.7, 0.0, 280.0],
        [datetime(1999, 1, 27, 21, 0), "KORD", " 21:18:00",
         -0.99, 2.01, 3.6, 0.0, 270.0],
        [datetime(1999, 1, 27, 22, 0), "KORD", " 21:56:00",
         -0.59, 1.71, 5.1, 0.0, 290.0],
        [datetime(1999, 1, 27, 23, 0), "KORD", " 22:56:00",
         -0.59, 1.71, 4.6, 0.0, 280.0],
    ], columns=["nominal", "ID", "ActualTime", "TDew",
                "TAir", "Windspeed", "Precip", "WindDir"])
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("data,parse_dates,msg", [
    ("""\
date_NominalTime,date,NominalTime
KORD1,19990127, 19:00:00
KORD2,19990127, 20:00:00""", [[1, 2]], ("New date column already "
                                        "in dict date_NominalTime")),
    ("""\
ID,date,nominalTime
KORD,19990127, 19:00:00
KORD,19990127, 20:00:00""", dict(ID=[1, 2]), "Date column ID already in dict")
])
def test_multiple_date_col_name_collision(all_parsers, data, parse_dates, msg):
    parser = all_parsers

    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(data), parse_dates=parse_dates)


def test_date_parser_int_bug(all_parsers):
    # see gh-3071
    parser = all_parsers
    data = ("posix_timestamp,elapsed,sys,user,queries,query_time,rows,"
            "accountid,userid,contactid,level,silo,method\n"
            "1343103150,0.062353,0,4,6,0.01690,3,"
            "12345,1,-1,3,invoice_InvoiceResource,search\n")

    result = parser.read_csv(
        StringIO(data), index_col=0, parse_dates=[0],
        date_parser=lambda x: datetime.utcfromtimestamp(int(x)))
    expected = DataFrame([[0.062353, 0, 4, 6, 0.01690, 3, 12345, 1, -1,
                           3, "invoice_InvoiceResource", "search"]],
                         columns=["elapsed", "sys", "user", "queries",
                                  "query_time", "rows", "accountid",
                                  "userid", "contactid", "level",
                                  "silo", "method"],
                         index=Index([Timestamp("2012-07-24 04:12:30")],
                                     name="posix_timestamp"))
    tm.assert_frame_equal(result, expected)


def test_nat_parse(all_parsers):
    # see gh-3062
    parser = all_parsers
    df = DataFrame(dict({"A": np.asarray(lrange(10), dtype="float64"),
                         "B": pd.Timestamp("20010101")}))
    df.iloc[3:6, :] = np.nan

    with tm.ensure_clean("__nat_parse_.csv") as path:
        df.to_csv(path)

        result = parser.read_csv(path, index_col=0, parse_dates=["B"])
        tm.assert_frame_equal(result, df)


def test_csv_custom_parser(all_parsers):
    data = """A,B,C
20090101,a,1,2
20090102,b,3,4
20090103,c,4,5
"""
    parser = all_parsers
    result = parser.read_csv(
        StringIO(data),
        date_parser=lambda x: datetime.strptime(x, "%Y%m%d"))
    expected = parser.read_csv(StringIO(data), parse_dates=True)
    tm.assert_frame_equal(result, expected)


def test_parse_dates_implicit_first_col(all_parsers):
    data = """A,B,C
20090101,a,1,2
20090102,b,3,4
20090103,c,4,5
"""
    parser = all_parsers
    result = parser.read_csv(StringIO(data), parse_dates=True)

    expected = parser.read_csv(StringIO(data), index_col=0,
                               parse_dates=True)
    tm.assert_frame_equal(result, expected)


def test_parse_dates_string(all_parsers):
    data = """date,A,B,C
20090101,a,1,2
20090102,b,3,4
20090103,c,4,5
"""
    parser = all_parsers
    result = parser.read_csv(StringIO(data), index_col="date",
                             parse_dates=["date"])
    index = date_range("1/1/2009", periods=3)
    index.name = "date"

    expected = DataFrame({"A": ["a", "b", "c"], "B": [1, 3, 4],
                          "C": [2, 4, 5]}, index=index)
    tm.assert_frame_equal(result, expected)


# Bug in https://github.com/dateutil/dateutil/issues/217
# has been addressed, but we just don't pass in the `yearfirst`
@pytest.mark.xfail(reason="yearfirst is not surfaced in read_*")
@pytest.mark.parametrize("parse_dates", [
    [["date", "time"]],
    [[0, 1]]
])
def test_yy_format_with_year_first(all_parsers, parse_dates):
    data = """date,time,B,C
090131,0010,1,2
090228,1020,3,4
090331,0830,5,6
"""
    parser = all_parsers
    result = parser.read_csv(StringIO(data), index_col=0,
                             parse_dates=parse_dates)
    index = DatetimeIndex([datetime(2009, 1, 31, 0, 10, 0),
                           datetime(2009, 2, 28, 10, 20, 0),
                           datetime(2009, 3, 31, 8, 30, 0)],
                          dtype=object, name="date_time")
    expected = DataFrame({"B": [1, 3, 5], "C": [2, 4, 6]}, index=index)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("parse_dates", [[0, 2], ["a", "c"]])
def test_parse_dates_column_list(all_parsers, parse_dates):
    data = "a,b,c\n01/01/2010,1,15/02/2010"
    parser = all_parsers

    expected = DataFrame({"a": [datetime(2010, 1, 1)], "b": [1],
                          "c": [datetime(2010, 2, 15)]})
    expected = expected.set_index(["a", "b"])

    result = parser.read_csv(StringIO(data), index_col=[0, 1],
                             parse_dates=parse_dates, dayfirst=True)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("index_col", [[0, 1], [1, 0]])
def test_multi_index_parse_dates(all_parsers, index_col):
    data = """index1,index2,A,B,C
20090101,one,a,1,2
20090101,two,b,3,4
20090101,three,c,4,5
20090102,one,a,1,2
20090102,two,b,3,4
20090102,three,c,4,5
20090103,one,a,1,2
20090103,two,b,3,4
20090103,three,c,4,5
"""
    parser = all_parsers
    index = MultiIndex.from_product([
        (datetime(2009, 1, 1), datetime(2009, 1, 2),
         datetime(2009, 1, 3)), ("one", "two", "three")],
        names=["index1", "index2"])

    # Out of order.
    if index_col == [1, 0]:
        index = index.swaplevel(0, 1)

    expected = DataFrame([["a", 1, 2], ["b", 3, 4], ["c", 4, 5],
                          ["a", 1, 2], ["b", 3, 4], ["c", 4, 5],
                          ["a", 1, 2], ["b", 3, 4], ["c", 4, 5]],
                         columns=["A", "B", "C"], index=index)
    result = parser.read_csv(StringIO(data), index_col=index_col,
                             parse_dates=True)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("kwargs", [
    dict(dayfirst=True), dict(day_first=True)
])
def test_parse_dates_custom_euro_format(all_parsers, kwargs):
    parser = all_parsers
    data = """foo,bar,baz
31/01/2010,1,2
01/02/2010,1,NA
02/02/2010,1,2
"""
    if "dayfirst" in kwargs:
        df = parser.read_csv(StringIO(data), names=["time", "Q", "NTU"],
                             date_parser=lambda d: parse_date(d, **kwargs),
                             header=0, index_col=0, parse_dates=True,
                             na_values=["NA"])
        exp_index = Index([datetime(2010, 1, 31), datetime(2010, 2, 1),
                           datetime(2010, 2, 2)], name="time")
        expected = DataFrame({"Q": [1, 1, 1], "NTU": [2, np.nan, 2]},
                             index=exp_index, columns=["Q", "NTU"])
        tm.assert_frame_equal(df, expected)
    else:
        msg = "got an unexpected keyword argument 'day_first'"
        with pytest.raises(TypeError, match=msg):
            parser.read_csv(StringIO(data), names=["time", "Q", "NTU"],
                            date_parser=lambda d: parse_date(d, **kwargs),
                            skiprows=[0], index_col=0, parse_dates=True,
                            na_values=["NA"])


def test_parse_tz_aware(all_parsers):
    # See gh-1693
    parser = all_parsers
    data = "Date,x\n2012-06-13T01:39:00Z,0.5"

    result = parser.read_csv(StringIO(data), index_col=0,
                             parse_dates=True)
    expected = DataFrame({"x": [0.5]}, index=Index([Timestamp(
        "2012-06-13 01:39:00+00:00")], name="Date"))
    tm.assert_frame_equal(result, expected)
    assert result.index.tz is pytz.utc


@pytest.mark.parametrize("parse_dates,index_col", [
    ({"nominal": [1, 2]}, "nominal"),
    ({"nominal": [1, 2]}, 0),
    ([[1, 2]], 0),
])
def test_multiple_date_cols_index(all_parsers, parse_dates, index_col):
    parser = all_parsers
    data = """
ID,date,NominalTime,ActualTime,TDew,TAir,Windspeed,Precip,WindDir
KORD1,19990127, 19:00:00, 18:56:00, 0.8100, 2.8100, 7.2000, 0.0000, 280.0000
KORD2,19990127, 20:00:00, 19:56:00, 0.0100, 2.2100, 7.2000, 0.0000, 260.0000
KORD3,19990127, 21:00:00, 20:56:00, -0.5900, 2.2100, 5.7000, 0.0000, 280.0000
KORD4,19990127, 21:00:00, 21:18:00, -0.9900, 2.0100, 3.6000, 0.0000, 270.0000
KORD5,19990127, 22:00:00, 21:56:00, -0.5900, 1.7100, 5.1000, 0.0000, 290.0000
KORD6,19990127, 23:00:00, 22:56:00, -0.5900, 1.7100, 4.6000, 0.0000, 280.0000
"""
    expected = DataFrame([
        [datetime(1999, 1, 27, 19, 0), "KORD1", " 18:56:00",
         0.81, 2.81, 7.2, 0.0, 280.0],
        [datetime(1999, 1, 27, 20, 0), "KORD2", " 19:56:00",
         0.01, 2.21, 7.2, 0.0, 260.0],
        [datetime(1999, 1, 27, 21, 0), "KORD3", " 20:56:00",
         -0.59, 2.21, 5.7, 0.0, 280.0],
        [datetime(1999, 1, 27, 21, 0), "KORD4", " 21:18:00",
         -0.99, 2.01, 3.6, 0.0, 270.0],
        [datetime(1999, 1, 27, 22, 0), "KORD5", " 21:56:00",
         -0.59, 1.71, 5.1, 0.0, 290.0],
        [datetime(1999, 1, 27, 23, 0), "KORD6", " 22:56:00",
         -0.59, 1.71, 4.6, 0.0, 280.0],
    ], columns=["nominal", "ID", "ActualTime", "TDew",
                "TAir", "Windspeed", "Precip", "WindDir"])
    expected = expected.set_index("nominal")

    if not isinstance(parse_dates, dict):
        expected.index.name = "date_NominalTime"

    result = parser.read_csv(StringIO(data), parse_dates=parse_dates,
                             index_col=index_col)
    tm.assert_frame_equal(result, expected)


def test_multiple_date_cols_chunked(all_parsers):
    parser = all_parsers
    data = """\
ID,date,nominalTime,actualTime,A,B,C,D,E
KORD,19990127, 19:00:00, 18:56:00, 0.8100, 2.8100, 7.2000, 0.0000, 280.0000
KORD,19990127, 20:00:00, 19:56:00, 0.0100, 2.2100, 7.2000, 0.0000, 260.0000
KORD,19990127, 21:00:00, 20:56:00, -0.5900, 2.2100, 5.7000, 0.0000, 280.0000
KORD,19990127, 21:00:00, 21:18:00, -0.9900, 2.0100, 3.6000, 0.0000, 270.0000
KORD,19990127, 22:00:00, 21:56:00, -0.5900, 1.7100, 5.1000, 0.0000, 290.0000
KORD,19990127, 23:00:00, 22:56:00, -0.5900, 1.7100, 4.6000, 0.0000, 280.0000
"""

    expected = DataFrame([
        [datetime(1999, 1, 27, 19, 0), "KORD", " 18:56:00",
         0.81, 2.81, 7.2, 0.0, 280.0],
        [datetime(1999, 1, 27, 20, 0), "KORD", " 19:56:00",
         0.01, 2.21, 7.2, 0.0, 260.0],
        [datetime(1999, 1, 27, 21, 0), "KORD", " 20:56:00",
         -0.59, 2.21, 5.7, 0.0, 280.0],
        [datetime(1999, 1, 27, 21, 0), "KORD", " 21:18:00",
         -0.99, 2.01, 3.6, 0.0, 270.0],
        [datetime(1999, 1, 27, 22, 0), "KORD", " 21:56:00",
         -0.59, 1.71, 5.1, 0.0, 290.0],
        [datetime(1999, 1, 27, 23, 0), "KORD", " 22:56:00",
         -0.59, 1.71, 4.6, 0.0, 280.0],
    ], columns=["nominal", "ID", "actualTime", "A", "B", "C", "D", "E"])
    expected = expected.set_index("nominal")

    reader = parser.read_csv(StringIO(data), parse_dates={"nominal": [1, 2]},
                             index_col="nominal", chunksize=2)
    chunks = list(reader)

    tm.assert_frame_equal(chunks[0], expected[:2])
    tm.assert_frame_equal(chunks[1], expected[2:4])
    tm.assert_frame_equal(chunks[2], expected[4:])


def test_multiple_date_col_named_index_compat(all_parsers):
    parser = all_parsers
    data = """\
ID,date,nominalTime,actualTime,A,B,C,D,E
KORD,19990127, 19:00:00, 18:56:00, 0.8100, 2.8100, 7.2000, 0.0000, 280.0000
KORD,19990127, 20:00:00, 19:56:00, 0.0100, 2.2100, 7.2000, 0.0000, 260.0000
KORD,19990127, 21:00:00, 20:56:00, -0.5900, 2.2100, 5.7000, 0.0000, 280.0000
KORD,19990127, 21:00:00, 21:18:00, -0.9900, 2.0100, 3.6000, 0.0000, 270.0000
KORD,19990127, 22:00:00, 21:56:00, -0.5900, 1.7100, 5.1000, 0.0000, 290.0000
KORD,19990127, 23:00:00, 22:56:00, -0.5900, 1.7100, 4.6000, 0.0000, 280.0000
"""

    with_indices = parser.read_csv(StringIO(data),
                                   parse_dates={"nominal": [1, 2]},
                                   index_col="nominal")
    with_names = parser.read_csv(StringIO(data), index_col="nominal",
                                 parse_dates={"nominal": [
                                     "date", "nominalTime"]})
    tm.assert_frame_equal(with_indices, with_names)


def test_multiple_date_col_multiple_index_compat(all_parsers):
    parser = all_parsers
    data = """\
ID,date,nominalTime,actualTime,A,B,C,D,E
KORD,19990127, 19:00:00, 18:56:00, 0.8100, 2.8100, 7.2000, 0.0000, 280.0000
KORD,19990127, 20:00:00, 19:56:00, 0.0100, 2.2100, 7.2000, 0.0000, 260.0000
KORD,19990127, 21:00:00, 20:56:00, -0.5900, 2.2100, 5.7000, 0.0000, 280.0000
KORD,19990127, 21:00:00, 21:18:00, -0.9900, 2.0100, 3.6000, 0.0000, 270.0000
KORD,19990127, 22:00:00, 21:56:00, -0.5900, 1.7100, 5.1000, 0.0000, 290.0000
KORD,19990127, 23:00:00, 22:56:00, -0.5900, 1.7100, 4.6000, 0.0000, 280.0000
"""
    result = parser.read_csv(StringIO(data), index_col=["nominal", "ID"],
                             parse_dates={"nominal": [1, 2]})
    expected = parser.read_csv(StringIO(data),
                               parse_dates={"nominal": [1, 2]})

    expected = expected.set_index(["nominal", "ID"])
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("kwargs", [dict(), dict(index_col="C")])
def test_read_with_parse_dates_scalar_non_bool(all_parsers, kwargs):
    # see gh-5636
    parser = all_parsers
    msg = ("Only booleans, lists, and dictionaries "
           "are accepted for the 'parse_dates' parameter")
    data = """A,B,C
    1,2,2003-11-1"""

    with pytest.raises(TypeError, match=msg):
        parser.read_csv(StringIO(data), parse_dates="C", **kwargs)


@pytest.mark.parametrize("parse_dates", [
    (1,), np.array([4, 5]), {1, 3, 3}
])
def test_read_with_parse_dates_invalid_type(all_parsers, parse_dates):
    parser = all_parsers
    msg = ("Only booleans, lists, and dictionaries "
           "are accepted for the 'parse_dates' parameter")
    data = """A,B,C
    1,2,2003-11-1"""

    with pytest.raises(TypeError, match=msg):
        parser.read_csv(StringIO(data), parse_dates=(1,))


def test_parse_dates_empty_string(all_parsers):
    # see gh-2263
    parser = all_parsers
    data = "Date,test\n2012-01-01,1\n,2"
    result = parser.read_csv(StringIO(data), parse_dates=["Date"],
                             na_filter=False)

    expected = DataFrame([[datetime(2012, 1, 1), 1], [pd.NaT, 2]],
                         columns=["Date", "test"])
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("data,kwargs,expected", [
    ("a\n04.15.2016", dict(parse_dates=["a"]),
     DataFrame([datetime(2016, 4, 15)], columns=["a"])),
    ("a\n04.15.2016", dict(parse_dates=True, index_col=0),
     DataFrame(index=DatetimeIndex(["2016-04-15"], name="a"))),
    ("a,b\n04.15.2016,09.16.2013", dict(parse_dates=["a", "b"]),
     DataFrame([[datetime(2016, 4, 15), datetime(2013, 9, 16)]],
               columns=["a", "b"])),
    ("a,b\n04.15.2016,09.16.2013", dict(parse_dates=True, index_col=[0, 1]),
     DataFrame(index=MultiIndex.from_tuples(
         [(datetime(2016, 4, 15), datetime(2013, 9, 16))], names=["a", "b"]))),
])
def test_parse_dates_no_convert_thousands(all_parsers, data, kwargs, expected):
    # see gh-14066
    parser = all_parsers

    result = parser.read_csv(StringIO(data), thousands=".", **kwargs)
    tm.assert_frame_equal(result, expected)


def test_parse_date_time_multi_level_column_name(all_parsers):
    data = """\
D,T,A,B
date, time,a,b
2001-01-05, 09:00:00, 0.0, 10.
2001-01-06, 00:00:00, 1.0, 11.
"""
    parser = all_parsers
    result = parser.read_csv(StringIO(data), header=[0, 1],
                             parse_dates={"date_time": [0, 1]},
                             date_parser=conv.parse_date_time)

    expected_data = [[datetime(2001, 1, 5, 9, 0, 0), 0., 10.],
                     [datetime(2001, 1, 6, 0, 0, 0), 1., 11.]]
    expected = DataFrame(expected_data,
                         columns=["date_time", ("A", "a"), ("B", "b")])
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("data,kwargs,expected", [
    ("""\
date,time,a,b
2001-01-05, 10:00:00, 0.0, 10.
2001-01-05, 00:00:00, 1., 11.
""", dict(header=0, parse_dates={"date_time": [0, 1]}),
     DataFrame([[datetime(2001, 1, 5, 10, 0, 0), 0.0, 10],
                [datetime(2001, 1, 5, 0, 0, 0), 1.0, 11.0]],
               columns=["date_time", "a", "b"])),
    (("KORD,19990127, 19:00:00, 18:56:00, 0.8100\n"
      "KORD,19990127, 20:00:00, 19:56:00, 0.0100\n"
      "KORD,19990127, 21:00:00, 20:56:00, -0.5900\n"
      "KORD,19990127, 21:00:00, 21:18:00, -0.9900\n"
      "KORD,19990127, 22:00:00, 21:56:00, -0.5900\n"
      "KORD,19990127, 23:00:00, 22:56:00, -0.5900"),
     dict(header=None, parse_dates={"actual": [1, 2], "nominal": [1, 3]}),
     DataFrame([
         [datetime(1999, 1, 27, 19, 0), datetime(1999, 1, 27, 18, 56),
          "KORD", 0.81],
         [datetime(1999, 1, 27, 20, 0), datetime(1999, 1, 27, 19, 56),
          "KORD", 0.01],
         [datetime(1999, 1, 27, 21, 0), datetime(1999, 1, 27, 20, 56),
          "KORD", -0.59],
         [datetime(1999, 1, 27, 21, 0), datetime(1999, 1, 27, 21, 18),
          "KORD", -0.99],
         [datetime(1999, 1, 27, 22, 0), datetime(1999, 1, 27, 21, 56),
          "KORD", -0.59],
         [datetime(1999, 1, 27, 23, 0), datetime(1999, 1, 27, 22, 56),
          "KORD", -0.59]], columns=["actual", "nominal", 0, 4])),
])
def test_parse_date_time(all_parsers, data, kwargs, expected):
    parser = all_parsers
    result = parser.read_csv(StringIO(data), date_parser=conv.parse_date_time,
                             **kwargs)

    # Python can sometimes be flaky about how
    # the aggregated columns are entered, so
    # this standardizes the order.
    result = result[expected.columns]
    tm.assert_frame_equal(result, expected)


def test_parse_date_fields(all_parsers):
    parser = all_parsers
    data = ("year,month,day,a\n2001,01,10,10.\n"
            "2001,02,1,11.")
    result = parser.read_csv(StringIO(data), header=0,
                             parse_dates={"ymd": [0, 1, 2]},
                             date_parser=conv.parse_date_fields)

    expected = DataFrame([[datetime(2001, 1, 10), 10.],
                          [datetime(2001, 2, 1), 11.]], columns=["ymd", "a"])
    tm.assert_frame_equal(result, expected)


def test_parse_date_all_fields(all_parsers):
    parser = all_parsers
    data = """\
year,month,day,hour,minute,second,a,b
2001,01,05,10,00,0,0.0,10.
2001,01,5,10,0,00,1.,11.
"""
    result = parser.read_csv(StringIO(data), header=0,
                             date_parser=conv.parse_all_fields,
                             parse_dates={"ymdHMS": [0, 1, 2, 3, 4, 5]})
    expected = DataFrame([[datetime(2001, 1, 5, 10, 0, 0), 0.0, 10.0],
                          [datetime(2001, 1, 5, 10, 0, 0), 1.0, 11.0]],
                         columns=["ymdHMS", "a", "b"])
    tm.assert_frame_equal(result, expected)


def test_datetime_fractional_seconds(all_parsers):
    parser = all_parsers
    data = """\
year,month,day,hour,minute,second,a,b
2001,01,05,10,00,0.123456,0.0,10.
2001,01,5,10,0,0.500000,1.,11.
"""
    result = parser.read_csv(StringIO(data), header=0,
                             date_parser=conv.parse_all_fields,
                             parse_dates={"ymdHMS": [0, 1, 2, 3, 4, 5]})
    expected = DataFrame([[datetime(2001, 1, 5, 10, 0, 0,
                                    microsecond=123456), 0.0, 10.0],
                          [datetime(2001, 1, 5, 10, 0, 0,
                                    microsecond=500000), 1.0, 11.0]],
                         columns=["ymdHMS", "a", "b"])
    tm.assert_frame_equal(result, expected)


def test_generic(all_parsers):
    parser = all_parsers
    data = "year,month,day,a\n2001,01,10,10.\n2001,02,1,11."

    result = parser.read_csv(StringIO(data), header=0,
                             parse_dates={"ym": [0, 1]},
                             date_parser=lambda y, m: date(year=int(y),
                                                           month=int(m),
                                                           day=1))
    expected = DataFrame([[date(2001, 1, 1), 10, 10.],
                          [date(2001, 2, 1), 1, 11.]],
                         columns=["ym", "day", "a"])
    tm.assert_frame_equal(result, expected)


def test_date_parser_resolution_if_not_ns(all_parsers):
    # see gh-10245
    parser = all_parsers
    data = """\
date,time,prn,rxstatus
2013-11-03,19:00:00,126,00E80000
2013-11-03,19:00:00,23,00E80000
2013-11-03,19:00:00,13,00E80000
"""

    def date_parser(dt, time):
        return np_array_datetime64_compat(dt + "T" + time + "Z",
                                          dtype="datetime64[s]")

    result = parser.read_csv(StringIO(data), date_parser=date_parser,
                             parse_dates={"datetime": ["date", "time"]},
                             index_col=["datetime", "prn"])

    datetimes = np_array_datetime64_compat(["2013-11-03T19:00:00Z"] * 3,
                                           dtype="datetime64[s]")
    expected = DataFrame(data={"rxstatus": ["00E80000"] * 3},
                         index=MultiIndex.from_tuples(
                             [(datetimes[0], 126), (datetimes[1], 23),
                              (datetimes[2], 13)], names=["datetime", "prn"]))
    tm.assert_frame_equal(result, expected)


def test_parse_date_column_with_empty_string(all_parsers):
    # see gh-6428
    parser = all_parsers
    data = "case,opdate\n7,10/18/2006\n7,10/18/2008\n621, "
    result = parser.read_csv(StringIO(data), parse_dates=["opdate"])

    expected_data = [[7, "10/18/2006"],
                     [7, "10/18/2008"],
                     [621, " "]]
    expected = DataFrame(expected_data, columns=["case", "opdate"])
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("data,expected", [
    ("a\n135217135789158401\n1352171357E+5",
     DataFrame({"a": [135217135789158401,
                      135217135700000]}, dtype="float64")),
    ("a\n99999999999\n123456789012345\n1234E+0",
     DataFrame({"a": [99999999999,
                      123456789012345,
                      1234]}, dtype="float64"))
])
@pytest.mark.parametrize("parse_dates", [True, False])
def test_parse_date_float(all_parsers, data, expected, parse_dates):
    # see gh-2697
    #
    # Date parsing should fail, so we leave the data untouched
    # (i.e. float precision should remain unchanged).
    parser = all_parsers

    result = parser.read_csv(StringIO(data), parse_dates=parse_dates)
    tm.assert_frame_equal(result, expected)


def test_parse_timezone(all_parsers):
    # see gh-22256
    parser = all_parsers
    data = """dt,val
              2018-01-04 09:01:00+09:00,23350
              2018-01-04 09:02:00+09:00,23400
              2018-01-04 09:03:00+09:00,23400
              2018-01-04 09:04:00+09:00,23400
              2018-01-04 09:05:00+09:00,23400"""
    result = parser.read_csv(StringIO(data), parse_dates=["dt"])

    dti = pd.date_range(start="2018-01-04 09:01:00",
                        end="2018-01-04 09:05:00", freq="1min",
                        tz=pytz.FixedOffset(540))
    expected_data = {"dt": dti, "val": [23350, 23400, 23400, 23400, 23400]}

    expected = DataFrame(expected_data)
    tm.assert_frame_equal(result, expected)
