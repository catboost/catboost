from dateutil.easter import easter
from dateutil.easter import EASTER_WESTERN, EASTER_ORTHODOX, EASTER_JULIAN

from datetime import date
import pytest

# List of easters between 1990 and 2050
western_easter_dates = [
    date(1990, 4, 15), date(1991, 3, 31), date(1992, 4, 19), date(1993, 4, 11),
    date(1994, 4,  3), date(1995, 4, 16), date(1996, 4,  7), date(1997, 3, 30),
    date(1998, 4, 12), date(1999, 4,  4),

    date(2000, 4, 23), date(2001, 4, 15), date(2002, 3, 31), date(2003, 4, 20),
    date(2004, 4, 11), date(2005, 3, 27), date(2006, 4, 16), date(2007, 4,  8),
    date(2008, 3, 23), date(2009, 4, 12),

    date(2010, 4,  4), date(2011, 4, 24), date(2012, 4,  8), date(2013, 3, 31),
    date(2014, 4, 20), date(2015, 4,  5), date(2016, 3, 27), date(2017, 4, 16),
    date(2018, 4,  1), date(2019, 4, 21),

    date(2020, 4, 12), date(2021, 4,  4), date(2022, 4, 17), date(2023, 4,  9),
    date(2024, 3, 31), date(2025, 4, 20), date(2026, 4,  5), date(2027, 3, 28),
    date(2028, 4, 16), date(2029, 4,  1),

    date(2030, 4, 21), date(2031, 4, 13), date(2032, 3, 28), date(2033, 4, 17),
    date(2034, 4,  9), date(2035, 3, 25), date(2036, 4, 13), date(2037, 4,  5),
    date(2038, 4, 25), date(2039, 4, 10),

    date(2040, 4,  1), date(2041, 4, 21), date(2042, 4,  6), date(2043, 3, 29),
    date(2044, 4, 17), date(2045, 4,  9), date(2046, 3, 25), date(2047, 4, 14),
    date(2048, 4,  5), date(2049, 4, 18), date(2050, 4, 10)
    ]

orthodox_easter_dates = [
    date(1990, 4, 15), date(1991, 4,  7), date(1992, 4, 26), date(1993, 4, 18),
    date(1994, 5,  1), date(1995, 4, 23), date(1996, 4, 14), date(1997, 4, 27),
    date(1998, 4, 19), date(1999, 4, 11),

    date(2000, 4, 30), date(2001, 4, 15), date(2002, 5,  5), date(2003, 4, 27),
    date(2004, 4, 11), date(2005, 5,  1), date(2006, 4, 23), date(2007, 4,  8),
    date(2008, 4, 27), date(2009, 4, 19),

    date(2010, 4,  4), date(2011, 4, 24), date(2012, 4, 15), date(2013, 5,  5),
    date(2014, 4, 20), date(2015, 4, 12), date(2016, 5,  1), date(2017, 4, 16),
    date(2018, 4,  8), date(2019, 4, 28),

    date(2020, 4, 19), date(2021, 5,  2), date(2022, 4, 24), date(2023, 4, 16),
    date(2024, 5,  5), date(2025, 4, 20), date(2026, 4, 12), date(2027, 5,  2),
    date(2028, 4, 16), date(2029, 4,  8),

    date(2030, 4, 28), date(2031, 4, 13), date(2032, 5,  2), date(2033, 4, 24),
    date(2034, 4,  9), date(2035, 4, 29), date(2036, 4, 20), date(2037, 4,  5),
    date(2038, 4, 25), date(2039, 4, 17),

    date(2040, 5,  6), date(2041, 4, 21), date(2042, 4, 13), date(2043, 5,  3),
    date(2044, 4, 24), date(2045, 4,  9), date(2046, 4, 29), date(2047, 4, 21),
    date(2048, 4,  5), date(2049, 4, 25), date(2050, 4, 17)
]

# A random smattering of Julian dates.
# Pulled values from http://www.kevinlaughery.com/east4099.html
julian_easter_dates = [
    date( 326, 4,  3), date( 375, 4,  5), date( 492, 4,  5), date( 552, 3, 31),
    date( 562, 4,  9), date( 569, 4, 21), date( 597, 4, 14), date( 621, 4, 19),
    date( 636, 3, 31), date( 655, 3, 29), date( 700, 4, 11), date( 725, 4,  8),
    date( 750, 3, 29), date( 782, 4,  7), date( 835, 4, 18), date( 849, 4, 14),
    date( 867, 3, 30), date( 890, 4, 12), date( 922, 4, 21), date( 934, 4,  6),
    date(1049, 3, 26), date(1058, 4, 19), date(1113, 4,  6), date(1119, 3, 30),
    date(1242, 4, 20), date(1255, 3, 28), date(1257, 4,  8), date(1258, 3, 24),
    date(1261, 4, 24), date(1278, 4, 17), date(1333, 4,  4), date(1351, 4, 17),
    date(1371, 4,  6), date(1391, 3, 26), date(1402, 3, 26), date(1412, 4,  3),
    date(1439, 4,  5), date(1445, 3, 28), date(1531, 4,  9), date(1555, 4, 14)
]


@pytest.mark.parametrize("easter_date", western_easter_dates)
def test_easter_western(easter_date):
    assert easter_date == easter(easter_date.year, EASTER_WESTERN)


@pytest.mark.parametrize("easter_date", orthodox_easter_dates)
def test_easter_orthodox(easter_date):
    assert easter_date == easter(easter_date.year, EASTER_ORTHODOX)


@pytest.mark.parametrize("easter_date", julian_easter_dates)
def test_easter_julian(easter_date):
    assert easter_date == easter(easter_date.year, EASTER_JULIAN)


def test_easter_bad_method():
    with pytest.raises(ValueError):
        easter(1975, 4)
