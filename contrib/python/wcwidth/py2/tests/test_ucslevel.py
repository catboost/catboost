# coding: utf-8
"""Unicode version level tests for wcwidth."""
# std imports
import warnings

# 3rd party
import pytest

# local
import wcwidth


def test_latest():
    """wcwidth._wcmatch_version('latest') returns tail item."""
    # given,
    expected = wcwidth.list_versions()[-1]

    # exercise,
    result = wcwidth._wcmatch_version('latest')

    # verify.
    assert result == expected


def test_exact_410_str():
    """wcwidth._wcmatch_version('4.1.0') returns equal value (str)."""
    # given,
    given = expected = '4.1.0'

    # exercise,
    result = wcwidth._wcmatch_version(given)

    # verify.
    assert result == expected


def test_exact_410_unicode():
    """wcwidth._wcmatch_version(u'4.1.0') returns equal value (unicode)."""
    # given,
    given = expected = u'4.1.0'

    # exercise,
    result = wcwidth._wcmatch_version(given)

    # verify.
    assert result == expected


def test_nearest_505_str():
    """wcwidth._wcmatch_version('5.0.5') returns nearest '5.0.0'. (str)"""
    # given
    given, expected = '5.0.5', '5.0.0'

    # exercise
    result = wcwidth._wcmatch_version(given)

    # verify.
    assert result == expected


def test_nearest_505_unicode():
    """wcwidth._wcmatch_version(u'5.0.5') returns nearest u'5.0.0'. (unicode)"""
    # given
    given, expected = u'5.0.5', u'5.0.0'

    # exercise
    result = wcwidth._wcmatch_version(given)

    # verify.
    assert result == expected


def test_nearest_lowint40_str():
    """wcwidth._wcmatch_version('4.0') returns nearest '4.1.0'."""
    # given
    given, expected = '4.0', '4.1.0'
    warnings.resetwarnings()
    wcwidth._wcmatch_version.cache_clear()

    # exercise
    with pytest.warns(UserWarning):
        # warns that given version is lower than any available
        result = wcwidth._wcmatch_version(given)

    # verify.
    assert result == expected


def test_nearest_lowint40_unicode():
    """wcwidth._wcmatch_version(u'4.0') returns nearest u'4.1.0'."""
    # given
    given, expected = u'4.0', u'4.1.0'
    warnings.resetwarnings()
    wcwidth._wcmatch_version.cache_clear()

    # exercise
    with pytest.warns(UserWarning):
        # warns that given version is lower than any available
        result = wcwidth._wcmatch_version(given)

    # verify.
    assert result == expected


def test_nearest_800_str():
    """wcwidth._wcmatch_version('8') returns nearest '8.0.0'."""
    # given
    given, expected = '8', '8.0.0'

    # exercise
    result = wcwidth._wcmatch_version(given)

    # verify.
    assert result == expected


def test_nearest_800_unicode():
    """wcwidth._wcmatch_version(u'8') returns nearest u'8.0.0'."""
    # given
    given, expected = u'8', u'8.0.0'

    # exercise
    result = wcwidth._wcmatch_version(given)

    # verify.
    assert result == expected


def test_nearest_999_str():
    """wcwidth._wcmatch_version('999.0') returns nearest (latest)."""
    # given
    given, expected = '999.0', wcwidth.list_versions()[-1]

    # exercise
    result = wcwidth._wcmatch_version(given)

    # verify.
    assert result == expected


def test_nearest_999_unicode():
    """wcwidth._wcmatch_version(u'999.0') returns nearest (latest)."""
    # given
    given, expected = u'999.0', wcwidth.list_versions()[-1]

    # exercise
    result = wcwidth._wcmatch_version(given)

    # verify.
    assert result == expected


def test_nonint_unicode():
    """wcwidth._wcmatch_version(u'x.y.z') returns latest (unicode)."""
    # given
    given, expected = u'x.y.z', wcwidth.list_versions()[-1]
    warnings.resetwarnings()
    wcwidth._wcmatch_version.cache_clear()

    # exercise
    with pytest.warns(UserWarning):
        # warns that given version is not valid
        result = wcwidth._wcmatch_version(given)

    # verify.
    assert result == expected


def test_nonint_str():
    """wcwidth._wcmatch_version(u'x.y.z') returns latest (str)."""
    # given
    given, expected = 'x.y.z', wcwidth.list_versions()[-1]
    warnings.resetwarnings()
    wcwidth._wcmatch_version.cache_clear()

    # exercise
    with pytest.warns(UserWarning):
        # warns that given version is not valid
        result = wcwidth._wcmatch_version(given)

    # verify.
    assert result == expected
