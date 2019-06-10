from __future__ import division

from contextlib import contextmanager
from datetime import datetime
from functools import wraps
import locale
import os
import re
from shutil import rmtree
import string
import subprocess
import sys
import tempfile
import traceback
import warnings

import numpy as np
from numpy.random import rand, randn

from pandas._libs import testing as _testing
import pandas.compat as compat
from pandas.compat import (
    PY2, PY3, Counter, callable, filter, httplib, lmap, lrange, lzip, map,
    raise_with_traceback, range, string_types, u, unichr, zip)

from pandas.core.dtypes.common import (
    is_bool, is_categorical_dtype, is_datetime64_dtype, is_datetime64tz_dtype,
    is_datetimelike_v_numeric, is_datetimelike_v_object,
    is_extension_array_dtype, is_interval_dtype, is_list_like, is_number,
    is_period_dtype, is_sequence, is_timedelta64_dtype, needs_i8_conversion)
from pandas.core.dtypes.missing import array_equivalent

import pandas as pd
from pandas import (
    Categorical, CategoricalIndex, DataFrame, DatetimeIndex, Index,
    IntervalIndex, MultiIndex, Panel, RangeIndex, Series, bdate_range)
from pandas.core.algorithms import take_1d
from pandas.core.arrays import (
    DatetimeArray, ExtensionArray, IntervalArray, PeriodArray, TimedeltaArray,
    period_array)
import pandas.core.common as com

from pandas.io.common import urlopen
from pandas.io.formats.printing import pprint_thing

N = 30
K = 4
_RAISE_NETWORK_ERROR_DEFAULT = False

# set testing_mode
_testing_mode_warnings = (DeprecationWarning, compat.ResourceWarning)


def set_testing_mode():
    # set the testing mode filters
    testing_mode = os.environ.get('PANDAS_TESTING_MODE', 'None')
    if 'deprecate' in testing_mode:
        warnings.simplefilter('always', _testing_mode_warnings)


def reset_testing_mode():
    # reset the testing mode filters
    testing_mode = os.environ.get('PANDAS_TESTING_MODE', 'None')
    if 'deprecate' in testing_mode:
        warnings.simplefilter('ignore', _testing_mode_warnings)


set_testing_mode()


def reset_display_options():
    """
    Reset the display options for printing and representing objects.
    """

    pd.reset_option('^display.', silent=True)


def round_trip_pickle(obj, path=None):
    """
    Pickle an object and then read it again.

    Parameters
    ----------
    obj : pandas object
        The object to pickle and then re-read.
    path : str, default None
        The path where the pickled object is written and then read.

    Returns
    -------
    round_trip_pickled_object : pandas object
        The original object that was pickled and then re-read.
    """

    if path is None:
        path = u('__{random_bytes}__.pickle'.format(random_bytes=rands(10)))
    with ensure_clean(path) as path:
        pd.to_pickle(obj, path)
        return pd.read_pickle(path)


def round_trip_pathlib(writer, reader, path=None):
    """
    Write an object to file specified by a pathlib.Path and read it back

    Parameters
    ----------
    writer : callable bound to pandas object
        IO writing function (e.g. DataFrame.to_csv )
    reader : callable
        IO reading function (e.g. pd.read_csv )
    path : str, default None
        The path where the object is written and then read.

    Returns
    -------
    round_trip_object : pandas object
        The original object that was serialized and then re-read.
    """

    import pytest
    Path = pytest.importorskip('pathlib').Path
    if path is None:
        path = '___pathlib___'
    with ensure_clean(path) as path:
        writer(Path(path))
        obj = reader(Path(path))
    return obj


def round_trip_localpath(writer, reader, path=None):
    """
    Write an object to file specified by a py.path LocalPath and read it back

    Parameters
    ----------
    writer : callable bound to pandas object
        IO writing function (e.g. DataFrame.to_csv )
    reader : callable
        IO reading function (e.g. pd.read_csv )
    path : str, default None
        The path where the object is written and then read.

    Returns
    -------
    round_trip_object : pandas object
        The original object that was serialized and then re-read.
    """
    import pytest
    LocalPath = pytest.importorskip('py.path').local
    if path is None:
        path = '___localpath___'
    with ensure_clean(path) as path:
        writer(LocalPath(path))
        obj = reader(LocalPath(path))
    return obj


@contextmanager
def decompress_file(path, compression):
    """
    Open a compressed file and return a file object

    Parameters
    ----------
    path : str
        The path where the file is read from

    compression : {'gzip', 'bz2', 'zip', 'xz', None}
        Name of the decompression to use

    Returns
    -------
    f : file object
    """

    if compression is None:
        f = open(path, 'rb')
    elif compression == 'gzip':
        import gzip
        f = gzip.open(path, 'rb')
    elif compression == 'bz2':
        import bz2
        f = bz2.BZ2File(path, 'rb')
    elif compression == 'xz':
        lzma = compat.import_lzma()
        f = lzma.LZMAFile(path, 'rb')
    elif compression == 'zip':
        import zipfile
        zip_file = zipfile.ZipFile(path)
        zip_names = zip_file.namelist()
        if len(zip_names) == 1:
            f = zip_file.open(zip_names.pop())
        else:
            raise ValueError('ZIP file {} error. Only one file per ZIP.'
                             .format(path))
    else:
        msg = 'Unrecognized compression type: {}'.format(compression)
        raise ValueError(msg)

    try:
        yield f
    finally:
        f.close()
        if compression == "zip":
            zip_file.close()


def write_to_compressed(compression, path, data, dest="test"):
    """
    Write data to a compressed file.

    Parameters
    ----------
    compression : {'gzip', 'bz2', 'zip', 'xz'}
        The compression type to use.
    path : str
        The file path to write the data.
    data : str
        The data to write.
    dest : str, default "test"
        The destination file (for ZIP only)

    Raises
    ------
    ValueError : An invalid compression value was passed in.
    """

    if compression == "zip":
        import zipfile
        compress_method = zipfile.ZipFile
    elif compression == "gzip":
        import gzip
        compress_method = gzip.GzipFile
    elif compression == "bz2":
        import bz2
        compress_method = bz2.BZ2File
    elif compression == "xz":
        lzma = compat.import_lzma()
        compress_method = lzma.LZMAFile
    else:
        msg = "Unrecognized compression type: {}".format(compression)
        raise ValueError(msg)

    if compression == "zip":
        mode = "w"
        args = (dest, data)
        method = "writestr"
    else:
        mode = "wb"
        args = (data,)
        method = "write"

    with compress_method(path, mode=mode) as f:
        getattr(f, method)(*args)


def assert_almost_equal(left, right, check_dtype="equiv",
                        check_less_precise=False, **kwargs):
    """
    Check that the left and right objects are approximately equal.

    By approximately equal, we refer to objects that are numbers or that
    contain numbers which may be equivalent to specific levels of precision.

    Parameters
    ----------
    left : object
    right : object
    check_dtype : bool / string {'equiv'}, default 'equiv'
        Check dtype if both a and b are the same type. If 'equiv' is passed in,
        then `RangeIndex` and `Int64Index` are also considered equivalent
        when doing type checking.
    check_less_precise : bool or int, default False
        Specify comparison precision. 5 digits (False) or 3 digits (True)
        after decimal points are compared. If int, then specify the number
        of digits to compare.

        When comparing two numbers, if the first number has magnitude less
        than 1e-5, we compare the two numbers directly and check whether
        they are equivalent within the specified precision. Otherwise, we
        compare the **ratio** of the second number to the first number and
        check whether it is equivalent to 1 within the specified precision.
    """

    if isinstance(left, pd.Index):
        return assert_index_equal(left, right,
                                  check_exact=False,
                                  exact=check_dtype,
                                  check_less_precise=check_less_precise,
                                  **kwargs)

    elif isinstance(left, pd.Series):
        return assert_series_equal(left, right,
                                   check_exact=False,
                                   check_dtype=check_dtype,
                                   check_less_precise=check_less_precise,
                                   **kwargs)

    elif isinstance(left, pd.DataFrame):
        return assert_frame_equal(left, right,
                                  check_exact=False,
                                  check_dtype=check_dtype,
                                  check_less_precise=check_less_precise,
                                  **kwargs)

    else:
        # Other sequences.
        if check_dtype:
            if is_number(left) and is_number(right):
                # Do not compare numeric classes, like np.float64 and float.
                pass
            elif is_bool(left) and is_bool(right):
                # Do not compare bool classes, like np.bool_ and bool.
                pass
            else:
                if (isinstance(left, np.ndarray) or
                        isinstance(right, np.ndarray)):
                    obj = "numpy array"
                else:
                    obj = "Input"
                assert_class_equal(left, right, obj=obj)
        return _testing.assert_almost_equal(
            left, right,
            check_dtype=check_dtype,
            check_less_precise=check_less_precise,
            **kwargs)


def _check_isinstance(left, right, cls):
    """
    Helper method for our assert_* methods that ensures that
    the two objects being compared have the right type before
    proceeding with the comparison.

    Parameters
    ----------
    left : The first object being compared.
    right : The second object being compared.
    cls : The class type to check against.

    Raises
    ------
    AssertionError : Either `left` or `right` is not an instance of `cls`.
    """

    err_msg = "{name} Expected type {exp_type}, found {act_type} instead"
    cls_name = cls.__name__

    if not isinstance(left, cls):
        raise AssertionError(err_msg.format(name=cls_name, exp_type=cls,
                                            act_type=type(left)))
    if not isinstance(right, cls):
        raise AssertionError(err_msg.format(name=cls_name, exp_type=cls,
                                            act_type=type(right)))


def assert_dict_equal(left, right, compare_keys=True):

    _check_isinstance(left, right, dict)
    return _testing.assert_dict_equal(left, right, compare_keys=compare_keys)


def randbool(size=(), p=0.5):
    return rand(*size) <= p


RANDS_CHARS = np.array(list(string.ascii_letters + string.digits),
                       dtype=(np.str_, 1))
RANDU_CHARS = np.array(list(u("").join(map(unichr, lrange(1488, 1488 + 26))) +
                            string.digits), dtype=(np.unicode_, 1))


def rands_array(nchars, size, dtype='O'):
    """Generate an array of byte strings."""
    retval = (np.random.choice(RANDS_CHARS, size=nchars * np.prod(size))
              .view((np.str_, nchars)).reshape(size))
    if dtype is None:
        return retval
    else:
        return retval.astype(dtype)


def randu_array(nchars, size, dtype='O'):
    """Generate an array of unicode strings."""
    retval = (np.random.choice(RANDU_CHARS, size=nchars * np.prod(size))
              .view((np.unicode_, nchars)).reshape(size))
    if dtype is None:
        return retval
    else:
        return retval.astype(dtype)


def rands(nchars):
    """
    Generate one random byte string.

    See `rands_array` if you want to create an array of random strings.

    """
    return ''.join(np.random.choice(RANDS_CHARS, nchars))


def randu(nchars):
    """
    Generate one random unicode string.

    See `randu_array` if you want to create an array of random unicode strings.

    """
    return ''.join(np.random.choice(RANDU_CHARS, nchars))


def close(fignum=None):
    from matplotlib.pyplot import get_fignums, close as _close

    if fignum is None:
        for fignum in get_fignums():
            _close(fignum)
    else:
        _close(fignum)


# -----------------------------------------------------------------------------
# locale utilities


def check_output(*popenargs, **kwargs):
    # shamelessly taken from Python 2.7 source
    r"""Run command with arguments and return its output as a byte string.

    If the exit code was non-zero it raises a CalledProcessError.  The
    CalledProcessError object will have the return code in the returncode
    attribute and output in the output attribute.

    The arguments are the same as for the Popen constructor.  Example:

    >>> check_output(["ls", "-l", "/dev/null"])
    'crw-rw-rw- 1 root root 1, 3 Oct 18  2007 /dev/null\n'

    The stdout argument is not allowed as it is used internally.
    To capture standard error in the result, use stderr=STDOUT.

    >>> check_output(["/bin/sh", "-c",
    ...               "ls -l non_existent_file ; exit 0"],
    ...              stderr=STDOUT)
    'ls: non_existent_file: No such file or directory\n'
    """
    if 'stdout' in kwargs:
        raise ValueError('stdout argument not allowed, it will be overridden.')
    process = subprocess.Popen(stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               *popenargs, **kwargs)
    output, unused_err = process.communicate()
    retcode = process.poll()
    if retcode:
        cmd = kwargs.get("args")
        if cmd is None:
            cmd = popenargs[0]
        raise subprocess.CalledProcessError(retcode, cmd, output=output)
    return output


def _default_locale_getter():
    try:
        raw_locales = check_output(['locale -a'], shell=True)
    except subprocess.CalledProcessError as e:
        raise type(e)("{exception}, the 'locale -a' command cannot be found "
                      "on your system".format(exception=e))
    return raw_locales


def get_locales(prefix=None, normalize=True,
                locale_getter=_default_locale_getter):
    """Get all the locales that are available on the system.

    Parameters
    ----------
    prefix : str
        If not ``None`` then return only those locales with the prefix
        provided. For example to get all English language locales (those that
        start with ``"en"``), pass ``prefix="en"``.
    normalize : bool
        Call ``locale.normalize`` on the resulting list of available locales.
        If ``True``, only locales that can be set without throwing an
        ``Exception`` are returned.
    locale_getter : callable
        The function to use to retrieve the current locales. This should return
        a string with each locale separated by a newline character.

    Returns
    -------
    locales : list of strings
        A list of locale strings that can be set with ``locale.setlocale()``.
        For example::

            locale.setlocale(locale.LC_ALL, locale_string)

    On error will return None (no locale available, e.g. Windows)

    """
    try:
        raw_locales = locale_getter()
    except Exception:
        return None

    try:
        # raw_locales is "\n" separated list of locales
        # it may contain non-decodable parts, so split
        # extract what we can and then rejoin.
        raw_locales = raw_locales.split(b'\n')
        out_locales = []
        for x in raw_locales:
            if PY3:
                out_locales.append(str(
                    x, encoding=pd.options.display.encoding))
            else:
                out_locales.append(str(x))

    except TypeError:
        pass

    if prefix is None:
        return _valid_locales(out_locales, normalize)

    pattern = re.compile('{prefix}.*'.format(prefix=prefix))
    found = pattern.findall('\n'.join(out_locales))
    return _valid_locales(found, normalize)


@contextmanager
def set_locale(new_locale, lc_var=locale.LC_ALL):
    """Context manager for temporarily setting a locale.

    Parameters
    ----------
    new_locale : str or tuple
        A string of the form <language_country>.<encoding>. For example to set
        the current locale to US English with a UTF8 encoding, you would pass
        "en_US.UTF-8".
    lc_var : int, default `locale.LC_ALL`
        The category of the locale being set.

    Notes
    -----
    This is useful when you want to run a particular block of code under a
    particular locale, without globally setting the locale. This probably isn't
    thread-safe.
    """
    current_locale = locale.getlocale()

    try:
        locale.setlocale(lc_var, new_locale)
        normalized_locale = locale.getlocale()
        if com._all_not_none(*normalized_locale):
            yield '.'.join(normalized_locale)
        else:
            yield new_locale
    finally:
        locale.setlocale(lc_var, current_locale)


def can_set_locale(lc, lc_var=locale.LC_ALL):
    """
    Check to see if we can set a locale, and subsequently get the locale,
    without raising an Exception.

    Parameters
    ----------
    lc : str
        The locale to attempt to set.
    lc_var : int, default `locale.LC_ALL`
        The category of the locale being set.

    Returns
    -------
    is_valid : bool
        Whether the passed locale can be set
    """

    try:
        with set_locale(lc, lc_var=lc_var):
            pass
    except (ValueError,
            locale.Error):  # horrible name for a Exception subclass
        return False
    else:
        return True


def _valid_locales(locales, normalize):
    """Return a list of normalized locales that do not throw an ``Exception``
    when set.

    Parameters
    ----------
    locales : str
        A string where each locale is separated by a newline.
    normalize : bool
        Whether to call ``locale.normalize`` on each locale.

    Returns
    -------
    valid_locales : list
        A list of valid locales.
    """
    if normalize:
        normalizer = lambda x: locale.normalize(x.strip())
    else:
        normalizer = lambda x: x.strip()

    return list(filter(can_set_locale, map(normalizer, locales)))

# -----------------------------------------------------------------------------
# Stdout / stderr decorators


@contextmanager
def set_defaultencoding(encoding):
    """
    Set default encoding (as given by sys.getdefaultencoding()) to the given
    encoding; restore on exit.

    Parameters
    ----------
    encoding : str
    """
    if not PY2:
        raise ValueError("set_defaultencoding context is only available "
                         "in Python 2.")
    orig = sys.getdefaultencoding()
    reload(sys)  # noqa:F821
    sys.setdefaultencoding(encoding)
    try:
        yield
    finally:
        sys.setdefaultencoding(orig)


# -----------------------------------------------------------------------------
# Console debugging tools


def debug(f, *args, **kwargs):
    from pdb import Pdb as OldPdb
    try:
        from IPython.core.debugger import Pdb
        kw = dict(color_scheme='Linux')
    except ImportError:
        Pdb = OldPdb
        kw = {}
    pdb = Pdb(**kw)
    return pdb.runcall(f, *args, **kwargs)


def pudebug(f, *args, **kwargs):
    import pudb
    return pudb.runcall(f, *args, **kwargs)


def set_trace():
    from IPython.core.debugger import Pdb
    try:
        Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)
    except Exception:
        from pdb import Pdb as OldPdb
        OldPdb().set_trace(sys._getframe().f_back)

# -----------------------------------------------------------------------------
# contextmanager to ensure the file cleanup


@contextmanager
def ensure_clean(filename=None, return_filelike=False):
    """Gets a temporary path and agrees to remove on close.

    Parameters
    ----------
    filename : str (optional)
        if None, creates a temporary file which is then removed when out of
        scope. if passed, creates temporary file with filename as ending.
    return_filelike : bool (default False)
        if True, returns a file-like which is *always* cleaned. Necessary for
        savefig and other functions which want to append extensions.
    """
    filename = filename or ''
    fd = None

    if return_filelike:
        f = tempfile.TemporaryFile(suffix=filename)
        try:
            yield f
        finally:
            f.close()
    else:
        # don't generate tempfile if using a path with directory specified
        if len(os.path.dirname(filename)):
            raise ValueError("Can't pass a qualified name to ensure_clean()")

        try:
            fd, filename = tempfile.mkstemp(suffix=filename)
        except UnicodeEncodeError:
            import pytest
            pytest.skip('no unicode file names on this system')

        try:
            yield filename
        finally:
            try:
                os.close(fd)
            except Exception:
                print("Couldn't close file descriptor: {fdesc} (file: {fname})"
                      .format(fdesc=fd, fname=filename))
            try:
                if os.path.exists(filename):
                    os.remove(filename)
            except Exception as e:
                print("Exception on removing file: {error}".format(error=e))


@contextmanager
def ensure_clean_dir():
    """
    Get a temporary directory path and agrees to remove on close.

    Yields
    ------
    Temporary directory path
    """
    directory_name = tempfile.mkdtemp(suffix='')
    try:
        yield directory_name
    finally:
        try:
            rmtree(directory_name)
        except Exception:
            pass


@contextmanager
def ensure_safe_environment_variables():
    """
    Get a context manager to safely set environment variables

    All changes will be undone on close, hence environment variables set
    within this contextmanager will neither persist nor change global state.
    """
    saved_environ = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved_environ)


# -----------------------------------------------------------------------------
# Comparators


def equalContents(arr1, arr2):
    """Checks if the set of unique elements of arr1 and arr2 are equivalent.
    """
    return frozenset(arr1) == frozenset(arr2)


def assert_index_equal(left, right, exact='equiv', check_names=True,
                       check_less_precise=False, check_exact=True,
                       check_categorical=True, obj='Index'):
    """Check that left and right Index are equal.

    Parameters
    ----------
    left : Index
    right : Index
    exact : bool / string {'equiv'}, default 'equiv'
        Whether to check the Index class, dtype and inferred_type
        are identical. If 'equiv', then RangeIndex can be substituted for
        Int64Index as well.
    check_names : bool, default True
        Whether to check the names attribute.
    check_less_precise : bool or int, default False
        Specify comparison precision. Only used when check_exact is False.
        5 digits (False) or 3 digits (True) after decimal points are compared.
        If int, then specify the digits to compare
    check_exact : bool, default True
        Whether to compare number exactly.
    check_categorical : bool, default True
        Whether to compare internal Categorical exactly.
    obj : str, default 'Index'
        Specify object name being compared, internally used to show appropriate
        assertion message
    """
    __tracebackhide__ = True

    def _check_types(l, r, obj='Index'):
        if exact:
            assert_class_equal(l, r, exact=exact, obj=obj)

            # Skip exact dtype checking when `check_categorical` is False
            if check_categorical:
                assert_attr_equal('dtype', l, r, obj=obj)

            # allow string-like to have different inferred_types
            if l.inferred_type in ('string', 'unicode'):
                assert r.inferred_type in ('string', 'unicode')
            else:
                assert_attr_equal('inferred_type', l, r, obj=obj)

    def _get_ilevel_values(index, level):
        # accept level number only
        unique = index.levels[level]
        labels = index.codes[level]
        filled = take_1d(unique.values, labels, fill_value=unique._na_value)
        values = unique._shallow_copy(filled, name=index.names[level])
        return values

    # instance validation
    _check_isinstance(left, right, Index)

    # class / dtype comparison
    _check_types(left, right, obj=obj)

    # level comparison
    if left.nlevels != right.nlevels:
        msg1 = '{obj} levels are different'.format(obj=obj)
        msg2 = '{nlevels}, {left}'.format(nlevels=left.nlevels, left=left)
        msg3 = '{nlevels}, {right}'.format(nlevels=right.nlevels, right=right)
        raise_assert_detail(obj, msg1, msg2, msg3)

    # length comparison
    if len(left) != len(right):
        msg1 = '{obj} length are different'.format(obj=obj)
        msg2 = '{length}, {left}'.format(length=len(left), left=left)
        msg3 = '{length}, {right}'.format(length=len(right), right=right)
        raise_assert_detail(obj, msg1, msg2, msg3)

    # MultiIndex special comparison for little-friendly error messages
    if left.nlevels > 1:
        for level in range(left.nlevels):
            # cannot use get_level_values here because it can change dtype
            llevel = _get_ilevel_values(left, level)
            rlevel = _get_ilevel_values(right, level)

            lobj = 'MultiIndex level [{level}]'.format(level=level)
            assert_index_equal(llevel, rlevel,
                               exact=exact, check_names=check_names,
                               check_less_precise=check_less_precise,
                               check_exact=check_exact, obj=lobj)
            # get_level_values may change dtype
            _check_types(left.levels[level], right.levels[level], obj=obj)

    # skip exact index checking when `check_categorical` is False
    if check_exact and check_categorical:
        if not left.equals(right):
            diff = np.sum((left.values != right.values)
                          .astype(int)) * 100.0 / len(left)
            msg = '{obj} values are different ({pct} %)'.format(
                obj=obj, pct=np.round(diff, 5))
            raise_assert_detail(obj, msg, left, right)
    else:
        _testing.assert_almost_equal(left.values, right.values,
                                     check_less_precise=check_less_precise,
                                     check_dtype=exact,
                                     obj=obj, lobj=left, robj=right)

    # metadata comparison
    if check_names:
        assert_attr_equal('names', left, right, obj=obj)
    if isinstance(left, pd.PeriodIndex) or isinstance(right, pd.PeriodIndex):
        assert_attr_equal('freq', left, right, obj=obj)
    if (isinstance(left, pd.IntervalIndex) or
            isinstance(right, pd.IntervalIndex)):
        assert_interval_array_equal(left.values, right.values)

    if check_categorical:
        if is_categorical_dtype(left) or is_categorical_dtype(right):
            assert_categorical_equal(left.values, right.values,
                                     obj='{obj} category'.format(obj=obj))


def assert_class_equal(left, right, exact=True, obj='Input'):
    """checks classes are equal."""
    __tracebackhide__ = True

    def repr_class(x):
        if isinstance(x, Index):
            # return Index as it is to include values in the error message
            return x

        try:
            return x.__class__.__name__
        except AttributeError:
            return repr(type(x))

    if exact == 'equiv':
        if type(left) != type(right):
            # allow equivalence of Int64Index/RangeIndex
            types = {type(left).__name__, type(right).__name__}
            if len(types - {'Int64Index', 'RangeIndex'}):
                msg = '{obj} classes are not equivalent'.format(obj=obj)
                raise_assert_detail(obj, msg, repr_class(left),
                                    repr_class(right))
    elif exact:
        if type(left) != type(right):
            msg = '{obj} classes are different'.format(obj=obj)
            raise_assert_detail(obj, msg, repr_class(left),
                                repr_class(right))


def assert_attr_equal(attr, left, right, obj='Attributes'):
    """checks attributes are equal. Both objects must have attribute.

    Parameters
    ----------
    attr : str
        Attribute name being compared.
    left : object
    right : object
    obj : str, default 'Attributes'
        Specify object name being compared, internally used to show appropriate
        assertion message
    """
    __tracebackhide__ = True

    left_attr = getattr(left, attr)
    right_attr = getattr(right, attr)

    if left_attr is right_attr:
        return True
    elif (is_number(left_attr) and np.isnan(left_attr) and
          is_number(right_attr) and np.isnan(right_attr)):
        # np.nan
        return True

    try:
        result = left_attr == right_attr
    except TypeError:
        # datetimetz on rhs may raise TypeError
        result = False
    if not isinstance(result, bool):
        result = result.all()

    if result:
        return True
    else:
        msg = 'Attribute "{attr}" are different'.format(attr=attr)
        raise_assert_detail(obj, msg, left_attr, right_attr)


def assert_is_valid_plot_return_object(objs):
    import matplotlib.pyplot as plt
    if isinstance(objs, (pd.Series, np.ndarray)):
        for el in objs.ravel():
            msg = ("one of 'objs' is not a matplotlib Axes instance, type "
                   "encountered {name!r}").format(name=el.__class__.__name__)
            assert isinstance(el, (plt.Axes, dict)), msg
    else:
        assert isinstance(objs, (plt.Artist, tuple, dict)), (
            'objs is neither an ndarray of Artist instances nor a '
            'single Artist instance, tuple, or dict, "objs" is a {name!r}'
            .format(name=objs.__class__.__name__))


def isiterable(obj):
    return hasattr(obj, '__iter__')


def is_sorted(seq):
    if isinstance(seq, (Index, Series)):
        seq = seq.values
    # sorting does not change precisions
    return assert_numpy_array_equal(seq, np.sort(np.array(seq)))


def assert_categorical_equal(left, right, check_dtype=True,
                             check_category_order=True, obj='Categorical'):
    """Test that Categoricals are equivalent.

    Parameters
    ----------
    left : Categorical
    right : Categorical
    check_dtype : bool, default True
        Check that integer dtype of the codes are the same
    check_category_order : bool, default True
        Whether the order of the categories should be compared, which
        implies identical integer codes.  If False, only the resulting
        values are compared.  The ordered attribute is
        checked regardless.
    obj : str, default 'Categorical'
        Specify object name being compared, internally used to show appropriate
        assertion message
    """
    _check_isinstance(left, right, Categorical)

    if check_category_order:
        assert_index_equal(left.categories, right.categories,
                           obj='{obj}.categories'.format(obj=obj))
        assert_numpy_array_equal(left.codes, right.codes,
                                 check_dtype=check_dtype,
                                 obj='{obj}.codes'.format(obj=obj))
    else:
        assert_index_equal(left.categories.sort_values(),
                           right.categories.sort_values(),
                           obj='{obj}.categories'.format(obj=obj))
        assert_index_equal(left.categories.take(left.codes),
                           right.categories.take(right.codes),
                           obj='{obj}.values'.format(obj=obj))

    assert_attr_equal('ordered', left, right, obj=obj)


def assert_interval_array_equal(left, right, exact='equiv',
                                obj='IntervalArray'):
    """Test that two IntervalArrays are equivalent.

    Parameters
    ----------
    left, right : IntervalArray
        The IntervalArrays to compare.
    exact : bool / string {'equiv'}, default 'equiv'
        Whether to check the Index class, dtype and inferred_type
        are identical. If 'equiv', then RangeIndex can be substituted for
        Int64Index as well.
    obj : str, default 'IntervalArray'
        Specify object name being compared, internally used to show appropriate
        assertion message
    """
    _check_isinstance(left, right, IntervalArray)

    assert_index_equal(left.left, right.left, exact=exact,
                       obj='{obj}.left'.format(obj=obj))
    assert_index_equal(left.right, right.right, exact=exact,
                       obj='{obj}.left'.format(obj=obj))
    assert_attr_equal('closed', left, right, obj=obj)


def assert_period_array_equal(left, right, obj='PeriodArray'):
    _check_isinstance(left, right, PeriodArray)

    assert_numpy_array_equal(left._data, right._data,
                             obj='{obj}.values'.format(obj=obj))
    assert_attr_equal('freq', left, right, obj=obj)


def assert_datetime_array_equal(left, right, obj='DatetimeArray'):
    __tracebackhide__ = True
    _check_isinstance(left, right, DatetimeArray)

    assert_numpy_array_equal(left._data, right._data,
                             obj='{obj}._data'.format(obj=obj))
    assert_attr_equal('freq', left, right, obj=obj)
    assert_attr_equal('tz', left, right, obj=obj)


def assert_timedelta_array_equal(left, right, obj='TimedeltaArray'):
    __tracebackhide__ = True
    _check_isinstance(left, right, TimedeltaArray)
    assert_numpy_array_equal(left._data, right._data,
                             obj='{obj}._data'.format(obj=obj))
    assert_attr_equal('freq', left, right, obj=obj)


def raise_assert_detail(obj, message, left, right, diff=None):
    __tracebackhide__ = True

    if isinstance(left, np.ndarray):
        left = pprint_thing(left)
    elif is_categorical_dtype(left):
        left = repr(left)

    if PY2 and isinstance(left, string_types):
        # left needs to be printable in native text type in python2
        left = left.encode('utf-8')

    if isinstance(right, np.ndarray):
        right = pprint_thing(right)
    elif is_categorical_dtype(right):
        right = repr(right)

    if PY2 and isinstance(right, string_types):
        # right needs to be printable in native text type in python2
        right = right.encode('utf-8')

    msg = """{obj} are different

{message}
[left]:  {left}
[right]: {right}""".format(obj=obj, message=message, left=left, right=right)

    if diff is not None:
        msg += "\n[diff]: {diff}".format(diff=diff)

    raise AssertionError(msg)


def assert_numpy_array_equal(left, right, strict_nan=False,
                             check_dtype=True, err_msg=None,
                             check_same=None, obj='numpy array'):
    """ Checks that 'np.ndarray' is equivalent

    Parameters
    ----------
    left : np.ndarray or iterable
    right : np.ndarray or iterable
    strict_nan : bool, default False
        If True, consider NaN and None to be different.
    check_dtype: bool, default True
        check dtype if both a and b are np.ndarray
    err_msg : str, default None
        If provided, used as assertion message
    check_same : None|'copy'|'same', default None
        Ensure left and right refer/do not refer to the same memory area
    obj : str, default 'numpy array'
        Specify object name being compared, internally used to show appropriate
        assertion message
    """
    __tracebackhide__ = True

    # instance validation
    # Show a detailed error message when classes are different
    assert_class_equal(left, right, obj=obj)
    # both classes must be an np.ndarray
    _check_isinstance(left, right, np.ndarray)

    def _get_base(obj):
        return obj.base if getattr(obj, 'base', None) is not None else obj

    left_base = _get_base(left)
    right_base = _get_base(right)

    if check_same == 'same':
        if left_base is not right_base:
            msg = "{left!r} is not {right!r}".format(
                left=left_base, right=right_base)
            raise AssertionError(msg)
    elif check_same == 'copy':
        if left_base is right_base:
            msg = "{left!r} is {right!r}".format(
                left=left_base, right=right_base)
            raise AssertionError(msg)

    def _raise(left, right, err_msg):
        if err_msg is None:
            if left.shape != right.shape:
                raise_assert_detail(obj, '{obj} shapes are different'
                                    .format(obj=obj), left.shape, right.shape)

            diff = 0
            for l, r in zip(left, right):
                # count up differences
                if not array_equivalent(l, r, strict_nan=strict_nan):
                    diff += 1

            diff = diff * 100.0 / left.size
            msg = '{obj} values are different ({pct} %)'.format(
                obj=obj, pct=np.round(diff, 5))
            raise_assert_detail(obj, msg, left, right)

        raise AssertionError(err_msg)

    # compare shape and values
    if not array_equivalent(left, right, strict_nan=strict_nan):
        _raise(left, right, err_msg)

    if check_dtype:
        if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
            assert_attr_equal('dtype', left, right, obj=obj)

    return True


def assert_extension_array_equal(left, right, check_dtype=True,
                                 check_less_precise=False,
                                 check_exact=False):
    """Check that left and right ExtensionArrays are equal.

    Parameters
    ----------
    left, right : ExtensionArray
        The two arrays to compare
    check_dtype : bool, default True
        Whether to check if the ExtensionArray dtypes are identical.
    check_less_precise : bool or int, default False
        Specify comparison precision. Only used when check_exact is False.
        5 digits (False) or 3 digits (True) after decimal points are compared.
        If int, then specify the digits to compare.
    check_exact : bool, default False
        Whether to compare number exactly.

    Notes
    -----
    Missing values are checked separately from valid values.
    A mask of missing values is computed for each and checked to match.
    The remaining all-valid values are cast to object dtype and checked.
    """
    assert isinstance(left, ExtensionArray), 'left is not an ExtensionArray'
    assert isinstance(right, ExtensionArray), 'right is not an ExtensionArray'
    if check_dtype:
        assert_attr_equal('dtype', left, right, obj='ExtensionArray')

    if hasattr(left, "asi8") and type(right) == type(left):
        # Avoid slow object-dtype comparisons
        assert_numpy_array_equal(left.asi8, right.asi8)
        return

    left_na = np.asarray(left.isna())
    right_na = np.asarray(right.isna())
    assert_numpy_array_equal(left_na, right_na, obj='ExtensionArray NA mask')

    left_valid = np.asarray(left[~left_na].astype(object))
    right_valid = np.asarray(right[~right_na].astype(object))
    if check_exact:
        assert_numpy_array_equal(left_valid, right_valid, obj='ExtensionArray')
    else:
        _testing.assert_almost_equal(left_valid, right_valid,
                                     check_dtype=check_dtype,
                                     check_less_precise=check_less_precise,
                                     obj='ExtensionArray')


# This could be refactored to use the NDFrame.equals method
def assert_series_equal(left, right, check_dtype=True,
                        check_index_type='equiv',
                        check_series_type=True,
                        check_less_precise=False,
                        check_names=True,
                        check_exact=False,
                        check_datetimelike_compat=False,
                        check_categorical=True,
                        obj='Series'):
    """Check that left and right Series are equal.

    Parameters
    ----------
    left : Series
    right : Series
    check_dtype : bool, default True
        Whether to check the Series dtype is identical.
    check_index_type : bool / string {'equiv'}, default 'equiv'
        Whether to check the Index class, dtype and inferred_type
        are identical.
    check_series_type : bool, default True
        Whether to check the Series class is identical.
    check_less_precise : bool or int, default False
        Specify comparison precision. Only used when check_exact is False.
        5 digits (False) or 3 digits (True) after decimal points are compared.
        If int, then specify the digits to compare.
    check_names : bool, default True
        Whether to check the Series and Index names attribute.
    check_exact : bool, default False
        Whether to compare number exactly.
    check_datetimelike_compat : bool, default False
        Compare datetime-like which is comparable ignoring dtype.
    check_categorical : bool, default True
        Whether to compare internal Categorical exactly.
    obj : str, default 'Series'
        Specify object name being compared, internally used to show appropriate
        assertion message.
    """
    __tracebackhide__ = True

    # instance validation
    _check_isinstance(left, right, Series)

    if check_series_type:
        # ToDo: There are some tests using rhs is sparse
        # lhs is dense. Should use assert_class_equal in future
        assert isinstance(left, type(right))
        # assert_class_equal(left, right, obj=obj)

    # length comparison
    if len(left) != len(right):
        msg1 = '{len}, {left}'.format(len=len(left), left=left.index)
        msg2 = '{len}, {right}'.format(len=len(right), right=right.index)
        raise_assert_detail(obj, 'Series length are different', msg1, msg2)

    # index comparison
    assert_index_equal(left.index, right.index, exact=check_index_type,
                       check_names=check_names,
                       check_less_precise=check_less_precise,
                       check_exact=check_exact,
                       check_categorical=check_categorical,
                       obj='{obj}.index'.format(obj=obj))

    if check_dtype:
        # We want to skip exact dtype checking when `check_categorical`
        # is False. We'll still raise if only one is a `Categorical`,
        # regardless of `check_categorical`
        if (is_categorical_dtype(left) and is_categorical_dtype(right) and
                not check_categorical):
            pass
        else:
            assert_attr_equal('dtype', left, right)

    if check_exact:
        assert_numpy_array_equal(left.get_values(), right.get_values(),
                                 check_dtype=check_dtype,
                                 obj='{obj}'.format(obj=obj),)
    elif check_datetimelike_compat:
        # we want to check only if we have compat dtypes
        # e.g. integer and M|m are NOT compat, but we can simply check
        # the values in that case
        if (is_datetimelike_v_numeric(left, right) or
            is_datetimelike_v_object(left, right) or
            needs_i8_conversion(left) or
                needs_i8_conversion(right)):

            # datetimelike may have different objects (e.g. datetime.datetime
            # vs Timestamp) but will compare equal
            if not Index(left.values).equals(Index(right.values)):
                msg = ('[datetimelike_compat=True] {left} is not equal to '
                       '{right}.').format(left=left.values, right=right.values)
                raise AssertionError(msg)
        else:
            assert_numpy_array_equal(left.get_values(), right.get_values(),
                                     check_dtype=check_dtype)
    elif is_interval_dtype(left) or is_interval_dtype(right):
        assert_interval_array_equal(left.array, right.array)

    elif (is_extension_array_dtype(left.dtype) and
          is_datetime64tz_dtype(left.dtype)):
        # .values is an ndarray, but ._values is the ExtensionArray.
        # TODO: Use .array
        assert is_extension_array_dtype(right.dtype)
        return assert_extension_array_equal(left._values, right._values)

    elif (is_extension_array_dtype(left) and not is_categorical_dtype(left) and
          is_extension_array_dtype(right) and not is_categorical_dtype(right)):
        return assert_extension_array_equal(left.array, right.array)

    else:
        _testing.assert_almost_equal(left.get_values(), right.get_values(),
                                     check_less_precise=check_less_precise,
                                     check_dtype=check_dtype,
                                     obj='{obj}'.format(obj=obj))

    # metadata comparison
    if check_names:
        assert_attr_equal('name', left, right, obj=obj)

    if check_categorical:
        if is_categorical_dtype(left) or is_categorical_dtype(right):
            assert_categorical_equal(left.values, right.values,
                                     obj='{obj} category'.format(obj=obj))


# This could be refactored to use the NDFrame.equals method
def assert_frame_equal(left, right, check_dtype=True,
                       check_index_type='equiv',
                       check_column_type='equiv',
                       check_frame_type=True,
                       check_less_precise=False,
                       check_names=True,
                       by_blocks=False,
                       check_exact=False,
                       check_datetimelike_compat=False,
                       check_categorical=True,
                       check_like=False,
                       obj='DataFrame'):
    """
    Check that left and right DataFrame are equal.

    This function is intended to compare two DataFrames and output any
    differences. Is is mostly intended for use in unit tests.
    Additional parameters allow varying the strictness of the
    equality checks performed.

    Parameters
    ----------
    left : DataFrame
        First DataFrame to compare.
    right : DataFrame
        Second DataFrame to compare.
    check_dtype : bool, default True
        Whether to check the DataFrame dtype is identical.
    check_index_type : bool / string {'equiv'}, default 'equiv'
        Whether to check the Index class, dtype and inferred_type
        are identical.
    check_column_type : bool / string {'equiv'}, default 'equiv'
        Whether to check the columns class, dtype and inferred_type
        are identical. Is passed as the ``exact`` argument of
        :func:`assert_index_equal`.
    check_frame_type : bool, default True
        Whether to check the DataFrame class is identical.
    check_less_precise : bool or int, default False
        Specify comparison precision. Only used when check_exact is False.
        5 digits (False) or 3 digits (True) after decimal points are compared.
        If int, then specify the digits to compare.
    check_names : bool, default True
        Whether to check that the `names` attribute for both the `index`
        and `column` attributes of the DataFrame is identical, i.e.

        * left.index.names == right.index.names
        * left.columns.names == right.columns.names
    by_blocks : bool, default False
        Specify how to compare internal data. If False, compare by columns.
        If True, compare by blocks.
    check_exact : bool, default False
        Whether to compare number exactly.
    check_datetimelike_compat : bool, default False
        Compare datetime-like which is comparable ignoring dtype.
    check_categorical : bool, default True
        Whether to compare internal Categorical exactly.
    check_like : bool, default False
        If True, ignore the order of index & columns.
        Note: index labels must match their respective rows
        (same as in columns) - same labels must be with the same data.
    obj : str, default 'DataFrame'
        Specify object name being compared, internally used to show appropriate
        assertion message.

    See Also
    --------
    assert_series_equal : Equivalent method for asserting Series equality.
    DataFrame.equals : Check DataFrame equality.

    Examples
    --------
    This example shows comparing two DataFrames that are equal
    but with columns of differing dtypes.

    >>> from pandas.util.testing import assert_frame_equal
    >>> df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    >>> df2 = pd.DataFrame({'a': [1, 2], 'b': [3.0, 4.0]})

    df1 equals itself.
    >>> assert_frame_equal(df1, df1)

    df1 differs from df2 as column 'b' is of a different type.
    >>> assert_frame_equal(df1, df2)
    Traceback (most recent call last):
    AssertionError: Attributes are different

    Attribute "dtype" are different
    [left]:  int64
    [right]: float64

    Ignore differing dtypes in columns with check_dtype.
    >>> assert_frame_equal(df1, df2, check_dtype=False)
    """
    __tracebackhide__ = True

    # instance validation
    _check_isinstance(left, right, DataFrame)

    if check_frame_type:
        # ToDo: There are some tests using rhs is SparseDataFrame
        # lhs is DataFrame. Should use assert_class_equal in future
        assert isinstance(left, type(right))
        # assert_class_equal(left, right, obj=obj)

    # shape comparison
    if left.shape != right.shape:
        raise_assert_detail(obj,
                            'DataFrame shape mismatch',
                            '{shape!r}'.format(shape=left.shape),
                            '{shape!r}'.format(shape=right.shape))

    if check_like:
        left, right = left.reindex_like(right), right

    # index comparison
    assert_index_equal(left.index, right.index, exact=check_index_type,
                       check_names=check_names,
                       check_less_precise=check_less_precise,
                       check_exact=check_exact,
                       check_categorical=check_categorical,
                       obj='{obj}.index'.format(obj=obj))

    # column comparison
    assert_index_equal(left.columns, right.columns, exact=check_column_type,
                       check_names=check_names,
                       check_less_precise=check_less_precise,
                       check_exact=check_exact,
                       check_categorical=check_categorical,
                       obj='{obj}.columns'.format(obj=obj))

    # compare by blocks
    if by_blocks:
        rblocks = right._to_dict_of_blocks()
        lblocks = left._to_dict_of_blocks()
        for dtype in list(set(list(lblocks.keys()) + list(rblocks.keys()))):
            assert dtype in lblocks
            assert dtype in rblocks
            assert_frame_equal(lblocks[dtype], rblocks[dtype],
                               check_dtype=check_dtype, obj='DataFrame.blocks')

    # compare by columns
    else:
        for i, col in enumerate(left.columns):
            assert col in right
            lcol = left.iloc[:, i]
            rcol = right.iloc[:, i]
            assert_series_equal(
                lcol, rcol, check_dtype=check_dtype,
                check_index_type=check_index_type,
                check_less_precise=check_less_precise,
                check_exact=check_exact, check_names=check_names,
                check_datetimelike_compat=check_datetimelike_compat,
                check_categorical=check_categorical,
                obj='DataFrame.iloc[:, {idx}]'.format(idx=i))


def assert_panel_equal(left, right,
                       check_dtype=True,
                       check_panel_type=False,
                       check_less_precise=False,
                       check_names=False,
                       by_blocks=False,
                       obj='Panel'):
    """Check that left and right Panels are equal.

    Parameters
    ----------
    left : Panel (or nd)
    right : Panel (or nd)
    check_dtype : bool, default True
        Whether to check the Panel dtype is identical.
    check_panel_type : bool, default False
        Whether to check the Panel class is identical.
    check_less_precise : bool or int, default False
        Specify comparison precision. Only used when check_exact is False.
        5 digits (False) or 3 digits (True) after decimal points are compared.
        If int, then specify the digits to compare
    check_names : bool, default True
        Whether to check the Index names attribute.
    by_blocks : bool, default False
        Specify how to compare internal data. If False, compare by columns.
        If True, compare by blocks.
    obj : str, default 'Panel'
        Specify the object name being compared, internally used to show
        the appropriate assertion message.
    """

    if check_panel_type:
        assert_class_equal(left, right, obj=obj)

    for axis in left._AXIS_ORDERS:
        left_ind = getattr(left, axis)
        right_ind = getattr(right, axis)
        assert_index_equal(left_ind, right_ind, check_names=check_names)

    if by_blocks:
        rblocks = right._to_dict_of_blocks()
        lblocks = left._to_dict_of_blocks()
        for dtype in list(set(list(lblocks.keys()) + list(rblocks.keys()))):
            assert dtype in lblocks
            assert dtype in rblocks
            array_equivalent(lblocks[dtype].values, rblocks[dtype].values)
    else:

        # can potentially be slow
        for i, item in enumerate(left._get_axis(0)):
            msg = "non-matching item (right) '{item}'".format(item=item)
            assert item in right, msg
            litem = left.iloc[i]
            ritem = right.iloc[i]
            assert_frame_equal(litem, ritem,
                               check_less_precise=check_less_precise,
                               check_names=check_names)

        for i, item in enumerate(right._get_axis(0)):
            msg = "non-matching item (left) '{item}'".format(item=item)
            assert item in left, msg


def assert_equal(left, right, **kwargs):
    """
    Wrapper for tm.assert_*_equal to dispatch to the appropriate test function.

    Parameters
    ----------
    left : Index, Series, DataFrame, ExtensionArray, or np.ndarray
    right : Index, Series, DataFrame, ExtensionArray, or np.ndarray
    **kwargs
    """
    __tracebackhide__ = True

    if isinstance(left, pd.Index):
        assert_index_equal(left, right, **kwargs)
    elif isinstance(left, pd.Series):
        assert_series_equal(left, right, **kwargs)
    elif isinstance(left, pd.DataFrame):
        assert_frame_equal(left, right, **kwargs)
    elif isinstance(left, IntervalArray):
        assert_interval_array_equal(left, right, **kwargs)
    elif isinstance(left, PeriodArray):
        assert_period_array_equal(left, right, **kwargs)
    elif isinstance(left, DatetimeArray):
        assert_datetime_array_equal(left, right, **kwargs)
    elif isinstance(left, TimedeltaArray):
        assert_timedelta_array_equal(left, right, **kwargs)
    elif isinstance(left, ExtensionArray):
        assert_extension_array_equal(left, right, **kwargs)
    elif isinstance(left, np.ndarray):
        assert_numpy_array_equal(left, right, **kwargs)
    else:
        raise NotImplementedError(type(left))


def box_expected(expected, box_cls, transpose=True):
    """
    Helper function to wrap the expected output of a test in a given box_class.

    Parameters
    ----------
    expected : np.ndarray, Index, Series
    box_cls : {Index, Series, DataFrame}

    Returns
    -------
    subclass of box_cls
    """
    if box_cls is pd.Index:
        expected = pd.Index(expected)
    elif box_cls is pd.Series:
        expected = pd.Series(expected)
    elif box_cls is pd.DataFrame:
        expected = pd.Series(expected).to_frame()
        if transpose:
            # for vector operations, we we need a DataFrame to be a single-row,
            #  not a single-column, in order to operate against non-DataFrame
            #  vectors of the same length.
            expected = expected.T
    elif box_cls is PeriodArray:
        # the PeriodArray constructor is not as flexible as period_array
        expected = period_array(expected)
    elif box_cls is DatetimeArray:
        expected = DatetimeArray(expected)
    elif box_cls is TimedeltaArray:
        expected = TimedeltaArray(expected)
    elif box_cls is np.ndarray:
        expected = np.array(expected)
    elif box_cls is to_array:
        expected = to_array(expected)
    else:
        raise NotImplementedError(box_cls)
    return expected


def to_array(obj):
    # temporary implementation until we get pd.array in place
    if is_period_dtype(obj):
        return period_array(obj)
    elif is_datetime64_dtype(obj) or is_datetime64tz_dtype(obj):
        return DatetimeArray._from_sequence(obj)
    elif is_timedelta64_dtype(obj):
        return TimedeltaArray._from_sequence(obj)
    else:
        return np.array(obj)


# -----------------------------------------------------------------------------
# Sparse


def assert_sp_array_equal(left, right, check_dtype=True, check_kind=True,
                          check_fill_value=True,
                          consolidate_block_indices=False):
    """Check that the left and right SparseArray are equal.

    Parameters
    ----------
    left : SparseArray
    right : SparseArray
    check_dtype : bool, default True
        Whether to check the data dtype is identical.
    check_kind : bool, default True
        Whether to just the kind of the sparse index for each column.
    check_fill_value : bool, default True
        Whether to check that left.fill_value matches right.fill_value
    consolidate_block_indices : bool, default False
        Whether to consolidate contiguous blocks for sparse arrays with
        a BlockIndex. Some operations, e.g. concat, will end up with
        block indices that could be consolidated. Setting this to true will
        create a new BlockIndex for that array, with consolidated
        block indices.
    """

    _check_isinstance(left, right, pd.SparseArray)

    assert_numpy_array_equal(left.sp_values, right.sp_values,
                             check_dtype=check_dtype)

    # SparseIndex comparison
    assert isinstance(left.sp_index, pd._libs.sparse.SparseIndex)
    assert isinstance(right.sp_index, pd._libs.sparse.SparseIndex)

    if not check_kind:
        left_index = left.sp_index.to_block_index()
        right_index = right.sp_index.to_block_index()
    else:
        left_index = left.sp_index
        right_index = right.sp_index

    if consolidate_block_indices and left.kind == 'block':
        # we'll probably remove this hack...
        left_index = left_index.to_int_index().to_block_index()
        right_index = right_index.to_int_index().to_block_index()

    if not left_index.equals(right_index):
        raise_assert_detail('SparseArray.index', 'index are not equal',
                            left_index, right_index)
    else:
        # Just ensure a
        pass

    if check_fill_value:
        assert_attr_equal('fill_value', left, right)
    if check_dtype:
        assert_attr_equal('dtype', left, right)
    assert_numpy_array_equal(left.values, right.values,
                             check_dtype=check_dtype)


def assert_sp_series_equal(left, right, check_dtype=True, exact_indices=True,
                           check_series_type=True, check_names=True,
                           check_kind=True,
                           check_fill_value=True,
                           consolidate_block_indices=False,
                           obj='SparseSeries'):
    """Check that the left and right SparseSeries are equal.

    Parameters
    ----------
    left : SparseSeries
    right : SparseSeries
    check_dtype : bool, default True
        Whether to check the Series dtype is identical.
    exact_indices : bool, default True
    check_series_type : bool, default True
        Whether to check the SparseSeries class is identical.
    check_names : bool, default True
        Whether to check the SparseSeries name attribute.
    check_kind : bool, default True
        Whether to just the kind of the sparse index for each column.
    check_fill_value : bool, default True
        Whether to check that left.fill_value matches right.fill_value
    consolidate_block_indices : bool, default False
        Whether to consolidate contiguous blocks for sparse arrays with
        a BlockIndex. Some operations, e.g. concat, will end up with
        block indices that could be consolidated. Setting this to true will
        create a new BlockIndex for that array, with consolidated
        block indices.
    obj : str, default 'SparseSeries'
        Specify the object name being compared, internally used to show
        the appropriate assertion message.
    """
    _check_isinstance(left, right, pd.SparseSeries)

    if check_series_type:
        assert_class_equal(left, right, obj=obj)

    assert_index_equal(left.index, right.index,
                       obj='{obj}.index'.format(obj=obj))

    assert_sp_array_equal(left.values, right.values,
                          check_kind=check_kind,
                          check_fill_value=check_fill_value,
                          consolidate_block_indices=consolidate_block_indices)

    if check_names:
        assert_attr_equal('name', left, right)
    if check_dtype:
        assert_attr_equal('dtype', left, right)

    assert_numpy_array_equal(np.asarray(left.values),
                             np.asarray(right.values))


def assert_sp_frame_equal(left, right, check_dtype=True, exact_indices=True,
                          check_frame_type=True, check_kind=True,
                          check_fill_value=True,
                          consolidate_block_indices=False,
                          obj='SparseDataFrame'):
    """Check that the left and right SparseDataFrame are equal.

    Parameters
    ----------
    left : SparseDataFrame
    right : SparseDataFrame
    check_dtype : bool, default True
        Whether to check the Series dtype is identical.
    exact_indices : bool, default True
        SparseSeries SparseIndex objects must be exactly the same,
        otherwise just compare dense representations.
    check_frame_type : bool, default True
        Whether to check the SparseDataFrame class is identical.
    check_kind : bool, default True
        Whether to just the kind of the sparse index for each column.
    check_fill_value : bool, default True
        Whether to check that left.fill_value matches right.fill_value
    consolidate_block_indices : bool, default False
        Whether to consolidate contiguous blocks for sparse arrays with
        a BlockIndex. Some operations, e.g. concat, will end up with
        block indices that could be consolidated. Setting this to true will
        create a new BlockIndex for that array, with consolidated
        block indices.
    obj : str, default 'SparseDataFrame'
        Specify the object name being compared, internally used to show
        the appropriate assertion message.
    """
    _check_isinstance(left, right, pd.SparseDataFrame)

    if check_frame_type:
        assert_class_equal(left, right, obj=obj)

    assert_index_equal(left.index, right.index,
                       obj='{obj}.index'.format(obj=obj))
    assert_index_equal(left.columns, right.columns,
                       obj='{obj}.columns'.format(obj=obj))

    if check_fill_value:
        assert_attr_equal('default_fill_value', left, right, obj=obj)

    for col, series in compat.iteritems(left):
        assert (col in right)
        # trade-off?

        if exact_indices:
            assert_sp_series_equal(
                series, right[col],
                check_dtype=check_dtype,
                check_kind=check_kind,
                check_fill_value=check_fill_value,
                consolidate_block_indices=consolidate_block_indices
            )
        else:
            assert_series_equal(series.to_dense(), right[col].to_dense(),
                                check_dtype=check_dtype)

    # do I care?
    # assert(left.default_kind == right.default_kind)

    for col in right:
        assert (col in left)

# -----------------------------------------------------------------------------
# Others


def assert_contains_all(iterable, dic):
    for k in iterable:
        assert k in dic, "Did not contain item: '{key!r}'".format(key=k)


def assert_copy(iter1, iter2, **eql_kwargs):
    """
    iter1, iter2: iterables that produce elements
    comparable with assert_almost_equal

    Checks that the elements are equal, but not
    the same object. (Does not check that items
    in sequences are also not the same object)
    """
    for elem1, elem2 in zip(iter1, iter2):
        assert_almost_equal(elem1, elem2, **eql_kwargs)
        msg = ("Expected object {obj1!r} and object {obj2!r} to be "
               "different objects, but they were the same object."
               ).format(obj1=type(elem1), obj2=type(elem2))
        assert elem1 is not elem2, msg


def getCols(k):
    return string.ascii_uppercase[:k]


# make index
def makeStringIndex(k=10, name=None):
    return Index(rands_array(nchars=10, size=k), name=name)


def makeUnicodeIndex(k=10, name=None):
    return Index(randu_array(nchars=10, size=k), name=name)


def makeCategoricalIndex(k=10, n=3, name=None, **kwargs):
    """ make a length k index or n categories """
    x = rands_array(nchars=4, size=n)
    return CategoricalIndex(np.random.choice(x, k), name=name, **kwargs)


def makeIntervalIndex(k=10, name=None, **kwargs):
    """ make a length k IntervalIndex """
    x = np.linspace(0, 100, num=(k + 1))
    return IntervalIndex.from_breaks(x, name=name, **kwargs)


def makeBoolIndex(k=10, name=None):
    if k == 1:
        return Index([True], name=name)
    elif k == 2:
        return Index([False, True], name=name)
    return Index([False, True] + [False] * (k - 2), name=name)


def makeIntIndex(k=10, name=None):
    return Index(lrange(k), name=name)


def makeUIntIndex(k=10, name=None):
    return Index([2**63 + i for i in lrange(k)], name=name)


def makeRangeIndex(k=10, name=None, **kwargs):
    return RangeIndex(0, k, 1, name=name, **kwargs)


def makeFloatIndex(k=10, name=None):
    values = sorted(np.random.random_sample(k)) - np.random.random_sample(1)
    return Index(values * (10 ** np.random.randint(0, 9)), name=name)


def makeDateIndex(k=10, freq='B', name=None, **kwargs):
    dt = datetime(2000, 1, 1)
    dr = bdate_range(dt, periods=k, freq=freq, name=name)
    return DatetimeIndex(dr, name=name, **kwargs)


def makeTimedeltaIndex(k=10, freq='D', name=None, **kwargs):
    return pd.timedelta_range(start='1 day', periods=k, freq=freq,
                              name=name, **kwargs)


def makePeriodIndex(k=10, name=None, **kwargs):
    dt = datetime(2000, 1, 1)
    dr = pd.period_range(start=dt, periods=k, freq='B', name=name, **kwargs)
    return dr


def makeMultiIndex(k=10, names=None, **kwargs):
    return MultiIndex.from_product(
        (('foo', 'bar'), (1, 2)), names=names, **kwargs)


def all_index_generator(k=10):
    """Generator which can be iterated over to get instances of all the various
    index classes.

    Parameters
    ----------
    k: length of each of the index instances
    """
    all_make_index_funcs = [makeIntIndex, makeFloatIndex, makeStringIndex,
                            makeUnicodeIndex, makeDateIndex, makePeriodIndex,
                            makeTimedeltaIndex, makeBoolIndex, makeRangeIndex,
                            makeIntervalIndex,
                            makeCategoricalIndex]
    for make_index_func in all_make_index_funcs:
        yield make_index_func(k=k)


def index_subclass_makers_generator():
    make_index_funcs = [
        makeDateIndex, makePeriodIndex,
        makeTimedeltaIndex, makeRangeIndex,
        makeIntervalIndex, makeCategoricalIndex,
        makeMultiIndex
    ]
    for make_index_func in make_index_funcs:
        yield make_index_func


def all_timeseries_index_generator(k=10):
    """Generator which can be iterated over to get instances of all the classes
    which represent time-seires.

    Parameters
    ----------
    k: length of each of the index instances
    """
    make_index_funcs = [makeDateIndex, makePeriodIndex, makeTimedeltaIndex]
    for make_index_func in make_index_funcs:
        yield make_index_func(k=k)


# make series
def makeFloatSeries(name=None):
    index = makeStringIndex(N)
    return Series(randn(N), index=index, name=name)


def makeStringSeries(name=None):
    index = makeStringIndex(N)
    return Series(randn(N), index=index, name=name)


def makeObjectSeries(name=None):
    dateIndex = makeDateIndex(N)
    dateIndex = Index(dateIndex, dtype=object)
    index = makeStringIndex(N)
    return Series(dateIndex, index=index, name=name)


def getSeriesData():
    index = makeStringIndex(N)
    return {c: Series(randn(N), index=index) for c in getCols(K)}


def makeTimeSeries(nper=None, freq='B', name=None):
    if nper is None:
        nper = N
    return Series(randn(nper), index=makeDateIndex(nper, freq=freq), name=name)


def makePeriodSeries(nper=None, name=None):
    if nper is None:
        nper = N
    return Series(randn(nper), index=makePeriodIndex(nper), name=name)


def getTimeSeriesData(nper=None, freq='B'):
    return {c: makeTimeSeries(nper, freq) for c in getCols(K)}


def getPeriodData(nper=None):
    return {c: makePeriodSeries(nper) for c in getCols(K)}


# make frame
def makeTimeDataFrame(nper=None, freq='B'):
    data = getTimeSeriesData(nper, freq)
    return DataFrame(data)


def makeDataFrame():
    data = getSeriesData()
    return DataFrame(data)


def getMixedTypeDict():
    index = Index(['a', 'b', 'c', 'd', 'e'])

    data = {
        'A': [0., 1., 2., 3., 4.],
        'B': [0., 1., 0., 1., 0.],
        'C': ['foo1', 'foo2', 'foo3', 'foo4', 'foo5'],
        'D': bdate_range('1/1/2009', periods=5)
    }

    return index, data


def makeMixedDataFrame():
    return DataFrame(getMixedTypeDict()[1])


def makePeriodFrame(nper=None):
    data = getPeriodData(nper)
    return DataFrame(data)


def makePanel(nper=None):
    with warnings.catch_warnings(record=True):
        warnings.filterwarnings("ignore", "\\nPanel", FutureWarning)
        cols = ['Item' + c for c in string.ascii_uppercase[:K - 1]]
        data = {c: makeTimeDataFrame(nper) for c in cols}
        return Panel.fromDict(data)


def makePeriodPanel(nper=None):
    with warnings.catch_warnings(record=True):
        warnings.filterwarnings("ignore", "\\nPanel", FutureWarning)
        cols = ['Item' + c for c in string.ascii_uppercase[:K - 1]]
        data = {c: makePeriodFrame(nper) for c in cols}
        return Panel.fromDict(data)


def makeCustomIndex(nentries, nlevels, prefix='#', names=False, ndupe_l=None,
                    idx_type=None):
    """Create an index/multindex with given dimensions, levels, names, etc'

    nentries - number of entries in index
    nlevels - number of levels (> 1 produces multindex)
    prefix - a string prefix for labels
    names - (Optional), bool or list of strings. if True will use default
       names, if false will use no names, if a list is given, the name of
       each level in the index will be taken from the list.
    ndupe_l - (Optional), list of ints, the number of rows for which the
       label will repeated at the corresponding level, you can specify just
       the first few, the rest will use the default ndupe_l of 1.
       len(ndupe_l) <= nlevels.
    idx_type - "i"/"f"/"s"/"u"/"dt"/"p"/"td".
       If idx_type is not None, `idx_nlevels` must be 1.
       "i"/"f" creates an integer/float index,
       "s"/"u" creates a string/unicode index
       "dt" create a datetime index.
       "td" create a datetime index.

        if unspecified, string labels will be generated.
    """

    if ndupe_l is None:
        ndupe_l = [1] * nlevels
    assert (is_sequence(ndupe_l) and len(ndupe_l) <= nlevels)
    assert (names is None or names is False or
            names is True or len(names) is nlevels)
    assert idx_type is None or (idx_type in ('i', 'f', 's', 'u',
                                             'dt', 'p', 'td')
                                and nlevels == 1)

    if names is True:
        # build default names
        names = [prefix + str(i) for i in range(nlevels)]
    if names is False:
        # pass None to index constructor for no name
        names = None

    # make singelton case uniform
    if isinstance(names, compat.string_types) and nlevels == 1:
        names = [names]

    # specific 1D index type requested?
    idx_func = dict(i=makeIntIndex, f=makeFloatIndex,
                    s=makeStringIndex, u=makeUnicodeIndex,
                    dt=makeDateIndex, td=makeTimedeltaIndex,
                    p=makePeriodIndex).get(idx_type)
    if idx_func:
        idx = idx_func(nentries)
        # but we need to fill in the name
        if names:
            idx.name = names[0]
        return idx
    elif idx_type is not None:
        raise ValueError('"{idx_type}" is not a legal value for `idx_type`, '
                         'use  "i"/"f"/"s"/"u"/"dt/"p"/"td".'
                         .format(idx_type=idx_type))

    if len(ndupe_l) < nlevels:
        ndupe_l.extend([1] * (nlevels - len(ndupe_l)))
    assert len(ndupe_l) == nlevels

    assert all(x > 0 for x in ndupe_l)

    tuples = []
    for i in range(nlevels):
        def keyfunc(x):
            import re
            numeric_tuple = re.sub(r"[^\d_]_?", "", x).split("_")
            return lmap(int, numeric_tuple)

        # build a list of lists to create the index from
        div_factor = nentries // ndupe_l[i] + 1
        cnt = Counter()
        for j in range(div_factor):
            label = '{prefix}_l{i}_g{j}'.format(prefix=prefix, i=i, j=j)
            cnt[label] = ndupe_l[i]
        # cute Counter trick
        result = list(sorted(cnt.elements(), key=keyfunc))[:nentries]
        tuples.append(result)

    tuples = lzip(*tuples)

    # convert tuples to index
    if nentries == 1:
        # we have a single level of tuples, i.e. a regular Index
        index = Index(tuples[0], name=names[0])
    elif nlevels == 1:
        name = None if names is None else names[0]
        index = Index((x[0] for x in tuples), name=name)
    else:
        index = MultiIndex.from_tuples(tuples, names=names)
    return index


def makeCustomDataframe(nrows, ncols, c_idx_names=True, r_idx_names=True,
                        c_idx_nlevels=1, r_idx_nlevels=1, data_gen_f=None,
                        c_ndupe_l=None, r_ndupe_l=None, dtype=None,
                        c_idx_type=None, r_idx_type=None):
    """
   nrows,  ncols - number of data rows/cols
   c_idx_names, idx_names  - False/True/list of strings,  yields No names ,
        default names or uses the provided names for the levels of the
        corresponding index. You can provide a single string when
        c_idx_nlevels ==1.
   c_idx_nlevels - number of levels in columns index. > 1 will yield MultiIndex
   r_idx_nlevels - number of levels in rows index. > 1 will yield MultiIndex
   data_gen_f - a function f(row,col) which return the data value
        at that position, the default generator used yields values of the form
        "RxCy" based on position.
   c_ndupe_l, r_ndupe_l - list of integers, determines the number
        of duplicates for each label at a given level of the corresponding
        index. The default `None` value produces a multiplicity of 1 across
        all levels, i.e. a unique index. Will accept a partial list of length
        N < idx_nlevels, for just the first N levels. If ndupe doesn't divide
        nrows/ncol, the last label might have lower multiplicity.
   dtype - passed to the DataFrame constructor as is, in case you wish to
        have more control in conjuncion with a custom `data_gen_f`
   r_idx_type, c_idx_type -  "i"/"f"/"s"/"u"/"dt"/"td".
       If idx_type is not None, `idx_nlevels` must be 1.
       "i"/"f" creates an integer/float index,
       "s"/"u" creates a string/unicode index
       "dt" create a datetime index.
       "td" create a timedelta index.

        if unspecified, string labels will be generated.

    Examples:

    # 5 row, 3 columns, default names on both, single index on both axis
    >> makeCustomDataframe(5,3)

    # make the data a random int between 1 and 100
    >> mkdf(5,3,data_gen_f=lambda r,c:randint(1,100))

    # 2-level multiindex on rows with each label duplicated
    # twice on first level, default names on both axis, single
    # index on both axis
    >> a=makeCustomDataframe(5,3,r_idx_nlevels=2,r_ndupe_l=[2])

    # DatetimeIndex on row, index with unicode labels on columns
    # no names on either axis
    >> a=makeCustomDataframe(5,3,c_idx_names=False,r_idx_names=False,
                             r_idx_type="dt",c_idx_type="u")

    # 4-level multindex on rows with names provided, 2-level multindex
    # on columns with default labels and default names.
    >> a=makeCustomDataframe(5,3,r_idx_nlevels=4,
                             r_idx_names=["FEE","FI","FO","FAM"],
                             c_idx_nlevels=2)

    >> a=mkdf(5,3,r_idx_nlevels=2,c_idx_nlevels=4)
    """

    assert c_idx_nlevels > 0
    assert r_idx_nlevels > 0
    assert r_idx_type is None or (r_idx_type in ('i', 'f', 's',
                                                 'u', 'dt', 'p', 'td')
                                  and r_idx_nlevels == 1)
    assert c_idx_type is None or (c_idx_type in ('i', 'f', 's',
                                                 'u', 'dt', 'p', 'td')
                                  and c_idx_nlevels == 1)

    columns = makeCustomIndex(ncols, nlevels=c_idx_nlevels, prefix='C',
                              names=c_idx_names, ndupe_l=c_ndupe_l,
                              idx_type=c_idx_type)
    index = makeCustomIndex(nrows, nlevels=r_idx_nlevels, prefix='R',
                            names=r_idx_names, ndupe_l=r_ndupe_l,
                            idx_type=r_idx_type)

    # by default, generate data based on location
    if data_gen_f is None:
        data_gen_f = lambda r, c: "R{rows}C{cols}".format(rows=r, cols=c)

    data = [[data_gen_f(r, c) for c in range(ncols)] for r in range(nrows)]

    return DataFrame(data, index, columns, dtype=dtype)


def _create_missing_idx(nrows, ncols, density, random_state=None):
    if random_state is None:
        random_state = np.random
    else:
        random_state = np.random.RandomState(random_state)

    # below is cribbed from scipy.sparse
    size = int(np.round((1 - density) * nrows * ncols))
    # generate a few more to ensure unique values
    min_rows = 5
    fac = 1.02
    extra_size = min(size + min_rows, fac * size)

    def _gen_unique_rand(rng, _extra_size):
        ind = rng.rand(int(_extra_size))
        return np.unique(np.floor(ind * nrows * ncols))[:size]

    ind = _gen_unique_rand(random_state, extra_size)
    while ind.size < size:
        extra_size *= 1.05
        ind = _gen_unique_rand(random_state, extra_size)

    j = np.floor(ind * 1. / nrows).astype(int)
    i = (ind - j * nrows).astype(int)
    return i.tolist(), j.tolist()


def makeMissingCustomDataframe(nrows, ncols, density=.9, random_state=None,
                               c_idx_names=True, r_idx_names=True,
                               c_idx_nlevels=1, r_idx_nlevels=1,
                               data_gen_f=None,
                               c_ndupe_l=None, r_ndupe_l=None, dtype=None,
                               c_idx_type=None, r_idx_type=None):
    """
    Parameters
    ----------
    Density : float, optional
        Float in (0, 1) that gives the percentage of non-missing numbers in
        the DataFrame.
    random_state : {np.random.RandomState, int}, optional
        Random number generator or random seed.

    See makeCustomDataframe for descriptions of the rest of the parameters.
    """
    df = makeCustomDataframe(nrows, ncols, c_idx_names=c_idx_names,
                             r_idx_names=r_idx_names,
                             c_idx_nlevels=c_idx_nlevels,
                             r_idx_nlevels=r_idx_nlevels,
                             data_gen_f=data_gen_f,
                             c_ndupe_l=c_ndupe_l, r_ndupe_l=r_ndupe_l,
                             dtype=dtype, c_idx_type=c_idx_type,
                             r_idx_type=r_idx_type)

    i, j = _create_missing_idx(nrows, ncols, density, random_state)
    df.values[i, j] = np.nan
    return df


def makeMissingDataframe(density=.9, random_state=None):
    df = makeDataFrame()
    i, j = _create_missing_idx(*df.shape, density=density,
                               random_state=random_state)
    df.values[i, j] = np.nan
    return df


def add_nans(panel):
    I, J, N = panel.shape
    for i, item in enumerate(panel.items):
        dm = panel[item]
        for j, col in enumerate(dm.columns):
            dm[col][:i + j] = np.NaN
    return panel


class TestSubDict(dict):

    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)


def optional_args(decorator):
    """allows a decorator to take optional positional and keyword arguments.
    Assumes that taking a single, callable, positional argument means that
    it is decorating a function, i.e. something like this::

        @my_decorator
        def function(): pass

    Calls decorator with decorator(f, *args, **kwargs)"""

    @wraps(decorator)
    def wrapper(*args, **kwargs):
        def dec(f):
            return decorator(f, *args, **kwargs)

        is_decorating = not kwargs and len(args) == 1 and callable(args[0])
        if is_decorating:
            f = args[0]
            args = []
            return dec(f)
        else:
            return dec

    return wrapper


# skip tests on exceptions with this message
_network_error_messages = (
    # 'urlopen error timed out',
    # 'timeout: timed out',
    # 'socket.timeout: timed out',
    'timed out',
    'Server Hangup',
    'HTTP Error 503: Service Unavailable',
    '502: Proxy Error',
    'HTTP Error 502: internal error',
    'HTTP Error 502',
    'HTTP Error 503',
    'HTTP Error 403',
    'HTTP Error 400',
    'Temporary failure in name resolution',
    'Name or service not known',
    'Connection refused',
    'certificate verify',
)

# or this e.errno/e.reason.errno
_network_errno_vals = (
    101,  # Network is unreachable
    111,  # Connection refused
    110,  # Connection timed out
    104,  # Connection reset Error
    54,   # Connection reset by peer
    60,   # urllib.error.URLError: [Errno 60] Connection timed out
)

# Both of the above shouldn't mask real issues such as 404's
# or refused connections (changed DNS).
# But some tests (test_data yahoo) contact incredibly flakey
# servers.

# and conditionally raise on these exception types
_network_error_classes = (IOError, httplib.HTTPException)

if PY3:
    _network_error_classes += (TimeoutError,)  # noqa


def can_connect(url, error_classes=_network_error_classes):
    """Try to connect to the given url. True if succeeds, False if IOError
    raised

    Parameters
    ----------
    url : basestring
        The URL to try to connect to

    Returns
    -------
    connectable : bool
        Return True if no IOError (unable to connect) or URLError (bad url) was
        raised
    """
    try:
        with urlopen(url):
            pass
    except error_classes:
        return False
    else:
        return True


@optional_args
def network(t, url="http://www.google.com",
            raise_on_error=_RAISE_NETWORK_ERROR_DEFAULT,
            check_before_test=False,
            error_classes=_network_error_classes,
            skip_errnos=_network_errno_vals,
            _skip_on_messages=_network_error_messages,
            ):
    """
    Label a test as requiring network connection and, if an error is
    encountered, only raise if it does not find a network connection.

    In comparison to ``network``, this assumes an added contract to your test:
    you must assert that, under normal conditions, your test will ONLY fail if
    it does not have network connectivity.

    You can call this in 3 ways: as a standard decorator, with keyword
    arguments, or with a positional argument that is the url to check.

    Parameters
    ----------
    t : callable
        The test requiring network connectivity.
    url : path
        The url to test via ``pandas.io.common.urlopen`` to check
        for connectivity. Defaults to 'http://www.google.com'.
    raise_on_error : bool
        If True, never catches errors.
    check_before_test : bool
        If True, checks connectivity before running the test case.
    error_classes : tuple or Exception
        error classes to ignore. If not in ``error_classes``, raises the error.
        defaults to IOError. Be careful about changing the error classes here.
    skip_errnos : iterable of int
        Any exception that has .errno or .reason.erno set to one
        of these values will be skipped with an appropriate
        message.
    _skip_on_messages: iterable of string
        any exception e for which one of the strings is
        a substring of str(e) will be skipped with an appropriate
        message. Intended to suppress errors where an errno isn't available.

    Notes
    -----
    * ``raise_on_error`` supercedes ``check_before_test``

    Returns
    -------
    t : callable
        The decorated test ``t``, with checks for connectivity errors.

    Example
    -------

    Tests decorated with @network will fail if it's possible to make a network
    connection to another URL (defaults to google.com)::

      >>> from pandas.util.testing import network
      >>> from pandas.io.common import urlopen
      >>> @network
      ... def test_network():
      ...     with urlopen("rabbit://bonanza.com"):
      ...         pass
      Traceback
         ...
      URLError: <urlopen error unknown url type: rabit>

      You can specify alternative URLs::

        >>> @network("http://www.yahoo.com")
        ... def test_something_with_yahoo():
        ...    raise IOError("Failure Message")
        >>> test_something_with_yahoo()
        Traceback (most recent call last):
            ...
        IOError: Failure Message

    If you set check_before_test, it will check the url first and not run the
    test on failure::

        >>> @network("failing://url.blaher", check_before_test=True)
        ... def test_something():
        ...     print("I ran!")
        ...     raise ValueError("Failure")
        >>> test_something()
        Traceback (most recent call last):
            ...

    Errors not related to networking will always be raised.
    """
    from pytest import skip
    t.network = True

    @compat.wraps(t)
    def wrapper(*args, **kwargs):
        if check_before_test and not raise_on_error:
            if not can_connect(url, error_classes):
                skip()
        try:
            return t(*args, **kwargs)
        except Exception as e:
            errno = getattr(e, 'errno', None)
            if not errno and hasattr(errno, "reason"):
                errno = getattr(e.reason, 'errno', None)

            if errno in skip_errnos:
                skip("Skipping test due to known errno"
                     " and error {error}".format(error=e))

            try:
                e_str = traceback.format_exc(e)
            except Exception:
                e_str = str(e)

            if any(m.lower() in e_str.lower() for m in _skip_on_messages):
                skip("Skipping test because exception "
                     "message is known and error {error}".format(error=e))

            if not isinstance(e, error_classes):
                raise

            if raise_on_error or can_connect(url, error_classes):
                raise
            else:
                skip("Skipping test due to lack of connectivity"
                     " and error {error}".format(error=e))

    return wrapper


with_connectivity_check = network


def assert_raises_regex(_exception, _regexp, _callable=None,
                        *args, **kwargs):
    r"""
    Check that the specified Exception is raised and that the error message
    matches a given regular expression pattern. This may be a regular
    expression object or a string containing a regular expression suitable
    for use by `re.search()`. This is a port of the `assertRaisesRegexp`
    function from unittest in Python 2.7.

    .. deprecated:: 0.24.0
        Use `pytest.raises` instead.

    Examples
    --------
    >>> assert_raises_regex(ValueError, 'invalid literal for.*XYZ', int, 'XYZ')
    >>> import re
    >>> assert_raises_regex(ValueError, re.compile('literal'), int, 'XYZ')

    If an exception of a different type is raised, it bubbles up.

    >>> assert_raises_regex(TypeError, 'literal', int, 'XYZ')
    Traceback (most recent call last):
        ...
    ValueError: invalid literal for int() with base 10: 'XYZ'
    >>> dct = dict()
    >>> assert_raises_regex(KeyError, 'pear', dct.__getitem__, 'apple')
    Traceback (most recent call last):
        ...
    AssertionError: "pear" does not match "'apple'"

    You can also use this in a with statement.

    >>> with assert_raises_regex(TypeError, r'unsupported operand type\(s\)'):
    ...     1 + {}
    >>> with assert_raises_regex(TypeError, 'banana'):
    ...     'apple'[0] = 'b'
    Traceback (most recent call last):
        ...
    AssertionError: "banana" does not match "'str' object does not support \
item assignment"
    """
    warnings.warn(("assert_raises_regex has been deprecated and will "
                   "be removed in the next release. Please use "
                   "`pytest.raises` instead."), FutureWarning, stacklevel=2)

    manager = _AssertRaisesContextmanager(exception=_exception, regexp=_regexp)
    if _callable is not None:
        with manager:
            _callable(*args, **kwargs)
    else:
        return manager


class _AssertRaisesContextmanager(object):
    """
    Context manager behind `assert_raises_regex`.
    """

    def __init__(self, exception, regexp=None):
        """
        Initialize an _AssertRaisesContextManager instance.

        Parameters
        ----------
        exception : class
            The expected Exception class.
        regexp : str, default None
            The regex to compare against the Exception message.
        """

        self.exception = exception

        if regexp is not None and not hasattr(regexp, "search"):
            regexp = re.compile(regexp, re.DOTALL)

        self.regexp = regexp

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, trace_back):
        expected = self.exception

        if not exc_type:
            exp_name = getattr(expected, "__name__", str(expected))
            raise AssertionError("{name} not raised.".format(name=exp_name))

        return self.exception_matches(exc_type, exc_value, trace_back)

    def exception_matches(self, exc_type, exc_value, trace_back):
        """
        Check that the Exception raised matches the expected Exception
        and expected error message regular expression.

        Parameters
        ----------
        exc_type : class
            The type of Exception raised.
        exc_value : Exception
            The instance of `exc_type` raised.
        trace_back : stack trace object
            The traceback object associated with `exc_value`.

        Returns
        -------
        is_matched : bool
            Whether or not the Exception raised matches the expected
            Exception class and expected error message regular expression.

        Raises
        ------
        AssertionError : The error message provided does not match
                         the expected error message regular expression.
        """

        if issubclass(exc_type, self.exception):
            if self.regexp is not None:
                val = str(exc_value)

                if not self.regexp.search(val):
                    msg = '"{pat}" does not match "{val}"'.format(
                        pat=self.regexp.pattern, val=val)
                    e = AssertionError(msg)
                    raise_with_traceback(e, trace_back)

            return True
        else:
            # Failed, so allow Exception to bubble up.
            return False


@contextmanager
def assert_produces_warning(expected_warning=Warning, filter_level="always",
                            clear=None, check_stacklevel=True):
    """
    Context manager for running code expected to either raise a specific
    warning, or not raise any warnings. Verifies that the code raises the
    expected warning, and that it does not raise any other unexpected
    warnings. It is basically a wrapper around ``warnings.catch_warnings``.

    Parameters
    ----------
    expected_warning : {Warning, False, None}, default Warning
        The type of Exception raised. ``exception.Warning`` is the base
        class for all warnings. To check that no warning is returned,
        specify ``False`` or ``None``.
    filter_level : str, default "always"
        Specifies whether warnings are ignored, displayed, or turned
        into errors.
        Valid values are:

        * "error" - turns matching warnings into exceptions
        * "ignore" - discard the warning
        * "always" - always emit a warning
        * "default" - print the warning the first time it is generated
          from each location
        * "module" - print the warning the first time it is generated
          from each module
        * "once" - print the warning the first time it is generated

    clear : str, default None
        If not ``None`` then remove any previously raised warnings from
        the ``__warningsregistry__`` to ensure that no warning messages are
        suppressed by this context manager. If ``None`` is specified,
        the ``__warningsregistry__`` keeps track of which warnings have been
        shown, and does not show them again.
    check_stacklevel : bool, default True
        If True, displays the line that called the function containing
        the warning to show were the function is called. Otherwise, the
        line that implements the function is displayed.

    Examples
    --------
    >>> import warnings
    >>> with assert_produces_warning():
    ...     warnings.warn(UserWarning())
    ...
    >>> with assert_produces_warning(False):
    ...     warnings.warn(RuntimeWarning())
    ...
    Traceback (most recent call last):
        ...
    AssertionError: Caused unexpected warning(s): ['RuntimeWarning'].
    >>> with assert_produces_warning(UserWarning):
    ...     warnings.warn(RuntimeWarning())
    Traceback (most recent call last):
        ...
    AssertionError: Did not see expected warning of class 'UserWarning'.

    ..warn:: This is *not* thread-safe.
    """
    __tracebackhide__ = True

    with warnings.catch_warnings(record=True) as w:

        if clear is not None:
            # make sure that we are clearing these warnings
            # if they have happened before
            # to guarantee that we will catch them
            if not is_list_like(clear):
                clear = [clear]
            for m in clear:
                try:
                    m.__warningregistry__.clear()
                except Exception:
                    pass

        saw_warning = False
        warnings.simplefilter(filter_level)
        yield w
        extra_warnings = []

        for actual_warning in w:
            if (expected_warning and issubclass(actual_warning.category,
                                                expected_warning)):
                saw_warning = True

                if check_stacklevel and issubclass(actual_warning.category,
                                                   (FutureWarning,
                                                    DeprecationWarning)):
                    from inspect import getframeinfo, stack
                    caller = getframeinfo(stack()[2][0])
                    msg = ("Warning not set with correct stacklevel. "
                           "File where warning is raised: {actual} != "
                           "{caller}. Warning message: {message}"
                           ).format(actual=actual_warning.filename,
                                    caller=caller.filename,
                                    message=actual_warning.message)
                    assert actual_warning.filename == caller.filename, msg
            else:
                extra_warnings.append((actual_warning.category.__name__,
                                       actual_warning.message,
                                       actual_warning.filename,
                                       actual_warning.lineno))
        if expected_warning:
            msg = "Did not see expected warning of class {name!r}.".format(
                name=expected_warning.__name__)
            assert saw_warning, msg
        assert not extra_warnings, ("Caused unexpected warning(s): {extra!r}."
                                    ).format(extra=extra_warnings)


class RNGContext(object):
    """
    Context manager to set the numpy random number generator speed. Returns
    to the original value upon exiting the context manager.

    Parameters
    ----------
    seed : int
        Seed for numpy.random.seed

    Examples
    --------

    with RNGContext(42):
        np.random.randn()
    """

    def __init__(self, seed):
        self.seed = seed

    def __enter__(self):

        self.start_state = np.random.get_state()
        np.random.seed(self.seed)

    def __exit__(self, exc_type, exc_value, traceback):

        np.random.set_state(self.start_state)


@contextmanager
def with_csv_dialect(name, **kwargs):
    """
    Context manager to temporarily register a CSV dialect for parsing CSV.

    Parameters
    ----------
    name : str
        The name of the dialect.
    kwargs : mapping
        The parameters for the dialect.

    Raises
    ------
    ValueError : the name of the dialect conflicts with a builtin one.

    See Also
    --------
    csv : Python's CSV library.
    """
    import csv
    _BUILTIN_DIALECTS = {"excel", "excel-tab", "unix"}

    if name in _BUILTIN_DIALECTS:
        raise ValueError("Cannot override builtin dialect.")

    csv.register_dialect(name, **kwargs)
    yield
    csv.unregister_dialect(name)


@contextmanager
def use_numexpr(use, min_elements=None):
    from pandas.core.computation import expressions as expr
    if min_elements is None:
        min_elements = expr._MIN_ELEMENTS

    olduse = expr._USE_NUMEXPR
    oldmin = expr._MIN_ELEMENTS
    expr.set_use_numexpr(use)
    expr._MIN_ELEMENTS = min_elements
    yield
    expr._MIN_ELEMENTS = oldmin
    expr.set_use_numexpr(olduse)


def test_parallel(num_threads=2, kwargs_list=None):
    """Decorator to run the same function multiple times in parallel.

    Parameters
    ----------
    num_threads : int, optional
        The number of times the function is run in parallel.
    kwargs_list : list of dicts, optional
        The list of kwargs to update original
        function kwargs on different threads.
    Notes
    -----
    This decorator does not pass the return value of the decorated function.

    Original from scikit-image:

    https://github.com/scikit-image/scikit-image/pull/1519

    """

    assert num_threads > 0
    has_kwargs_list = kwargs_list is not None
    if has_kwargs_list:
        assert len(kwargs_list) == num_threads
    import threading

    def wrapper(func):
        @wraps(func)
        def inner(*args, **kwargs):
            if has_kwargs_list:
                update_kwargs = lambda i: dict(kwargs, **kwargs_list[i])
            else:
                update_kwargs = lambda i: kwargs
            threads = []
            for i in range(num_threads):
                updated_kwargs = update_kwargs(i)
                thread = threading.Thread(target=func, args=args,
                                          kwargs=updated_kwargs)
                threads.append(thread)
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        return inner
    return wrapper


class SubclassedSeries(Series):
    _metadata = ['testattr', 'name']

    @property
    def _constructor(self):
        return SubclassedSeries

    @property
    def _constructor_expanddim(self):
        return SubclassedDataFrame


class SubclassedDataFrame(DataFrame):
    _metadata = ['testattr']

    @property
    def _constructor(self):
        return SubclassedDataFrame

    @property
    def _constructor_sliced(self):
        return SubclassedSeries


class SubclassedSparseSeries(pd.SparseSeries):
    _metadata = ['testattr']

    @property
    def _constructor(self):
        return SubclassedSparseSeries

    @property
    def _constructor_expanddim(self):
        return SubclassedSparseDataFrame


class SubclassedSparseDataFrame(pd.SparseDataFrame):
    _metadata = ['testattr']

    @property
    def _constructor(self):
        return SubclassedSparseDataFrame

    @property
    def _constructor_sliced(self):
        return SubclassedSparseSeries


class SubclassedCategorical(Categorical):

    @property
    def _constructor(self):
        return SubclassedCategorical


@contextmanager
def set_timezone(tz):
    """Context manager for temporarily setting a timezone.

    Parameters
    ----------
    tz : str
        A string representing a valid timezone.

    Examples
    --------

    >>> from datetime import datetime
    >>> from dateutil.tz import tzlocal
    >>> tzlocal().tzname(datetime.now())
    'IST'

    >>> with set_timezone('US/Eastern'):
    ...     tzlocal().tzname(datetime.now())
    ...
    'EDT'
    """

    import os
    import time

    def setTZ(tz):
        if tz is None:
            try:
                del os.environ['TZ']
            except KeyError:
                pass
        else:
            os.environ['TZ'] = tz
            time.tzset()

    orig_tz = os.environ.get('TZ')
    setTZ(tz)
    try:
        yield
    finally:
        setTZ(orig_tz)


def _make_skipna_wrapper(alternative, skipna_alternative=None):
    """Create a function for calling on an array.

    Parameters
    ----------
    alternative : function
        The function to be called on the array with no NaNs.
        Only used when 'skipna_alternative' is None.
    skipna_alternative : function
        The function to be called on the original array

    Returns
    -------
    skipna_wrapper : function
    """
    if skipna_alternative:
        def skipna_wrapper(x):
            return skipna_alternative(x.values)
    else:
        def skipna_wrapper(x):
            nona = x.dropna()
            if len(nona) == 0:
                return np.nan
            return alternative(nona)

    return skipna_wrapper


def convert_rows_list_to_csv_str(rows_list):
    """
    Convert list of CSV rows to single CSV-formatted string for current OS.

    This method is used for creating expected value of to_csv() method.

    Parameters
    ----------
    rows_list : list
        The list of string. Each element represents the row of csv.

    Returns
    -------
    expected : string
        Expected output of to_csv() in current OS
    """
    sep = os.linesep
    expected = sep.join(rows_list) + sep
    return expected
