"""Common IO api utilities"""

import codecs
from contextlib import closing, contextmanager
import csv
import mmap
import os
import zipfile

import pandas.compat as compat
from pandas.compat import BytesIO, StringIO, string_types, text_type
from pandas.errors import (  # noqa
    AbstractMethodError, DtypeWarning, EmptyDataError, ParserError,
    ParserWarning)

from pandas.core.dtypes.common import is_file_like, is_number

from pandas.io.formats.printing import pprint_thing

# gh-12665: Alias for now and remove later.
CParserError = ParserError

# common NA values
# no longer excluding inf representations
# '1.#INF','-1.#INF', '1.#INF000000',
_NA_VALUES = {'-1.#IND', '1.#QNAN', '1.#IND', '-1.#QNAN', '#N/A N/A', '#N/A',
              'N/A', 'n/a', 'NA', '#NA', 'NULL', 'null', 'NaN', '-NaN', 'nan',
              '-nan', ''}


if compat.PY3:
    from urllib.request import urlopen, pathname2url
    _urlopen = urlopen
    from urllib.parse import urlparse as parse_url
    from urllib.parse import (uses_relative, uses_netloc, uses_params,
                              urlencode, urljoin)
    from urllib.error import URLError
    from http.client import HTTPException  # noqa
else:
    from urllib2 import urlopen as _urlopen
    from urllib import urlencode, pathname2url  # noqa
    from urlparse import urlparse as parse_url
    from urlparse import uses_relative, uses_netloc, uses_params, urljoin
    from urllib2 import URLError  # noqa
    from httplib import HTTPException  # noqa
    from contextlib import contextmanager, closing  # noqa
    from functools import wraps  # noqa

    # @wraps(_urlopen)
    @contextmanager
    def urlopen(*args, **kwargs):
        with closing(_urlopen(*args, **kwargs)) as f:
            yield f


_VALID_URLS = set(uses_relative + uses_netloc + uses_params)
_VALID_URLS.discard('')


class BaseIterator(object):
    """Subclass this and provide a "__next__()" method to obtain an iterator.
    Useful only when the object being iterated is non-reusable (e.g. OK for a
    parser, not for an in-memory table, yes for its iterator)."""

    def __iter__(self):
        return self

    def __next__(self):
        raise AbstractMethodError(self)


if not compat.PY3:
    BaseIterator.next = lambda self: self.__next__()


def _is_url(url):
    """Check to see if a URL has a valid protocol.

    Parameters
    ----------
    url : str or unicode

    Returns
    -------
    isurl : bool
        If `url` has a valid protocol return True otherwise False.
    """
    try:
        return parse_url(url).scheme in _VALID_URLS
    except Exception:
        return False


def _expand_user(filepath_or_buffer):
    """Return the argument with an initial component of ~ or ~user
       replaced by that user's home directory.

    Parameters
    ----------
    filepath_or_buffer : object to be converted if possible

    Returns
    -------
    expanded_filepath_or_buffer : an expanded filepath or the
                                  input if not expandable
    """
    if isinstance(filepath_or_buffer, string_types):
        return os.path.expanduser(filepath_or_buffer)
    return filepath_or_buffer


def _validate_header_arg(header):
    if isinstance(header, bool):
        raise TypeError("Passing a bool to header is invalid. "
                        "Use header=None for no header or "
                        "header=int or list-like of ints to specify "
                        "the row(s) making up the column names")


def _stringify_path(filepath_or_buffer):
    """Attempt to convert a path-like object to a string.

    Parameters
    ----------
    filepath_or_buffer : object to be converted

    Returns
    -------
    str_filepath_or_buffer : maybe a string version of the object

    Notes
    -----
    Objects supporting the fspath protocol (python 3.6+) are coerced
    according to its __fspath__ method.

    For backwards compatibility with older pythons, pathlib.Path and
    py.path objects are specially coerced.

    Any other object is passed through unchanged, which includes bytes,
    strings, buffers, or anything else that's not even path-like.
    """
    try:
        import pathlib
        _PATHLIB_INSTALLED = True
    except ImportError:
        _PATHLIB_INSTALLED = False

    try:
        from py.path import local as LocalPath
        _PY_PATH_INSTALLED = True
    except ImportError:
        _PY_PATH_INSTALLED = False

    if hasattr(filepath_or_buffer, '__fspath__'):
        return filepath_or_buffer.__fspath__()
    if _PATHLIB_INSTALLED and isinstance(filepath_or_buffer, pathlib.Path):
        return text_type(filepath_or_buffer)
    if _PY_PATH_INSTALLED and isinstance(filepath_or_buffer, LocalPath):
        return filepath_or_buffer.strpath
    return _expand_user(filepath_or_buffer)


def is_s3_url(url):
    """Check for an s3, s3n, or s3a url"""
    try:
        return parse_url(url).scheme in ['s3', 's3n', 's3a']
    except Exception:
        return False


def is_gcs_url(url):
    """Check for a gcs url"""
    try:
        return parse_url(url).scheme in ['gcs', 'gs']
    except Exception:
        return False


def get_filepath_or_buffer(filepath_or_buffer, encoding=None,
                           compression=None, mode=None):
    """
    If the filepath_or_buffer is a url, translate and return the buffer.
    Otherwise passthrough.

    Parameters
    ----------
    filepath_or_buffer : a url, filepath (str, py.path.local or pathlib.Path),
                         or buffer
    encoding : the encoding to use to decode py3 bytes, default is 'utf-8'
    mode : str, optional

    Returns
    -------
    tuple of ({a filepath_ or buffer or S3File instance},
              encoding, str,
              compression, str,
              should_close, bool)
    """
    filepath_or_buffer = _stringify_path(filepath_or_buffer)

    if _is_url(filepath_or_buffer):
        req = _urlopen(filepath_or_buffer)
        content_encoding = req.headers.get('Content-Encoding', None)
        if content_encoding == 'gzip':
            # Override compression based on Content-Encoding header
            compression = 'gzip'
        reader = BytesIO(req.read())
        req.close()
        return reader, encoding, compression, True

    if is_s3_url(filepath_or_buffer):
        from pandas.io import s3
        return s3.get_filepath_or_buffer(filepath_or_buffer,
                                         encoding=encoding,
                                         compression=compression,
                                         mode=mode)

    if is_gcs_url(filepath_or_buffer):
        from pandas.io import gcs
        return gcs.get_filepath_or_buffer(filepath_or_buffer,
                                          encoding=encoding,
                                          compression=compression,
                                          mode=mode)

    if isinstance(filepath_or_buffer, (compat.string_types,
                                       compat.binary_type,
                                       mmap.mmap)):
        return _expand_user(filepath_or_buffer), None, compression, False

    if not is_file_like(filepath_or_buffer):
        msg = "Invalid file path or buffer object type: {_type}"
        raise ValueError(msg.format(_type=type(filepath_or_buffer)))

    return filepath_or_buffer, None, compression, False


def file_path_to_url(path):
    """
    converts an absolute native path to a FILE URL.

    Parameters
    ----------
    path : a path in native format

    Returns
    -------
    a valid FILE URL
    """
    return urljoin('file:', pathname2url(path))


_compression_to_extension = {
    'gzip': '.gz',
    'bz2': '.bz2',
    'zip': '.zip',
    'xz': '.xz',
}


def _infer_compression(filepath_or_buffer, compression):
    """
    Get the compression method for filepath_or_buffer. If compression='infer',
    the inferred compression method is returned. Otherwise, the input
    compression method is returned unchanged, unless it's invalid, in which
    case an error is raised.

    Parameters
    ----------
    filepath_or_buffer :
        a path (str) or buffer
    compression : {'infer', 'gzip', 'bz2', 'zip', 'xz', None}
        If 'infer' and `filepath_or_buffer` is path-like, then detect
        compression from the following extensions: '.gz', '.bz2', '.zip',
        or '.xz' (otherwise no compression).

    Returns
    -------
    string or None :
        compression method

    Raises
    ------
    ValueError on invalid compression specified
    """

    # No compression has been explicitly specified
    if compression is None:
        return None

    # Infer compression
    if compression == 'infer':
        # Convert all path types (e.g. pathlib.Path) to strings
        filepath_or_buffer = _stringify_path(filepath_or_buffer)
        if not isinstance(filepath_or_buffer, compat.string_types):
            # Cannot infer compression of a buffer, assume no compression
            return None

        # Infer compression from the filename/URL extension
        for compression, extension in _compression_to_extension.items():
            if filepath_or_buffer.endswith(extension):
                return compression
        return None

    # Compression has been specified. Check that it's valid
    if compression in _compression_to_extension:
        return compression

    msg = 'Unrecognized compression type: {}'.format(compression)
    valid = ['infer', None] + sorted(_compression_to_extension)
    msg += '\nValid compression types are {}'.format(valid)
    raise ValueError(msg)


def _get_handle(path_or_buf, mode, encoding=None, compression=None,
                memory_map=False, is_text=True):
    """
    Get file handle for given path/buffer and mode.

    Parameters
    ----------
    path_or_buf :
        a path (str) or buffer
    mode : str
        mode to open path_or_buf with
    encoding : str or None
    compression : {'infer', 'gzip', 'bz2', 'zip', 'xz', None}, default None
        If 'infer' and `filepath_or_buffer` is path-like, then detect
        compression from the following extensions: '.gz', '.bz2', '.zip',
        or '.xz' (otherwise no compression).
    memory_map : boolean, default False
        See parsers._parser_params for more information.
    is_text : boolean, default True
        whether file/buffer is in text format (csv, json, etc.), or in binary
        mode (pickle, etc.)

    Returns
    -------
    f : file-like
        A file-like object
    handles : list of file-like objects
        A list of file-like object that were opened in this function.
    """
    try:
        from s3fs import S3File
        need_text_wrapping = (BytesIO, S3File)
    except ImportError:
        need_text_wrapping = (BytesIO,)

    handles = list()
    f = path_or_buf

    # Convert pathlib.Path/py.path.local or string
    path_or_buf = _stringify_path(path_or_buf)
    is_path = isinstance(path_or_buf, compat.string_types)

    if is_path:
        compression = _infer_compression(path_or_buf, compression)

    if compression:

        if compat.PY2 and not is_path and encoding:
            msg = 'compression with encoding is not yet supported in Python 2'
            raise ValueError(msg)

        # GZ Compression
        if compression == 'gzip':
            import gzip
            if is_path:
                f = gzip.open(path_or_buf, mode)
            else:
                f = gzip.GzipFile(fileobj=path_or_buf)

        # BZ Compression
        elif compression == 'bz2':
            import bz2
            if is_path:
                f = bz2.BZ2File(path_or_buf, mode)
            elif compat.PY2:
                # Python 2's bz2 module can't take file objects, so have to
                # run through decompress manually
                f = StringIO(bz2.decompress(path_or_buf.read()))
                path_or_buf.close()
            else:
                f = bz2.BZ2File(path_or_buf)

        # ZIP Compression
        elif compression == 'zip':
            zf = BytesZipFile(path_or_buf, mode)
            # Ensure the container is closed as well.
            handles.append(zf)
            if zf.mode == 'w':
                f = zf
            elif zf.mode == 'r':
                zip_names = zf.namelist()
                if len(zip_names) == 1:
                    f = zf.open(zip_names.pop())
                elif len(zip_names) == 0:
                    raise ValueError('Zero files found in ZIP file {}'
                                     .format(path_or_buf))
                else:
                    raise ValueError('Multiple files found in ZIP file.'
                                     ' Only one file per ZIP: {}'
                                     .format(zip_names))

        # XZ Compression
        elif compression == 'xz':
            lzma = compat.import_lzma()
            f = lzma.LZMAFile(path_or_buf, mode)

        # Unrecognized Compression
        else:
            msg = 'Unrecognized compression type: {}'.format(compression)
            raise ValueError(msg)

        handles.append(f)

    elif is_path:
        if compat.PY2:
            # Python 2
            mode = "wb" if mode == "w" else mode
            f = open(path_or_buf, mode)
        elif encoding:
            # Python 3 and encoding
            f = open(path_or_buf, mode, encoding=encoding, newline="")
        elif is_text:
            # Python 3 and no explicit encoding
            f = open(path_or_buf, mode, errors='replace', newline="")
        else:
            # Python 3 and binary mode
            f = open(path_or_buf, mode)
        handles.append(f)

    # in Python 3, convert BytesIO or fileobjects passed with an encoding
    if (compat.PY3 and is_text and
            (compression or isinstance(f, need_text_wrapping))):
        from io import TextIOWrapper
        f = TextIOWrapper(f, encoding=encoding, newline='')
        handles.append(f)

    if memory_map and hasattr(f, 'fileno'):
        try:
            g = MMapWrapper(f)
            f.close()
            f = g
        except Exception:
            # we catch any errors that may have occurred
            # because that is consistent with the lower-level
            # functionality of the C engine (pd.read_csv), so
            # leave the file handler as is then
            pass

    return f, handles


class BytesZipFile(zipfile.ZipFile, BytesIO):
    """
    Wrapper for standard library class ZipFile and allow the returned file-like
    handle to accept byte strings via `write` method.

    BytesIO provides attributes of file-like object and ZipFile.writestr writes
    bytes strings into a member of the archive.
    """
    # GH 17778
    def __init__(self, file, mode, compression=zipfile.ZIP_DEFLATED, **kwargs):
        if mode in ['wb', 'rb']:
            mode = mode.replace('b', '')
        super(BytesZipFile, self).__init__(file, mode, compression, **kwargs)

    def write(self, data):
        super(BytesZipFile, self).writestr(self.filename, data)

    @property
    def closed(self):
        return self.fp is None


class MMapWrapper(BaseIterator):
    """
    Wrapper for the Python's mmap class so that it can be properly read in
    by Python's csv.reader class.

    Parameters
    ----------
    f : file object
        File object to be mapped onto memory. Must support the 'fileno'
        method or have an equivalent attribute

    """

    def __init__(self, f):
        self.mmap = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    def __getattr__(self, name):
        return getattr(self.mmap, name)

    def __iter__(self):
        return self

    def __next__(self):
        newline = self.mmap.readline()

        # readline returns bytes, not str, in Python 3,
        # but Python's CSV reader expects str, so convert
        # the output to str before continuing
        if compat.PY3:
            newline = compat.bytes_to_str(newline)

        # mmap doesn't raise if reading past the allocated
        # data but instead returns an empty string, so raise
        # if that is returned
        if newline == '':
            raise StopIteration
        return newline


if not compat.PY3:
    MMapWrapper.next = lambda self: self.__next__()


class UTF8Recoder(BaseIterator):

    """
    Iterator that reads an encoded stream and reencodes the input to UTF-8
    """

    def __init__(self, f, encoding):
        self.reader = codecs.getreader(encoding)(f)

    def read(self, bytes=-1):
        return self.reader.read(bytes).encode("utf-8")

    def readline(self):
        return self.reader.readline().encode("utf-8")

    def next(self):
        return next(self.reader).encode("utf-8")


if compat.PY3:  # pragma: no cover
    def UnicodeReader(f, dialect=csv.excel, encoding="utf-8", **kwds):
        # ignore encoding
        return csv.reader(f, dialect=dialect, **kwds)

    def UnicodeWriter(f, dialect=csv.excel, encoding="utf-8", **kwds):
        return csv.writer(f, dialect=dialect, **kwds)
else:
    class UnicodeReader(BaseIterator):

        """
        A CSV reader which will iterate over lines in the CSV file "f",
        which is encoded in the given encoding.

        On Python 3, this is replaced (below) by csv.reader, which handles
        unicode.
        """

        def __init__(self, f, dialect=csv.excel, encoding="utf-8", **kwds):
            f = UTF8Recoder(f, encoding)
            self.reader = csv.reader(f, dialect=dialect, **kwds)

        def __next__(self):
            row = next(self.reader)
            return [compat.text_type(s, "utf-8") for s in row]

    class UnicodeWriter(object):

        """
        A CSV writer which will write rows to CSV file "f",
        which is encoded in the given encoding.
        """

        def __init__(self, f, dialect=csv.excel, encoding="utf-8", **kwds):
            # Redirect output to a queue
            self.queue = StringIO()
            self.writer = csv.writer(self.queue, dialect=dialect, **kwds)
            self.stream = f
            self.encoder = codecs.getincrementalencoder(encoding)()
            self.quoting = kwds.get("quoting", None)

        def writerow(self, row):
            def _check_as_is(x):
                return (self.quoting == csv.QUOTE_NONNUMERIC and
                        is_number(x)) or isinstance(x, str)

            row = [x if _check_as_is(x)
                   else pprint_thing(x).encode("utf-8") for x in row]

            self.writer.writerow([s for s in row])
            # Fetch UTF-8 output from the queue ...
            data = self.queue.getvalue()
            data = data.decode("utf-8")
            # ... and re-encode it into the target encoding
            data = self.encoder.encode(data)
            # write to the target stream
            self.stream.write(data)
            # empty queue
            self.queue.truncate(0)

        def writerows(self, rows):
            def _check_as_is(x):
                return (self.quoting == csv.QUOTE_NONNUMERIC and
                        is_number(x)) or isinstance(x, str)

            for i, row in enumerate(rows):
                rows[i] = [x if _check_as_is(x)
                           else pprint_thing(x).encode("utf-8") for x in row]

            self.writer.writerows([[s for s in row] for row in rows])
            # Fetch UTF-8 output from the queue ...
            data = self.queue.getvalue()
            data = data.decode("utf-8")
            # ... and re-encode it into the target encoding
            data = self.encoder.encode(data)
            # write to the target stream
            self.stream.write(data)
            # empty queue
            self.queue.truncate(0)
