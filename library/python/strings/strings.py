import locale
import logging
import six
import sys
import codecs

import library.python.func

logger = logging.getLogger(__name__)


DEFAULT_ENCODING = 'utf-8'
ENCODING_ERRORS_POLICY = 'replace'


def left_strip(el, prefix):
    """
    Strips prefix at the left of el
    """
    if el.startswith(prefix):
        return el[len(prefix) :]
    return el


# Explicit to-text conversion
# Chooses between str/unicode, i.e. six.binary_type/six.text_type
def to_basestring(value):
    if isinstance(value, (six.binary_type, six.text_type)):
        return value
    try:
        if six.PY2:
            return unicode(value)  # noqa
        else:
            return str(value)
    except UnicodeDecodeError:
        try:
            return str(value)
        except UnicodeEncodeError:
            return repr(value)


to_text = to_basestring


def to_unicode(value, from_enc=DEFAULT_ENCODING):
    if isinstance(value, six.text_type):
        return value
    if isinstance(value, six.binary_type):
        if six.PY2:
            return unicode(value, from_enc, ENCODING_ERRORS_POLICY)  # noqa
        else:
            return value.decode(from_enc, errors=ENCODING_ERRORS_POLICY)
    return six.text_type(value)


# Optional from_enc enables transcoding
def to_str(value, to_enc=DEFAULT_ENCODING, from_enc=None):
    if isinstance(value, six.binary_type):
        if from_enc is None or to_enc == from_enc:
            # Unknown input encoding or input and output encoding are the same
            return value
        value = to_unicode(value, from_enc=from_enc)
    if isinstance(value, six.text_type):
        return value.encode(to_enc, ENCODING_ERRORS_POLICY)
    return six.binary_type(value)


def _convert_deep(x, enc, convert, relaxed=True):
    if x is None:
        return None
    if isinstance(x, (six.text_type, six.binary_type)):
        return convert(x, enc)
    if isinstance(x, dict):
        return {convert(k, enc): _convert_deep(v, enc, convert, relaxed) for k, v in six.iteritems(x)}
    if isinstance(x, list):
        return [_convert_deep(e, enc, convert, relaxed) for e in x]
    if isinstance(x, tuple):
        return tuple([_convert_deep(e, enc, convert, relaxed) for e in x])

    if relaxed:
        return x
    raise TypeError('unsupported type')


# Result as from six.ensure_text
def unicodize_deep(x, enc=DEFAULT_ENCODING, relaxed=True):
    return _convert_deep(x, enc, to_unicode, relaxed)


# Result as from six.ensure_str
def ensure_str_deep(x, enc=DEFAULT_ENCODING, relaxed=True):
    return _convert_deep(x, enc, six.ensure_str, relaxed)


# Result as from six.ensure_binary
def stringize_deep(x, enc=DEFAULT_ENCODING, relaxed=True):
    return _convert_deep(x, enc, to_str, relaxed)


@library.python.func.memoize()
def locale_encoding():
    try:
        if six.PY3:
            loc = locale.getencoding()
        else:
            loc = locale.getdefaultlocale()[1]
        if loc:
            codecs.lookup(loc)
        return loc
    except LookupError as e:
        logger.debug('Cannot get system locale: %s', e)
        return None
    except ValueError as e:
        logger.debug('Cannot get system locale: %s', e)
        return None


def fs_encoding():
    return sys.getfilesystemencoding()


def guess_default_encoding():
    enc = locale_encoding()
    return enc if enc else DEFAULT_ENCODING


@library.python.func.memoize()
def get_stream_encoding(stream):
    if stream.encoding:
        try:
            codecs.lookup(stream.encoding)
            return stream.encoding
        except LookupError:
            pass
    return DEFAULT_ENCODING


def encode(value, encoding=DEFAULT_ENCODING):
    if isinstance(value, six.binary_type):
        value = value.decode(encoding, errors='ignore')
    return value.encode(encoding)


class Whence(object):
    Start = 0
    End = 1
    Middle = 2


def truncate(data, limit, whence=None, msg=None):
    msg = "..." if msg is None else msg
    msg = six.ensure_binary(msg)
    whence = Whence.End if whence is None else whence
    data = six.ensure_binary(data)

    if len(data) <= limit:
        return six.ensure_str(data)
    text_limit = limit - len(msg)
    assert text_limit >= 0

    if whence == Whence.Start:
        data = msg + data[-text_limit:]
    elif whence == Whence.End:
        data = data[:text_limit] + msg
    elif whence == Whence.Middle:
        headpos = limit // 2 - len(msg) // 2
        tailpos = len(data) - (text_limit - headpos)
        data = data[:headpos] + msg + data[tailpos:]
    else:
        raise AssertionError("Unknown whence: %s" % str(whence))
    return fix_utf8(data)


def fix_utf8(data):
    # type: (six.string_types) -> str
    # remove destroyed symbol code
    udata = six.ensure_text(data, 'utf-8', 'ignore')
    return six.ensure_str(udata, 'utf-8', errors='ignore')
