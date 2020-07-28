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
        return el[len(prefix):]
    return el


# Explicit to-text conversion
# Chooses between str/unicode, i.e. six.binary_type/six.text_type
def to_basestring(value):
    if isinstance(value, (six.binary_type, six.text_type)):
        return value
    try:
        if six.PY2:
            return unicode(value)
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
            return unicode(value, from_enc, ENCODING_ERRORS_POLICY)
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
        return tuple(_convert_deep(e, enc, convert, relaxed) for e in x)

    if relaxed:
        return x
    raise TypeError('unsupported type')


def unicodize_deep(x, enc=DEFAULT_ENCODING, relaxed=True):
    return _convert_deep(x, enc, to_unicode, relaxed)


def stringize_deep(x, enc=DEFAULT_ENCODING, relaxed=True):
    return _convert_deep(x, enc, to_str, relaxed)


@library.python.func.memoize(thread_safe=True)
def locale_encoding():
    try:
        loc = locale.getdefaultlocale()[1]
        if loc:
            codecs.lookup(loc)
        return loc
    except LookupError as e:
        logger.debug('Cannot get system locale: %s', e)
        return None
    except ValueError as e:
        logger.warn('Cannot get system locale: %s', e)
        return None


def fs_encoding():
    return sys.getfilesystemencoding()


def guess_default_encoding():
    enc = locale_encoding()
    return enc if enc else DEFAULT_ENCODING


@library.python.func.memoize(thread_safe=True)
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
