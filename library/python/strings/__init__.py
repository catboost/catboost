import locale
import logging
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
# Chooses between str/unicode
def to_basestring(value):
    if isinstance(value, basestring):
        return value
    try:
        return unicode(value)
    except UnicodeDecodeError:
        try:
            return str(value)
        except UnicodeEncodeError:
            return repr(value)
to_text = to_basestring


def to_unicode(value, from_enc=DEFAULT_ENCODING):
    if isinstance(value, unicode):
        return value
    if isinstance(value, basestring):
        return unicode(value, from_enc, ENCODING_ERRORS_POLICY)
    return unicode(value)


# Optional from_enc enables transcoding
def to_str(value, to_enc=DEFAULT_ENCODING, from_enc=None):
    if isinstance(value, str):
        if from_enc is None or to_enc == from_enc:
            # Unknown input encoding or input and output encoding are the same
            return value
        value = to_unicode(value, from_enc=from_enc)
    if isinstance(value, unicode):
        return value.encode(to_enc, ENCODING_ERRORS_POLICY)
    return str(value)


def _convert_deep(x, enc, convert, relaxed=True):
    if x is None:
        return None
    if isinstance(x, (str, unicode)):
        return convert(x, enc)
    if isinstance(x, dict):
        return {convert(k, enc): _convert_deep(v, enc, convert, relaxed) for k, v in x.iteritems()}
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
        return locale.getdefaultlocale()[1]
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
    if type(value) == str:
        value = value.decode(encoding, errors='ignore')
    return value.encode(encoding)
