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


_hexdig = "0123456789ABCDEFabcdef"
_hextobyte = {
    (a + b).encode(): bytes.fromhex(a + b) if six.PY3 else (a + b).decode("hex") for a in _hexdig for b in _hexdig
}


def parse_qs_binary(qs, keep_blank_values=False, strict_parsing=False, max_num_fields=None, separator=b'&'):
    """Parse a query like original `parse_qs` from `urlparse`, `urllib.parse`, but query given as a bytes argument.

    Arguments:

    qs: percent-encoded query string to be parsed

    keep_blank_values: flag indicating whether blank values in
        percent-encoded queries should be treated as blank byte strings.
        A true value indicates that blanks should be retained as
        blank byte strings. The default false value indicates that
        blank values are to be ignored and treated as if they were
        not included.

    strict_parsing: flag indicating what to do with parsing errors.
        If false (the default), errors are silently ignored.
        If true, errors raise a ValueError exception.

    max_num_fields: int. If set, then throws a ValueError if there
        are more than n fields read by parse_qsl_binary().

    separator: bytes. The symbol to use for separating the query arguments.
        Defaults to &.

    Returns a dictionary.
    """
    parsed_result = {}
    pairs = parse_qsl_binary(qs, keep_blank_values, strict_parsing, max_num_fields=max_num_fields, separator=separator)
    for name, value in pairs:
        if name in parsed_result:
            parsed_result[name].append(value)
        else:
            parsed_result[name] = [value]
    return parsed_result


def parse_qsl_binary(qs, keep_blank_values=False, strict_parsing=False, max_num_fields=None, separator=b'&'):
    """Parse a query like original `parse_qs` from `urlparse`, `urllib.parse`, but query given as a bytes argument.

    Arguments:

    qs: percent-encoded query bytes to be parsed

    keep_blank_values: flag indicating whether blank values in
        percent-encoded queries should be treated as blank byte strings.
        A true value indicates that blanks should be retained as blank
        byte strings. The default false value indicates that blank values
        are to be ignored and treated as if they were not included.

    strict_parsing: flag indicating what to do with parsing errors. If
        false (the default), errors are silently ignored. If true,
        errors raise a ValueError exception.

    max_num_fields: int. If set, then throws a ValueError
        if there are more than n fields read by parse_qsl_binary().

    separator: bytes. The symbol to use for separating the query arguments.
        Defaults to &.

    Returns a list.
    """

    if max_num_fields is not None:
        num_fields = 1 + qs.count(separator) if qs else 0
        if max_num_fields < num_fields:
            raise ValueError('Max number of fields exceeded')

    r = []
    query_args = qs.split(separator) if qs else []
    for name_value in query_args:
        if not name_value and not strict_parsing:
            continue
        nv = name_value.split(b'=', 1)

        if len(nv) != 2:
            if strict_parsing:
                raise ValueError("bad query field: %r" % (name_value,))
            # Handle case of a control-name with no equal sign
            if keep_blank_values:
                nv.append(b'')
            else:
                continue
        if len(nv[1]) or keep_blank_values:
            name = nv[0].replace(b'+', b' ')
            name = unquote_binary(name)
            value = nv[1].replace(b'+', b' ')
            value = unquote_binary(value)
            r.append((name, value))
    return r


def unquote_binary(string):
    """Replace %xx escapes by their single-character equivalent.
    By default, percent-encoded sequences are replaced by ASCII character or
    byte code, and invalid sequences are replaced by a placeholder character.

    unquote('abc%20def') -> 'abc def'
    unquote('abc%FFdef') -> 'abc\xffdef'
    unquote('%no') -> '%no'
    """
    bits = string.split(b"%")
    if len(bits) == 1:
        return bits[0]

    res = [bits[0]]
    for item in bits[1:]:
        res.append(_hextobyte.get(item[:2], b"%"))
        res.append(item if res[-1] == b"%" else item[2:])
    return b"".join(res)
