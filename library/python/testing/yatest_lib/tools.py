import six
import sys


def to_utf8(value):
    """
    Converts value to string encoded into utf-8
    :param value:
    :return:
    """
    if sys.version_info[0] < 3:
        if not isinstance(value, basestring):  # noqa
            value = unicode(value)  # noqa
        if type(value) == str:
            value = value.decode("utf-8", errors="ignore")
        return value.encode('utf-8', 'ignore')
    else:
        return str(value)


def trim_string(s, max_bytes):
    """
    Adjusts the length of the string s in order to fit it
    into max_bytes bytes of storage after encoding as UTF-8.
    Useful when cutting filesystem paths.
    :param s: unicode string
    :param max_bytes: number of bytes
    :return the prefix of s
    """
    if isinstance(s, six.text_type):
        return _trim_unicode_string(s, max_bytes)

    if isinstance(s, six.binary_type):
        if len(s) <= max_bytes:
            return s
        s = s.decode('utf-8', errors='ignore')
        s = _trim_unicode_string(s, max_bytes)
        s = s.encode('utf-8', errors='ignore')
        return s

    raise TypeError('a string is expected')


def _trim_unicode_string(s, max_bytes):
    if len(s) * 4 <= max_bytes:
        # UTF-8 uses at most 4 bytes per character
        return s

    result = []
    cur_byte_length = 0

    for ch in s:
        cur_byte_length += len(ch.encode('utf-8'))
        if cur_byte_length > max_bytes:
            break
        result.append(ch)

    return ''.join(result)


def to_str(s):
    if six.PY2 and isinstance(s, six.text_type):
        return s.encode('utf8')
    return s
