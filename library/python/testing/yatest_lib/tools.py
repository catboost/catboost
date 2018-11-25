import sys


def to_utf8(value):
    """
    Converts value to string encoded into utf-8
    :param value:
    :return:
    """
    if sys.version_info[0] < 3:
        if not isinstance(value, basestring):
            value = unicode(value)
        if type(value) == str:
            value = value.decode("utf-8", errors="ignore")
        return value.encode('utf-8', 'ignore')
    else:
        return str(value)
