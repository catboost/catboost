# _compat.py - Python 2/3 compatibility

import os
import sys

PY2 = sys.version_info[0] == 2


if PY2:
    string_classes = (str, unicode)  # needed individually for sublassing
    text_type = unicode

    def iteritems(d):
        return d.iteritems()

    def makedirs(name, mode=0o777, exist_ok=False):
        try:
            os.makedirs(name, mode)
        except OSError:
            if not exist_ok or not os.path.isdir(name):
                raise

    def stderr_write_binary(data):
        sys.stderr.write(data)


else:
    string_classes = (str,)
    text_type = str

    def iteritems(d):
        return iter(d.items())

    def makedirs(name, mode=0o777, exist_ok=False):  # allow os.makedirs mocking
        return os.makedirs(name, mode, exist_ok=exist_ok)

    def stderr_write_binary(data):
        encoding = sys.stderr.encoding or sys.getdefaultencoding()
        sys.stderr.write(data.decode(encoding))
