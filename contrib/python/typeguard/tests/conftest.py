import re
import sys

version_re = re.compile(r'_py(\d)(\d)\.py$')


def pytest_ignore_collect(path, config):
    match = version_re.search(path.basename)
    if match:
        version = tuple(int(x) for x in match.groups())
        if sys.version_info < version:
            return True
