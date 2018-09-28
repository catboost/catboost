import library.python.resource.ut_py3.check.test_simple as test
import sys

def main():
    assert(sys.version_info.major == 3)
    test.test_simple()
    test.test_iter()

