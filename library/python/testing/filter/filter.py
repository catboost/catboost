# coding: utf-8
# TODO move devtools/ya/test/filter.py to library/python/testing/filter/filter.py
import re
import fnmatch
import logging

logger = logging.getLogger(__name__)
TEST_SUBTEST_SEPARATOR = '::'

PARSE_TAG_RE = re.compile(r"([+-]?[\w:]*)")


class FilterException(Exception):
    mute = True


def fix_filter(flt):
    if TEST_SUBTEST_SEPARATOR not in flt and "*" not in flt:
        # user wants to filter by test module name
        flt = flt + TEST_SUBTEST_SEPARATOR + "*"
    return flt


def escape_for_fnmatch(s):
    return s.replace("[", "&#91;").replace("]", "&#93;")


def make_py_file_filter(filter_names):
    if filter_names is not None:
        with_star = []
        wo_star = set()
        for flt in filter_names:
            flt = flt.split(':')[0]
            if '*' in flt:
                with_star.append(flt.split('*')[0] + '*')
            else:
                wo_star.add(flt)

    def predicate(filename):
        if filter_names is None:
            return True
        return filename in wo_star or any([fnmatch.fnmatch(escape_for_fnmatch(filename), escape_for_fnmatch(filter_name)) for filter_name in with_star])

    return predicate


def make_name_filter(filter_names):
    filter_names = map(fix_filter, filter_names)
    filter_full_names = set()
    for name in filter_names:
        if '*' not in name:
            filter_full_names.add(name)

    def predicate(testname):
        return testname in filter_full_names or any([fnmatch.fnmatch(escape_for_fnmatch(testname), escape_for_fnmatch(filter_name)) for filter_name in filter_names])

    return predicate
