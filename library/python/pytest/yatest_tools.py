# coding: utf-8

import os
import re
import sys
import math
import functools

import yatest_lib.tools


class Subtest(object):
    def __init__(self, name, test_name, status, comment, elapsed, result=None, test_type=None, logs=None, cwd=None, metrics=None):
        self._name = name
        self._test_name = test_name
        self.status = status
        self.elapsed = elapsed
        self.comment = comment
        self.result = result
        self.test_type = test_type
        self.logs = logs or {}
        self.cwd = cwd
        self.metrics = metrics

    def __eq__(self, other):
        if not isinstance(other, Subtest):
            return False
        return self.name == other.name and self.test_name == other.test_name

    def __str__(self):
        return yatest_lib.tools.to_utf8(unicode(self))

    def __unicode__(self):
        return u"{}::{}".format(self.test_name, self.test_name)

    @property
    def name(self):
        return yatest_lib.tools.to_utf8(self._name)

    @property
    def test_name(self):
        return yatest_lib.tools.to_utf8(self._test_name)

    def __repr__(self):
        return "Subtest [{}::{} - {}[{}]: {}]".format(self.name, self.test_name, self.status, self.elapsed, self.comment)

    def __hash__(self):
        return hash(str(self))


class SubtestInfo(object):

    skipped_prefix = '[SKIPPED] '

    @classmethod
    def from_str(cls, s):
        if s.startswith(SubtestInfo.skipped_prefix):
            s = s[len(SubtestInfo.skipped_prefix):]
            skipped = True

        else:
            skipped = False

        return SubtestInfo(*s.rsplit(TEST_SUBTEST_SEPARATOR, 1), skipped=skipped)

    def __init__(self, test, subtest="", skipped=False, **kwargs):
        self.test = test
        self.subtest = subtest
        self.skipped = skipped
        for key, value in kwargs.iteritems():
            setattr(self, key, value)

    def __str__(self):
        s = ''

        if self.skipped:
            s += SubtestInfo.skipped_prefix

        return s + TEST_SUBTEST_SEPARATOR.join([self.test, self.subtest])

    def __repr__(self):
        return str(self)


class Status(object):
    GOOD, XFAIL, FAIL, XPASS, MISSING, CRASHED, TIMEOUT = range(7)
    SKIPPED = -100
    NOT_LAUNCHED = -200
    CANON_DIFF = -300
    FLAKY = -1
    BY_NAME = {'good': GOOD, 'fail': FAIL, 'xfail': XFAIL, 'xpass': XPASS, 'missing': MISSING, 'crashed': CRASHED,
               'skipped': SKIPPED, 'flaky': FLAKY, 'not_launched': NOT_LAUNCHED, 'timeout': TIMEOUT, 'diff': CANON_DIFF}
    TO_STR = {GOOD: 'good', FAIL: 'fail', XFAIL: 'xfail', XPASS: 'xpass', MISSING: 'missing', CRASHED: 'crashed',
              SKIPPED: 'skipped', FLAKY: 'flaky', NOT_LAUNCHED: 'not_launched', TIMEOUT: 'timeout', CANON_DIFF: 'diff'}


class Test(object):
    def __init__(self, name, path, status=None, comment=None, subtests=None):
        self.name = name
        self.path = path
        self.status = status
        self.comment = comment
        self.subtests = subtests or []

    def __eq__(self, other):
        if not isinstance(other, Test):
            return False
        return self.name == other.name and self.path == other.path

    def __str__(self):
        return "Test [{} {}] - {} - {}".format(self.name, self.path, self.status, self.comment)

    def __repr__(self):
        return str(self)

    def add_subtest(self, subtest):
        self.subtests.append(subtest)

    def setup_status(self, status, comment):
        self.status = Status.BY_NAME[status or 'good']
        if len(self.subtests) != 0:
            self.status = max(self.status, max(s.status for s in self.subtests))
        self.comment = comment

    def subtests_by_status(self, status):
        return [x.status for x in self.subtests].count(status)


class NoMd5FileException(Exception):
    pass


TEST_SUBTEST_SEPARATOR = '::'


# TODO: extract color theme logic from ya
COLOR_THEME = {
    'test_name': 'light-blue',
    'test_project_path': 'dark-blue',
    'test_dir_desc': 'dark-magenta',
    'test_binary_path': 'light-gray',
}


# XXX: remove me
class YaCtx(object):
    pass

ya_ctx = YaCtx()

TRACE_FILE_NAME = "ytest.report.trace"
TESTING_OUT_DIR_NAME = "testing_out_stuff"


def lazy(func):
    mem = {}

    @functools.wraps(func)
    def wrapper():
        if "results" not in mem:
            mem["results"] = func()
        return mem["results"]

    return wrapper


@lazy
def _get_mtab():
    if os.path.exists("/etc/mtab"):
        with open("/etc/mtab") as afile:
            data = afile.read()
        return [line.split(" ") for line in data.split("\n") if line]
    return []


def get_max_filename_length(dirname):
    """
    Return maximum filename length for the filesystem
    :return:
    """
    if sys.platform.startswith("linux"):
        # Linux user's may work on mounted ecryptfs filesystem
        # which has filename length limitations
        for entry in _get_mtab():
            mounted_dir, filesystem = entry[1], entry[2]
            # http://unix.stackexchange.com/questions/32795/what-is-the-maximum-allowed-filename-and-folder-size-with-ecryptfs
            if filesystem == "ecryptfs" and dirname and dirname.startswith(mounted_dir):
                return 140
    # default maximum filename length for most filesystems
    return 255


def get_unique_file_path(dir_path, filename):
    """
    Get unique filename in dir with proper filename length, using given filename
    :param dir_path: path to dir
    :param filename: original filename
    :return: unique filename
    """
    counter = 0
    max_suffix = 10000
    # + 1 symbol for dot before suffix
    tail_length = int(round(math.log(max_suffix, 10))) + 1
    # truncate filename length in accordance with filesystem limitations
    filename, extension = os.path.splitext(filename)
    # XXX
    if sys.platform.startswith("win"):
        # Trying to fit into MAX_PATH if it's possible.
        # Remove after DEVTOOLS-1646
        max_path = 260
        filename_len = len(dir_path) + len(extension) + tail_length + len(os.sep)
        if filename_len < max_path:
            filename = yatest_lib.tools.trim_string(filename, max_path - filename_len)
    filename = yatest_lib.tools.trim_string(filename, get_max_filename_length(dir_path) - tail_length - len(extension)) + extension
    candidate = os.path.join(dir_path, filename)
    while os.path.exists(candidate):
        counter += 1
        assert counter < max_suffix
        candidate = os.path.join(dir_path, filename + ".{}".format(counter))
    return candidate


def escape_for_fnmatch(s):
    return s.replace("[", "&#91;").replace("]", "&#93;")


def get_python_cmd(opts=None, use_huge=True, suite=None):
    if opts and getattr(opts, 'flags', {}).get("USE_ARCADIA_PYTHON") == "no":
        return ["python"]
    if suite and not suite._use_arcadia_python:
        return ["python"]
    if use_huge:
        return ["$(PYTHON)/python"]
    ymake_path = opts.ymake_bin if opts and getattr(opts, 'ymake_bin', None) else "$(YMAKE)/ymake"
    return [ymake_path, "--python"]


def normalize_name(name):
    replacements = [
        ("\\", "\\\\"),
        ("\n", "\\n"),
        ("\t", "\\t"),
        ("\r", "\\r"),
    ]
    for l, r in replacements:
        name = name.replace(l, r)
    return name


def normalize_filename(filename):
    """
    Replace invalid for file names characters with string equivalents
    :param some_string: string to be converted to a valid file name
    :return: valid file name
    """
    not_allowed_pattern = r"[\[\]\/:*?\"\'<>|+\0\\\s\x0b\x0c]"
    filename = re.sub(not_allowed_pattern, ".", filename)
    return re.sub(r"\.{2,}", ".", filename)


def get_test_log_file_path(output_dir, class_name, test_name, extension="log"):
    """
    get test log file path, platform dependant
    :param output_dir: dir where log file should be placed
    :param class_name: test class name
    :param test_name: test name
    :return: test log file name
    """
    if os.name == "nt":
        # don't add class name to the log's filename
        # to reduce it's length on windows
        filename = test_name
    else:
        filename = "{}.{}".format(class_name, test_name)
    if not filename:
        filename = "test"
    filename += "." + extension
    filename = normalize_filename(filename)
    return get_unique_file_path(output_dir, filename)


def split_node_id(nodeid, test_suffix=None):
    path, possible_open_bracket, params = nodeid.partition('[')
    separator = "::"
    if separator in path:
        path, test_name = path.split(separator, 1)
    else:
        test_name = os.path.basename(path)
    if test_suffix:
        test_name += "::" + test_suffix
    class_name = os.path.basename(path.strip())
    if separator in test_name:
        klass_name, test_name = test_name.split(separator, 1)
        if not test_suffix:
            # test suffix is used for flakes and pep8, no need to add class_name as it's === class_name
            class_name += separator + klass_name
    if separator in test_name:
        test_name = test_name.split(separator)[-1]
    test_name += possible_open_bracket + params
    return yatest_lib.tools.to_utf8(class_name), yatest_lib.tools.to_utf8(test_name)
