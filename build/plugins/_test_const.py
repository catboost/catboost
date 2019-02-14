# coding: utf-8
import re
import sys


RESTART_TEST_INDICATOR = '##restart-test##'
INFRASTRUCTURE_ERROR_INDICATOR = '##infrastructure-error##'

RESTART_TEST_INDICATORS = [
    RESTART_TEST_INDICATOR,
    "network error",
]

# testing
BIN_DIRECTORY = 'bin'
CANONIZATION_RESULT_FILE_NAME = "canonization_res.json"
CONSOLE_SNIPPET_LIMIT = 5000
LIST_NODE_LOG_FILE = "test_list.log"
LIST_NODE_RESULT_FILE = "test_list.json"
LIST_RESULT_NODE_LOG_FILE = "list_result.log"
MAX_FILE_SIZE = 1024 * 1024 * 2  # 2 MB
MAX_TEST_RESTART_COUNT = 3
REPORT_SNIPPET_LIMIT = 10000
SANITIZER_ERROR_RC = 100
TEST_SUBTEST_SEPARATOR = '::'
TESTING_OUT_DIR_NAME = "testing_out_stuff"
TESTING_OUT_TAR_NAME = TESTING_OUT_DIR_NAME + ".tar"
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S.%f"
TRACE_FILE_NAME = "ytest.report.trace"
TRUNCATING_IGNORE_FILE_LIST = {TRACE_FILE_NAME, "run_test.log"}

# distbuild
TEST_NODE_FINISHING_TIME = 5 * 60

# coverage
COVERAGE_TESTS_TIMEOUT_FACTOR = 1.5
COVERAGE_RESOLVED_FILE_NAME_PATTERN = "coverage_resolved.{}.json"
CPP_COVERAGE_RESOLVED_FILE_NAME = COVERAGE_RESOLVED_FILE_NAME_PATTERN.format("cpp")
JAVA_COVERAGE_RESOLVED_FILE_NAME = COVERAGE_RESOLVED_FILE_NAME_PATTERN.format("java")
PYTHON_COVERAGE_RESOLVED_FILE_NAME = COVERAGE_RESOLVED_FILE_NAME_PATTERN.format("python")
CLANG_COVERAGE_TEST_TYPES = ("unittest", "coverage_extractor", "pytest", "gtest", "boost_test", "exectest")
COVERAGE_TABLE_CHUNKS = 20
COVERAGE_YT_PROXY = "hahn.yt.yandex.net"
COVERAGE_YT_ROOT_PATH = "//home/codecoverage"
COVERAGE_YT_TABLE_PREFIX = "datatable"

# fuzzing
CORPUS_DATA_FILE_NAME = 'corpus.json'
CORPUS_DATA_ROOT_DIR = 'fuzzing'
CORPUS_DIR_NAME = 'corpus'
FUZZING_COMPRESSION_COEF = 1.1
FUZZING_DEFAULT_TIMEOUT = 3600
FUZZING_FINISHING_TIME = 600
FUZZING_TIMEOUT_RE = re.compile(r'(^|\s)-max_total_time=(?P<max_time>\d+)')
GENERATED_CORPUS_DIR_NAME = 'mined_corpus'
MAX_CORPUS_RESOURCES_ALLOWED = 5


class Enum(object):

    @classmethod
    def enumerate(cls):
        return [v for k, v in cls.__dict__.items() if not k.startswith("_")]


class TestRequirements(Enum):
    Container = 'container'
    Cpu = 'cpu'
    DiskUsage = 'disk_usage'
    Ram = 'ram'
    RamDisk = 'ram_disk'
    SbVault = 'sb_vault'
    Network = 'network'


class TestRequirementsConstants(Enum):
    All = 'all'
    AllCpuValue = 50
    AllRamDiskValue = 50
    MinCpu = 1
    MinRam = 1
    MinRamDisk = 0

    @classmethod
    def is_all_cpu(cls, value):
        return value == cls.All

    @classmethod
    def get_cpu_value(cls, value):
        return cls.AllCpuValue if cls.is_all_cpu(value) else value

    @classmethod
    def is_all_ram_disk(cls, value):
        return value == cls.All

    @classmethod
    def get_ram_disk_value(cls, value):
        return cls.AllRamDiskValue if cls.is_all_ram_disk(value) else value


class TestSize(Enum):
    Small = 'small'
    Medium = 'medium'
    Large = 'large'

    DefaultTimeouts = {
        Small: 60,
        Medium: 600,
        Large: 3600,
    }

    DefaultPriorities = {
        Small: -1,
        Medium: -2,
        Large: -3,
    }

    DefaultRequirements = {
        Small: {
            TestRequirements.Cpu: 1,
            TestRequirements.Ram: 32,
            # TestRequirements.Ram: 2,
            TestRequirements.RamDisk: 0,
        },
        Medium: {
            TestRequirements.Cpu: 1,
            TestRequirements.Ram: 32,
            # TestRequirements.Ram: 4,
            TestRequirements.RamDisk: 0,
        },
        Large: {
            TestRequirements.Cpu: 1,
            TestRequirements.Ram: 32,
            # TestRequirements.Ram: 8,
            TestRequirements.RamDisk: 0,
        },
    }

    MaxRequirements = {
        Small: {
            TestRequirements.Cpu: 4,
            TestRequirements.Ram: 32,
            # TestRequirements.Ram: 4,
            TestRequirements.RamDisk: 4,
        },
        Medium: {
            TestRequirements.Cpu: 4,
            # TestRequirements.Cpu: 8,
            TestRequirements.Ram: 32,
            # TestRequirements.Ram: 16,
            TestRequirements.RamDisk: 4,
        },
        Large: {
            TestRequirements.Cpu: 4,
            TestRequirements.Ram: 32,
            TestRequirements.RamDisk: 4,
        },
    }

    @classmethod
    def sizes(cls):
        return cls.DefaultTimeouts.keys()

    @classmethod
    def get_default_timeout(cls, size):
        if size in cls.DefaultTimeouts:
            return cls.DefaultTimeouts[size]
        raise Exception("Unknown test size '{}'".format(size))

    @classmethod
    def get_default_priorities(cls, size):
        if size in cls.DefaultPriorities:
            return cls.DefaultPriorities[size]
        raise Exception("Unknown test size '{}'".format(size))

    @classmethod
    def get_default_requirements(cls, size):
        if size in cls.DefaultRequirements:
            return cls.DefaultRequirements[size]
        raise Exception("Unknown test size '{}'".format(size))

    @classmethod
    def get_max_requirements(cls, size):
        if size in cls.MaxRequirements:
            return cls.MaxRequirements[size]
        raise Exception("Unknown test size '{}'".format(size))


class TestRunExitCode(Enum):
    Skipped = 2
    TimeOut = 10
    InfrastructureError = 12


class YaTestTags(Enum):
    Manual = "ya:manual"
    Notags = "ya:notags"
    Norestart = "ya:norestart"
    Dirty = "ya:dirty"
    Noretries = "ya:noretries"
    Fat = "ya:fat"
    RunWithAsserts = "ya:relwithdebinfo"
    Privileged = "ya:privileged"


class Status(object):
    GOOD, XFAIL, FAIL, XPASS, MISSING, CRASHED, TIMEOUT = range(1, 8)
    SKIPPED = -100
    NOT_LAUNCHED = -200
    CANON_DIFF = -300
    DESELECTED = -400
    INTERNAL = -sys.maxint
    FLAKY = -50
    BY_NAME = {'good': GOOD, 'fail': FAIL, 'xfail': XFAIL, 'xpass': XPASS, 'missing': MISSING, 'crashed': CRASHED,
               'skipped': SKIPPED, 'flaky': FLAKY, 'not_launched': NOT_LAUNCHED, 'timeout': TIMEOUT, 'diff': CANON_DIFF,
               'internal': INTERNAL, 'deselected': DESELECTED}
    TO_STR = {GOOD: 'good', FAIL: 'fail', XFAIL: 'xfail', XPASS: 'xpass', MISSING: 'missing', CRASHED: 'crashed',
              SKIPPED: 'skipped', FLAKY: 'flaky', NOT_LAUNCHED: 'not_launched', TIMEOUT: 'timeout', CANON_DIFF: 'diff',
              INTERNAL: 'internal', DESELECTED: 'deselected'}


class _Colors(object):

    _NAMES = [
        "blue",
        "cyan",
        "default",
        "green",
        "grey",
        "magenta",
        "red",
        "white",
        "yellow",
    ]
    _PREFIXES = ["", "light"]

    def __init__(self):
        self._table = {}
        for prefix in self._PREFIXES:
            for value in self._NAMES:
                name = value
                if prefix:
                    name = "{}_{}".format(prefix, value)
                    value = "{}-{}".format(prefix, value)
                self.__add_color(name.upper(), value)

    def __add_color(self, name, value):
        self._table[name] = value
        self.__setattr__(name, value)


Colors = _Colors()


class _Highlight(object):

    _MARKERS = {
        # special
        "RESET": "rst",

        "IMPORTANT": "imp",
        "UNIMPORTANT": "unimp",
        "BAD": "bad",
        "WARNING": "warn",
        "GOOD": "good",
        "PATH": "path",
        "ALTERNATIVE1": "alt1",
        "ALTERNATIVE2": "alt2",
        "ALTERNATIVE3": "alt3",
    }

    def __init__(self):
        # setting attributes because __getattr__ is much slower
        for attr, value in self._MARKERS.items():
            self.__setattr__(attr, value)


Highlight = _Highlight()


class _StatusColorMap(object):

    _MAP = {
        'good': Highlight.GOOD,
        'fail': Highlight.BAD,
        'missing': Highlight.ALTERNATIVE1,
        'crashed': Highlight.WARNING,
        'skipped': Highlight.UNIMPORTANT,
        'not_launched': Highlight.BAD,
        'timeout': Highlight.BAD,
        'flaky': Highlight.ALTERNATIVE3,
        'xfail': Highlight.WARNING,
        'xpass': Highlight.BAD,
        'diff': Highlight.BAD,
        'internal': Highlight.BAD,
        'deselected': Highlight.UNIMPORTANT,
    }

    def __getitem__(self, item):
        return self._MAP[item]


StatusColorMap = _StatusColorMap()
