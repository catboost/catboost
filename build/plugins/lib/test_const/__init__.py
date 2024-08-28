# coding: utf-8
import re

TEST_BT_COLORS = {
    "function_name": "[[alt1]]",
    "function_arg": "[[good]]",
    "stack_frame": "[[bad]]",
    "thread_prefix": "[[alt3]]",
    "thread_id": "[[bad]]",
    "file_path": "[[warn]]",
    "line_num": "[[alt2]]",
    "address": "[[unimp]]",
}

RESTART_TEST_INDICATOR = '##restart-test##'
INFRASTRUCTURE_ERROR_INDICATOR = '##infrastructure-error##'

RESTART_TEST_INDICATORS = [
    RESTART_TEST_INDICATOR,
    "network error",
]

UID_PREFIX_DELIMITER = '-'

# testing
BIN_DIRECTORY = 'bin'
CANON_DATA_DIR_NAME = "canondata"
CANON_RESULT_FILE_NAME = "result.json"
CANONIZATION_RESULT_FILE_NAME = "canonization_res.json"
COMMON_CONTEXT_FILE_NAME = "common_test.context"
CONSOLE_SNIPPET_LIMIT = 5000
FAKE_OUTPUT_EXTS = frozenset([".mf", ".fake", ".cpf", ".cpsf"])
LIST_NODE_LOG_FILE = "test_list.log"
LIST_NODE_RESULT_FILE = "test_list.json"
LIST_RESULT_NODE_LOG_FILE = "list_result.log"
LIST_TRACE_FILE_NAME = "ytest_list.report.trace"
MAX_FILE_SIZE = 1024 * 1024 * 2  # 2 MB
MAX_TEST_RESTART_COUNT = 3
NO_LISTED_TESTS = "NO_LISTED_TESTS"
REPORT_SNIPPET_LIMIT = 12000
SANITIZER_ERROR_RC = 100
SUITE_CONTEXT_FILE_NAME = "test.context"
TEST_LIST_FILE = "test_names_list.json"
TEST_SUBTEST_SEPARATOR = '::'
TESTING_OUT_DIR_NAME = "testing_out_stuff"
TESTING_OUT_RAM_DRIVE_DIR_NAME = "ram_drive_output"
TESTING_OUT_TAR_NAME = TESTING_OUT_DIR_NAME + ".tar.zstd"
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S.%f"
TRACE_FILE_NAME = "ytest.report.trace"
TRUNCATING_IGNORE_FILE_LIST = {TRACE_FILE_NAME, SUITE_CONTEXT_FILE_NAME, "run_test.log"}
YT_RUN_TEST_DIR_NAME = "yt_run_test"
YT_RUN_TEST_TAR_NAME = "yt_run_test.tar"
COVERAGE_CFLAGS = ["-fprofile-instr-generate", "-fcoverage-mapping", "-DCLANG_COVERAGE"]
COVERAGE_LDFLAGS = ["-fprofile-instr-generate", "-fcoverage-mapping"]

CANON_BACKEND_KEY = "{canondata_backend}"
DEFAULT_CANONIZATION_BACKEND = "storage.yandex-team.ru/get-devtools"
MDS_URI_PREFIX = 'https://storage.yandex-team.ru/get-devtools/'
BACKEND_URI_PREFIX = 'https://' + CANON_BACKEND_KEY + '/'
MDS_SCHEME = 'mds'
CANON_MDS_RESOURCE_REGEX = re.compile(re.escape(MDS_URI_PREFIX) + r'(.*?)($|#)')
CANON_BACKEND_RESOURCE_REGEX = re.compile(re.escape(BACKEND_URI_PREFIX) + r'(.*?)($|#)')
CANON_SBR_RESOURCE_REGEX = re.compile(r'(sbr:/?/?(\d+))')

MANDATORY_ENV_VAR_NAME = 'YA_MANDATORY_ENV_VARS'

STYLE_CPP_SOURCE_EXTS = [".cpp", ".cxx", ".cc", ".c", ".C"]
STYLE_CPP_HEADER_EXTS = [".h", ".H", ".hh", ".hpp", ".hxx", ".ipp"]
STYLE_CPP_ALL_EXTS = STYLE_CPP_SOURCE_EXTS + STYLE_CPP_HEADER_EXTS

BUILD_FLAGS_ALLOWED_IN_CONTEXT = {
    'AUTOCHECK',
    # Required for local test runs
    'TESTS_REQUESTED',
    'USE_ARCADIA_PYTHON',
    'USE_SYSTEM_PYTHON',
}

TEST_NODE_OUTPUT_RESULTS = [TESTING_OUT_TAR_NAME, YT_RUN_TEST_TAR_NAME]

# kvm
DEFAULT_RAM_REQUIREMENTS_FOR_KVM = 4
MAX_RAM_REQUIREMENTS_FOR_KVM = 16

# distbuild
DISTBUILD_STATUS_REPORT_ENV_NAME = 'NODE_EXTENDED_STATUS_FILE_PATH'
DEFAULT_TEST_NODE_TIMEOUT = 15 * 60
TEST_NODE_FINISHING_TIME = 5 * 60

# coverage
COVERAGE_FUNCTION_ENTRIES_LIMIT = 2
COVERAGE_PYTHON_EXTS = (".py", ".pyx", ".pxi", ".pxd")

COVERAGE_RESOLVED_FILE_NAME_PATTERN = "coverage_resolved.{}.json"
CPP_COVERAGE_RESOLVED_FILE_NAME = COVERAGE_RESOLVED_FILE_NAME_PATTERN.format("cpp")
GO_COVERAGE_RESOLVED_FILE_NAME = COVERAGE_RESOLVED_FILE_NAME_PATTERN.format("go")
JAVA_COVERAGE_RESOLVED_FILE_NAME = COVERAGE_RESOLVED_FILE_NAME_PATTERN.format("java")
NLG_COVERAGE_RESOLVED_FILE_NAME = COVERAGE_RESOLVED_FILE_NAME_PATTERN.format("nlg")
PYTHON2_COVERAGE_RESOLVED_FILE_NAME = COVERAGE_RESOLVED_FILE_NAME_PATTERN.format("py2")
PYTHON3_COVERAGE_RESOLVED_FILE_NAME = COVERAGE_RESOLVED_FILE_NAME_PATTERN.format("py3")
TS_COVERAGE_RESOLVED_FILE_NAME = COVERAGE_RESOLVED_FILE_NAME_PATTERN.format("ts")

COVERAGE_CLANG_ENV_NAME = 'LLVM_PROFILE_FILE'
COVERAGE_GCOV_ENV_NAME = 'GCOV_PREFIX'
COVERAGE_GO_ENV_NAME = 'GO_COVERAGE_PREFIX'
COVERAGE_PYTHON_ENV_NAME = 'PYTHON_COVERAGE_PREFIX'
COVERAGE_TS_ENV_NAME = 'TS_COVERAGE_PREFIX'
COVERAGE_NLG_ENV_NAME = 'NLG_COVERAGE_FILENAME'
COVERAGE_ENV_VARS = (
    COVERAGE_CLANG_ENV_NAME,
    COVERAGE_GCOV_ENV_NAME,
    COVERAGE_GO_ENV_NAME,
    COVERAGE_NLG_ENV_NAME,
    COVERAGE_PYTHON_ENV_NAME,
    COVERAGE_TS_ENV_NAME,
)
PYTHON_COVERAGE_PREFIX_FILTER_ENV_NAME = 'PYTHON_COVERAGE_PREFIX_FILTER'
PYTHON_COVERAGE_EXCLUDE_REGEXP_ENV_NAME = 'PYTHON_COVERAGE_EXCLUDE_REGEXP'

# TODO get rid of this list - resolve nodes should be added automatically depending on the lang of the target module and their deps
CLANG_COVERAGE_TEST_TYPES = (
    "boost_test",
    "coverage_extractor",
    "exectest",
    "fuzz",
    "gtest",
    "go_test",
    # java tests might use shared libraries
    "java",
    "py2test",
    "py3test",
    "pytest",
    "unittest",
)

COVERAGE_TABLE_CHUNKS = 20
COVERAGE_TESTS_TIMEOUT_FACTOR = 1.5
COVERAGE_YT_PROXY = "hahn.yt.yandex.net"
COVERAGE_YT_ROOT_PATH = "//home/codecoverage"
COVERAGE_YT_TABLE_PREFIX = "datatable"

# fuzzing
CORPUS_DATA_FILE_NAME = 'corpus.json'
CORPUS_DATA_ROOT_DIR = 'fuzzing'
CORPUS_DIR_NAME = 'corpus'
FUZZING_COVERAGE_ARGS = ['--sanitize-coverage=trace-div,trace-gep']
FUZZING_COMPRESSION_COEF = 1.1
FUZZING_DEFAULT_TIMEOUT = 3600
FUZZING_FINISHING_TIME = 600
FUZZING_TIMEOUT_RE = re.compile(r'(^|\s)-max_total_time=(?P<max_time>\d+)')
GENERATED_CORPUS_DIR_NAME = 'mined_corpus'
MAX_CORPUS_RESOURCES_ALLOWED = 5

# hermione
HERMIONE_REPORT_DIR_NAME = "hermione-report"
HERMIONE_REPORT_TAR_NAME = HERMIONE_REPORT_DIR_NAME + ".tar"
HERMIONE_REPORT_INDEX_FILE_NAME = "index.html"
HERMIONE_REPORT_DB_URLS_FILE_NAME = "databaseUrls.json"
HERMIONE_TESTS_READ_FILE_NAME = "tests.json"
HERMIONE_TESTS_READ_STDOUT_FILE_NAME = "read_tests.out"
HERMIONE_TESTS_READ_STDERR_FILE_NAME = "read_tests.err"
HERMIONE_TESTS_RUN_FILE_NAME = "test_results.jsonl"
HERMIONE_TESTS_RUN_STDOUT_FILE_NAME = "run_tests.out"
HERMIONE_TESTS_RUN_STDERR_FILE_NAME = "run_tests.err"

# yt
YT_OPERATION_ID_SUBSTITUTION = '$OPERATION_ID'
YT_SANDBOX_ROOT_PREFIX = '$(YT_SANDBOX_ROOT)'

# sandbox
SANDBOX_RUN_TEST_YT_TOKEN_VALUE_NAME = 'YA_MAKE_SANDBOX_RUN_TEST_YT_TOKEN'

# global resources
ANDROID_AVD_ROOT = 'ANDROID_AVD_RESOURCE_GLOBAL'
ANDROID_SDK_ROOT = 'ANDROID_SDK_RESOURCE_GLOBAL'
COVERAGE_PUSH_TOOL_LOCAL = 'USE_SYSTEM_COVERAGE_PUSH_TOOL'
COVERAGE_PUSH_TOOL_RESOURCE = 'COVERAGE_PUSH_TOOL_RESOURCE_GLOBAL'
COVERAGE_PUSH_TOOL_LB_LOCAL = 'USE_SYSTEM_COVERAGE_PUSH_TOOL_LB'
COVERAGE_PUSH_TOOL_LB_RESOURCE = 'COVERAGE_PUSH_TOOL_LB_RESOURCE_GLOBAL'
FLAKE8_PY2_RESOURCE = 'FLAKE8_PY2_RESOURCE_GLOBAL'
FLAKE8_PY3_RESOURCE = 'FLAKE8_PY3_RESOURCE_GLOBAL'
GO_TOOLS_RESOURCE = 'GO_TOOLS_RESOURCE_GLOBAL'
JSTYLE_RUNNER_LIB = 'JSTYLE_LIB_RESOURCE_GLOBAL'
NODEJS_RESOURCE = 'NODEJS_RESOURCE_GLOBAL'
NYC_RESOURCE = 'NYC_RESOURCE_GLOBAL'
RUFF_RESOURCE = 'RUFF_RESOURCE_GLOBAL'

# test_tool resource for host platform.
# source - build/platform/test_tool/host.ya.make.inc.
# always using this test_tool resource except 2 cases:
# 1. when we use TEST_TOOL_TARGET
# 2. when --test-tool-bin passed
TEST_TOOL_HOST = 'TEST_TOOL_HOST_RESOURCE_GLOBAL'

# path to locally built test_tool passed by --test-tool-bin opt
TEST_TOOL_HOST_LOCAL = 'TEST_TOOL_HOST_LOCAL'

# test_tool resource for target platform.
# source - build/platform/test_tool/ya.make.
# The only usage of this resource is running tests under ios emulator
TEST_TOOL_TARGET = 'TEST_TOOL_TARGET_RESOURCE_GLOBAL'

# path to locally built test_tool passed by --test-tool-bin opt
# always same as TEST_TOOL_HOST_LOCAL
# The only usage of this path is running tests under ios emulator
TEST_TOOL_TARGET_LOCAL = 'TEST_TOOL_TARGET_LOCAL'

XCODE_TOOLS_RESOURCE = 'XCODE_TOOLS_ROOT_RESOURCE_GLOBAL'
WINE_TOOL = 'WINE_TOOL_RESOURCE_GLOBAL'
WINE32_TOOL = 'WINE32_TOOL_RESOURCE_GLOBAL'


class Enum(object):
    @classmethod
    def enumerate(cls):
        return [v for k, v in cls.__dict__.items() if not k.startswith("_")]


class SuiteClassType(Enum):
    UNCLASSIFIED = '0'
    REGULAR = '1'
    STYLE = '2'


class TestRequirements(Enum):
    Container = 'container'
    Cpu = 'cpu'
    DiskUsage = 'disk_usage'
    Dns = 'dns'
    Kvm = 'kvm'
    Network = 'network'
    Ram = 'ram'
    RamDisk = 'ram_disk'
    SbVault = 'sb_vault'
    YavSecret = 'yav'


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
            TestRequirements.Ram: 8,
            # TestRequirements.Ram: 2,
            TestRequirements.RamDisk: 0,
        },
        Medium: {
            TestRequirements.Cpu: 1,
            TestRequirements.Ram: 8,
            # TestRequirements.Ram: 4,
            TestRequirements.RamDisk: 0,
        },
        Large: {
            TestRequirements.Cpu: 1,
            TestRequirements.Ram: 8,
            # TestRequirements.Ram: 8,
            TestRequirements.RamDisk: 0,
        },
    }

    MaxRequirements = {
        Small: {
            TestRequirements.Cpu: 4,
            TestRequirements.Ram: 32,
            # TestRequirements.Ram: 4,
            TestRequirements.RamDisk: 32,
        },
        Medium: {
            TestRequirements.Cpu: 4,
            # TestRequirements.Cpu: 8,
            TestRequirements.Ram: 32,
            # TestRequirements.Ram: 16,
            TestRequirements.RamDisk: 32,
        },
        Large: {
            TestRequirements.Cpu: 4,
            TestRequirements.Ram: 32,
            TestRequirements.RamDisk: 32,
        },
    }

    LargeMarker = "TL"
    MediumMarker = "TM"
    SmallMarker = "TS"
    SizeMarkers = (LargeMarker, MediumMarker, SmallMarker)

    SizeShorthandMap = {
        Large: LargeMarker,
        Medium: MediumMarker,
        Small: SmallMarker,
    }

    @classmethod
    def sizes(cls):
        return cls.DefaultTimeouts.keys()

    @classmethod
    def get_shorthand(cls, size):
        return cls.SizeShorthandMap[size]

    @classmethod
    def is_test_shorthand(cls, name):
        return name in cls.SizeMarkers

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


class ModuleLang(Enum):
    ABSENT = "absent"
    NUMEROUS = "numerous"
    UNKNOWN = "unknown"
    CPP = "cpp"
    DOCS = "docs"
    GO = "go"
    JAVA = "java"
    KOTLIN = "kotlin"
    PY = "py"
    TS = "ts"


class NodeType(Enum):
    TEST = "test"
    TEST_AUX = "test-aux"
    TEST_RESULTS = "test-results"
    DOWNLOAD = "download"


class TestRunExitCode(Enum):
    Skipped = 2
    Failed = 3
    TimeOut = 10
    InfrastructureError = 12


class YaTestTags(Enum):
    AlwaysMinimize = "ya:always_minimize"
    CopyData = "ya:copydata"
    CopyDataRO = "ya:copydataro"
    Dirty = "ya:dirty"
    DumpNodeEnvironment = "ya:dump_node_env"
    DumpTestEnvironment = "ya:dump_test_env"
    ExoticPlatform = "ya:exotic_platform"
    External = "ya:external"
    Fat = "ya:fat"
    ForceDistbuild = "ya:force_distbuild"
    ForceSandbox = "ya:force_sandbox"
    GoNoSubtestReport = "ya:go_no_subtest_report"
    GoTotalReport = "ya:go_total_report"
    HugeLogs = "ya:huge_logs"
    JavaTmpInRamDisk = "ya:java_tmp_in_ram_disk"
    Manual = "ya:manual"
    MapRootUser = "ya:map_root_user"
    NoGracefulShutdown = "ya:no_graceful_shutdown"
    NoPstreeTrim = "ya:no_pstree_trim"
    Norestart = "ya:norestart"
    Noretries = "ya:noretries"
    NotAutocheck = "ya:not_autocheck"
    Notags = "ya:notags"
    PerfTest = "ya:perftest"
    Privileged = "ya:privileged"
    ReportChunks = "ya:report_chunks"
    RunWithAsserts = "ya:relwithdebinfo"
    SandboxCoverage = "ya:sandbox_coverage"
    SequentialRun = "ya:sequential_run"
    TraceOutput = "ya:trace_output"
    YtRunner = "ya:yt"


class ServiceTags(Enum):
    AnyTag = "ya:anytag"


class Status(object):
    GOOD, XFAIL, FAIL, XPASS, MISSING, CRASHED, TIMEOUT = range(1, 8)
    SKIPPED = -100
    NOT_LAUNCHED = -200
    CANON_DIFF = -300
    DESELECTED = -400
    INTERNAL = -int(2**31 - 1)  # maxint
    FLAKY = -50
    # XFAILDIFF is internal status and should be replaced
    # with XFAIL or XPASS during verification stage of canon data
    XFAILDIFF = -90

    BY_NAME = {
        'crashed': CRASHED,
        'deselected': DESELECTED,
        'diff': CANON_DIFF,
        'fail': FAIL,
        'flaky': FLAKY,
        'good': GOOD,
        'internal': INTERNAL,
        'missing': MISSING,
        'not_launched': NOT_LAUNCHED,
        'skipped': SKIPPED,
        'timeout': TIMEOUT,
        'xfail': XFAIL,
        'xfaildiff': XFAILDIFF,
        'xpass': XPASS,
    }
    TO_STR = {
        CANON_DIFF: 'diff',
        CRASHED: 'crashed',
        DESELECTED: 'deselected',
        FAIL: 'fail',
        FLAKY: 'flaky',
        GOOD: 'good',
        INTERNAL: 'internal',
        MISSING: 'missing',
        NOT_LAUNCHED: 'not_launched',
        SKIPPED: 'skipped',
        TIMEOUT: 'timeout',
        XFAIL: 'xfail',
        XFAILDIFF: 'xfaildiff',
        XPASS: 'xpass',
    }


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
    _PREFIXES = ["", "light", "dark"]

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
    # There should be no XFAILDIFF, because it's internal status.
    # It should be replaced with XFAIL or XPASS during verification of canon data.

    _MAP = {
        'crashed': Highlight.WARNING,
        'deselected': Highlight.UNIMPORTANT,
        'diff': Highlight.BAD,
        'fail': Highlight.BAD,
        'flaky': Highlight.ALTERNATIVE3,
        'good': Highlight.GOOD,
        'internal': Highlight.BAD,
        'missing': Highlight.ALTERNATIVE1,
        'not_launched': Highlight.BAD,
        'skipped': Highlight.UNIMPORTANT,
        'timeout': Highlight.BAD,
        'xfail': Highlight.WARNING,
        'xpass': Highlight.WARNING,
    }

    def __getitem__(self, item):
        return self._MAP[item]


StatusColorMap = _StatusColorMap()
