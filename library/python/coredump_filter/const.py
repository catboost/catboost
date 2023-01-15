from enum import Enum

TEST_DATA_SOURCE_PATH = 'library/python/coredump_filter/tests/data'


class CoredumpType(Enum):
    LLDB = 'LLDB'
    GDB = 'GDB'
