import os
import yatest.common

from library.python.coredump_filter import core_proc, const


def test_coredump_detect_type():
    source_path = yatest.common.source_path('library/python/coredump_filter/tests/data')
    answers = {
        'typical_lldb_coredump.txt': const.CoredumpType.LLDB,
        'typical_gdb_coredump.txt': const.CoredumpType.GDB,
    }
    for filename, fmt in answers.items():
        with open(os.path.join(source_path, filename)) as fd:
            core_text = '\n'.join(fd.readlines())
            core_type = core_proc.detect_coredump_type(core_text)
            assert core_type == fmt
