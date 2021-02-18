import unittest
import yatest.common
import os
from library.python.coredump_filter import core_proc, const

AMOUNT_TEST_CASES = 7


class MockStream:
    def __init__(self):
        self._store = []

    def write(self, payload):
        self._store.append(payload)

    @property
    def get(self):
        return '\n'.join(self._store)


class TestFingerprint(unittest.TestCase):
    def setUp(self):
        self.fingerprints = {}
        self.test_cases = {}
        self.data_dir = yatest.common.source_path(const.TEST_DATA_SOURCE_PATH)

        for test_case_id in range(1, AMOUNT_TEST_CASES + 1):
            filename = 'test{}.txt'.format(test_case_id)
            with open(os.path.join(self.data_dir, filename)) as fd:
                stream = MockStream()
                parsed_lines = fd.readlines()
                core_type = core_proc.detect_coredump_type(''.join(parsed_lines))
                if core_type == const.CoredumpType.GDB:
                    core_proc.filter_stackdump(
                        file_lines=parsed_lines,
                        use_fingerprint=True,
                        stream=stream,
                    )
                else:
                    core_proc.filter_stackdump_lldb(
                        file_lines=parsed_lines,
                        use_fingerprint=True,
                        stream=stream,
                    )
                self.fingerprints[test_case_id] = stream.get

            with open(os.path.join(self.data_dir, 'test{}.txt.fp'.format(test_case_id))) as fd:
                parsed_lines = fd.read()
                self.test_cases[test_case_id] = parsed_lines

    def test_fingerprint(self):
        for test_case_id, test_fingerprints in self.test_cases.items():
            for fg in test_fingerprints:
                self.assertIn(fg, self.fingerprints[test_case_id], 'Fingerprint not found.')
