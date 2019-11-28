import os
import logging
from yatest import common as ytc

logger = logging.getLogger("test_logger")


class TestArchiver(object):
    @classmethod
    def setup_class(cls):
        cls.archiver_path = ytc.binary_path("tools/archiver/archiver")

    def test_recursive(self):
        assert 'archiver' == os.path.basename(self.archiver_path)
        assert os.path.exists(self.archiver_path)
        contents = ytc.source_path("tools/archiver/tests/directory")
        ytc.execute(
            command=[
                self.archiver_path,
                "--output", "archive",
                "--recursive",
                contents,
            ]
        )
        with open('result', 'w') as archive_list:
            ytc.execute(
                command=[
                    self.archiver_path,
                    "--list",
                    "archive",
                ],
                stdout=archive_list,
                stderr=None,
            )
        archive_list = sorted(open('result').read().strip().split('\n'))
        assert len(archive_list) == 3
        assert archive_list[0] == 'file1'
        assert archive_list[1] == 'file2'
        assert archive_list[2] == 'file3'

    def test_deduplicate(self):
        assert 'archiver' == os.path.basename(self.archiver_path)
        assert os.path.exists(self.archiver_path)
        contents = ytc.source_path("tools/archiver/tests/directory")
        ytc.execute(
            command=[
                self.archiver_path,
                "--output", "result_dedup",
                "--recursive",
                "--deduplicate",
                "--plain",
                contents,
            ]
        )
        ytc.execute(
            command=[
                self.archiver_path,
                "--output", "result_no_dedup",
                "--recursive",
                "--plain",
                contents,
            ]
        )
        with open('result_dedup', 'rb') as f_dedup, open('result_no_dedup', 'rb') as f_no_dedup:
            archive_dedup = f_dedup.read()
            archive_no_dedup = f_no_dedup.read()
        assert len(archive_dedup) == 58
        assert len(archive_no_dedup) == 75
