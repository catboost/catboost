import json
import sys
from enum import Enum
from typing import Optional


class LintStatus(Enum):
    GOOD = "GOOD"
    FAIL = "FAIL"
    SKIPPED = "SKIPPED"


class LintReport():
    def __init__(self):
        self._report = {}

    def add(self, file_name: str, status: LintStatus, message: str = "", elapsed: float = 0.0):
        self._report[file_name] = {
            "status": status.value,
            "message": message,
            "elapsed": elapsed,
        }

    def dump(self, report_file, pretty: Optional[bool] = None):
        data = {
            "report": self._report,
        }
        if report_file == "-":
            if pretty is None:
                pretty = True
            self._do_dump(sys.stdout, data, pretty)
        else:
            with open(report_file, "w") as f:
                self._do_dump(f, data, pretty)

    @staticmethod
    def _do_dump(dest, data, pretty):
        indent = 4 if pretty else None
        json.dump(data, dest, indent=indent)
