import json
import os

import library.python.import_tracing.lib.converters.base as base_converter


class ChromiumTraceConverter(base_converter.BaseTraceConverter):
    @staticmethod
    def _yield_in_chrome_trace_format(events, pid):
        for event in events:
            yield {
                "cat": event.modname,
                "name": event.filename,
                "ph": "B",
                "ts": event.start_time,
                "pid": pid,
                "tid": event.tid,
                "args": {},
            }

            yield {
                "cat": event.modname,
                "name": event.filename,
                "ph": "E",
                "ts": event.end_time,
                "pid": pid,
                "tid": event.tid,
            }

    def dump(self, events, filepath):
        pid = os.getpid()
        with open(filepath, "w") as file:
            file.write(json.dumps(tuple(self._yield_in_chrome_trace_format(events, pid))))
