import threading
import time
import collections
import library.python.import_tracing.lib.event as events
import library.python.import_tracing.lib.constants as constants


class ImportTracer:
    def __init__(self):
        self.events = collections.OrderedDict()
        self.start_time = time.time()

    def start_event(self, modname, filename, tid=None):
        tid = tid if tid is not None else threading.current_thread().name
        time_from_start = self._get_current_time_from_start()

        new_event = events.Event(
            modname=modname,
            filename=filename,
            tid=tid,
            start_time=time_from_start,
            end_time=None,
        )

        self.events[modname] = new_event

    def finish_event(self, modname, filename, tid=None):
        event = self.events[modname]
        end_time = self._get_current_time_from_start()
        event.end_time = end_time

    def get_events(self, close_not_finished=False):
        end_time = self._get_current_time_from_start()

        for event in self.events.values():
            if close_not_finished and event.end_time is None:
                yield events.Event(
                    modname=event.modname,
                    filename=event.filename,
                    tid=event.tid,
                    start_time=event.start_time,
                    end_time=end_time,
                )
            else:
                yield event

    def _get_current_time_from_start(self):
        return (time.time() - self.start_time) * constants.MCS_IN_SEC
