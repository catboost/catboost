import sys
import threading
import logging

from six.moves import queue as Queue
from six import reraise

logger = logging.getLogger(__name__)


def wrap(f):
    try:
        return f(), None
    except Exception as e:
        try:
            if not getattr(e, 'tame', False):
                import traceback

                logger.debug('async exception, %s', traceback.format_exc())
        except Exception:
            pass

        return None, sys.exc_info()


def unwrap(res):
    r, e = res

    if e:
        reraise(e[0], e[1], e[2])

    return r


class ProperEvent(object):
    def __init__(self):
        self._q = Queue.Queue()
        self._m = threading.Lock()

    def set(self, v):
        self._q.put(v)

    def do_wait(self):
        while True:
            try:
                return self._q.get(timeout=1)
            except Queue.Empty:
                pass

    def wait(self):
        with self._m:
            try:
                self._v
            except AttributeError:
                self._v = self.do_wait()
            return self._v


def asyncthread(f, daemon=True):
    e = ProperEvent()

    thr = threading.Thread(target=lambda: e.set(wrap(f)))
    thr.daemon = daemon
    thr.start()

    return e.wait


def future(f, daemon=True):
    e = asyncthread(f, daemon)

    return lambda: unwrap(e())


def apply_parallel(funcs):
    def iter_items():
        return [asyncthread(f) for f in funcs]

    for y in [x() for x in iter_items()]:
        yield unwrap(y)


def par_map(func, data, threads):
    def chunks(lst, size):
        while lst:
            yield lst[:size]

            lst = lst[size:]

    def calc_chunk(items):
        return lambda: list(map(func, items))

    return sum(apply_parallel([calc_chunk(chunk) for chunk in chunks(data, len(data) // threads + 1)]), [])


class CancellableTimer(object):
    def __init__(self, action, timeout):
        self._cancelled = threading.Event()
        self._timer = threading.Thread(target=self._run, args=(action, self._cancelled, timeout))

    def start(self, cancel_on=None):
        self._timer.start()
        if cancel_on is not None:
            cancel_on()
            self.cancel()

    def cancel(self):
        self._cancelled.set()

    @staticmethod
    def _run(action, cancelled, timeout):
        if not cancelled.wait(timeout):
            action()
