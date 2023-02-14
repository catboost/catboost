import time
import logging


logger = logging.getLogger(__name__)


class Timer(object):
    def __init__(self, name):
        self._start_time = time.time()
        self._last_step_time = self._start_time
        self._name = name

    def show_step(self, arg):
        """Returns time from last step"""
        t = time.time()

        duration = t - self._last_step_time
        logger.debug("Timer %s, stage %s: %s", self._name, arg, duration)
        self._last_step_time = t

        return duration

    def full_duration(self):
        """Return time between last step and timer starts"""
        return self._last_step_time - self._start_time


class AccumulateTime(object):
    def __init__(self, acc_func):
        self.acc_func = acc_func

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *exc_details):
        finish_time = time.time()
        self.acc_func(finish_time - self.start_time)
        return False
