import logging


def add_handler(handler):
    class LogHandler(object):
        def __init__(self, handler):
            self._handler = handler

        def __enter__(self):
            if self._handler:
                logging.getLogger().addHandler(self._handler)
            return self._handler

        def __exit__(self, exc_type, exc_value, traceback):
            if self._handler:
                logging.getLogger().removeHandler(self._handler)

    return LogHandler(handler)
