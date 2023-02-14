import logging
import os
import threading

import library.python.compress as lpc


class UCompressor(object):
    def __init__(self, filename, codec, mode=None):
        rd, wd = os.pipe()
        self.uc_stream = os.fdopen(rd, 'rb')

        def _open(path, mode_):
            if 'r' in mode_:
                return self.uc_stream
            return open(path, mode_)

        logging.getLogger('compress').setLevel(logging.ERROR)
        self.uc_thread = threading.Thread(
            target=lpc.compress, args=(None, filename), kwargs={'codec': codec, 'fopen': _open}
        )
        self.uc_thread.daemon = True
        self.uc_thread_started = False
        self.in_stream = os.fdopen(wd, mode or 'wb')

    def getInputStream(self):
        assert self.uc_thread_started, 'not running'
        return self.in_stream

    def isStarted(self):
        return self.uc_thread_started

    def start(self):
        assert not self.uc_thread_started, 'can only be started once'
        assert not self.uc_thread.is_alive()
        self.uc_thread_started = True
        self.uc_thread.start()

    def stop(self):
        assert self.uc_thread_started, 'not running'
        assert self.uc_thread.is_alive()
        try:
            try:
                self.in_stream.close()
            finally:
                self.in_stream = None
                self.uc_thread.join()
        finally:
            self.uc_stream.close()

    def __enter__(self):
        self.start()
        return self.getInputStream()

    def __exit__(self, *exc_details):
        self.stop()


def ucopen(filename, mode=None):
    # Default result file-object mode is 'wb' for compressed and 'w' for raw file.
    # Though it makes Python 3 usage inconsistent, it is made to preserve compatibility.
    return UCompressor(filename, 'zstd_1', mode or 'wb') if filename.endswith('.uc') else open(filename, mode or 'w')
