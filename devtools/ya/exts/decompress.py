import logging
import os
import threading

import library.python.compress as lpc


class UDecompressor(object):
    def __init__(self, filename, mode=None):
        rd, wd = os.pipe()
        self.uc_stream = os.fdopen(wd, 'wb')

        def _target():
            def _open(path, mode_):
                if 'w' in mode_:
                    return self.uc_stream
                return open(path, mode_)

            try:
                try:
                    lpc.decompress(filename, None, fopen=_open)
                finally:
                    self.uc_stream.close()
            finally:
                self.uc_stream = None

        logging.getLogger('compress').setLevel(logging.ERROR)
        self.uc_thread = threading.Thread(target=_target)
        self.uc_thread.daemon = True
        self.uc_thread_started = False
        self.out_stream = os.fdopen(rd, mode or 'rb')

    def getOutputStream(self):
        assert self.uc_thread_started, 'not running'
        return self.out_stream

    def isStarted(self):
        return self.uc_thread_started

    def start(self):
        assert not self.uc_thread_started, 'can only be started once'
        assert not self.uc_thread.is_alive()
        self.uc_thread_started = True
        self.uc_thread.start()

    def stop(self):
        assert self.uc_thread_started, 'not running'
        assert self.out_stream is not None
        try:
            self.out_stream.close()
        finally:
            self.out_stream = None
            self.uc_thread.join()

    def __enter__(self):
        self.start()
        return self.getOutputStream()

    def __exit__(self, *exc_details):
        self.stop()


def udopen(filename, mode=None):
    # Default result file-object mode is 'rb' for compressed and 'r' for raw file.
    # Though it makes Python 3 usage inconsistent, it is made to preserve compatibility.
    return UDecompressor(filename, mode or 'rb') if filename.endswith('.uc') else open(filename, mode or 'r')
