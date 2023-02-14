from six.moves import SimpleHTTPServer
from six.moves import socketserver as SocketServer
import logging
import threading

logger = logging.getLogger(__name__)


class SilentHandler(SimpleHTTPServer.SimpleHTTPRequestHandler):
    def log_message(self, _, *__):
        return


class SilentHTTPServer(object):
    def __init__(self, host='localhost', port=0):
        self.host = host
        self.port = port

    def __enter__(self):
        self._httpd = SocketServer.TCPServer((self.host, 0), SilentHandler)
        self.port = self._httpd.socket.getsockname()[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever)
        self._thread.start()
        logger.info('Server started on %s:%s', self.host, self.port)
        return self

    def __exit__(self, type, value, traceback):
        self._httpd.shutdown()
        self._thread.join()
        logger.info('Server stopped')
