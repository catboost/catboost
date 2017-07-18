# coding=utf-8

import os
import socket


class PortManager(object):
    """
    See documentation here

    https://wiki.yandex-team.ru/yatool/test/#poluchenieportovdljatestirovanija
    """

    def __init__(self):
        self._sockets = {}

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.release()

    def get_port(self, port=0):
        if os.environ.get("NO_RANDOM_PORTS") and port:
            return port

        sock = socket.socket(socket.AF_INET6)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('::', 0))
        port = sock.getsockname()[1]
        self._sockets[port] = sock
        return port

    def release(self):
        while self._sockets:
            _, sock = self._sockets.popitem()
            sock.close()
