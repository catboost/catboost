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
        '''
        Gets free TCP port
        '''
        return self.get_tcp_port(port)

    def get_tcp_port(self, port=0):
        '''
        Gets free TCP port
        '''
        return self._get_port(port, socket.SOCK_STREAM)

    def get_udp_port(self, port=0):
        '''
        Gets free UDP port
        '''
        return self._get_port(port, socket.SOCK_DGRAM)

    def get_tcp_and_udp_port(self, port=0):
        '''
        Gets one free port for use in both TCP and UDP protocols
        '''
        if port and self._no_random_ports():
            return port

        retries = 20
        while retries > 0:
            retries -= 1

            # 1. Get random free TCP port. Ports are guaranteed to be different with
            #    ports given by get_tcp_port() and other get_tcp_and_udp_port() methods.
            # 2. Bind the same UDP port without SO_REUSEADDR to avoid race with get_udp_port() method:
            #    if get_udp_port() from other thread/process gets this port, bind() fails; if bind()
            #    succeeds, then get_udp_port() from other thread/process gives other port.
            # 3. Set SO_REUSEADDR option to let use this UDP port from test.
            result_port = self.get_tcp_port()
            sock = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
            try:
                sock.bind(('::', result_port))
            except:
                sock.close()
                self.release_port(result_port)
                continue
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._save_socket(result_port, sock)
            return result_port
        raise Exception('Failed to find port')

    def release_port(self, port):
        sockets = self._sockets.get(port, [])
        for sock in sockets:
            sock.close()

    def release(self):
        while self._sockets:
            _, sockets = self._sockets.popitem()
            for sock in sockets:
                sock.close()

    def _get_port(self, port, sock_type):
        if port and self._no_random_ports():
            return port

        sock = socket.socket(socket.AF_INET6, sock_type)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('::', 0))
        port = sock.getsockname()[1]
        self._save_socket(port, sock)
        return port

    def _save_socket(self, port, sock):
        sockets = self._sockets.get(port, [])
        sockets.append(sock)
        self._sockets[port] = sockets

    def _no_random_ports(self):
        return os.environ.get("NO_RANDOM_PORTS")
