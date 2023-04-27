# Copyright (c) 2011-2012 Denis Bilenko. See LICENSE for details.
# Automatic pickling of cdef classes was added in 0.26. Unfortunately it
# seems to be buggy (at least for the `result` class) and produces code that
# can't compile ("local variable 'result' referenced before assignment").
# See https://github.com/cython/cython/issues/1786
# cython: auto_pickle=False,language_level=3str
cimport libcares as cares
import sys

from cpython.version cimport PY_MAJOR_VERSION
from cpython.tuple cimport PyTuple_Check
from cpython.getargs cimport PyArg_ParseTuple
from cpython.ref cimport Py_INCREF
from cpython.ref cimport Py_DECREF
from cpython.mem cimport PyMem_Malloc
from cpython.mem cimport PyMem_Free
from libc.string cimport memset

from gevent._compat import MAC

import _socket
from _socket import gaierror
from _socket import herror


__all__ = ['channel']

cdef tuple string_types
cdef type text_type

if PY_MAJOR_VERSION >= 3:
    string_types = str,
    text_type = str
else:
    string_types = __builtins__.basestring,
    text_type = __builtins__.unicode

# These three constants used to be DEF, but the DEF construct
# is deprecated in Cython. Using a cdef extern, the generated
# C code refers to the symbol (DEF would have inlined the value).
# That's great when we're strictly in a C context, but for passing to
# Python, it means we do a runtime translation from the C int to the
# Python int. That is avoided if we use a cdef constant. TIMEOUT
# is the only one that interacts with Python, but not in a performance-sensitive
# way, so runtime translation is fine to keep it consistent.
cdef extern from *:
    """
    #define TIMEOUT 1
    #define EV_READ 1
    #define EV_WRITE 2
    """
    int TIMEOUT
    int EV_READ
    int EV_WRITE


cdef extern from *:
    """
#ifdef CARES_EMBED
#include "ares_setup.h"
#endif

#ifdef HAVE_NETDB_H
#include <netdb.h>
#endif

    #ifndef EAI_ADDRFAMILY
    #define EAI_ADDRFAMILY -1
    #endif

    #ifndef EAI_BADHINTS
    #define EAI_BADHINTS -2
    #endif

    #ifndef EAI_NODATA
    #define EAI_NODATA -3
    #endif

    #ifndef EAI_OVERFLOW
    #define EAI_OVERFLOW -4
    #endif

    #ifndef EAI_PROTOCOL
    #define EAI_PROTOCOL -5
    #endif

    #ifndef EAI_SYSTEM
    #define EAI_SYSTEM
    #endif

    """

cdef extern from "ares.h":
    int AF_INET
    int AF_INET6
    int INET6_ADDRSTRLEN

    struct hostent:
        char* h_name
        int h_addrtype
        char** h_aliases
        char** h_addr_list

    struct sockaddr_t "sockaddr":
        pass

    struct ares_channeldata:
        pass

    struct in_addr:
        unsigned int s_addr

    struct sockaddr_in:
        int sin_family
        int sin_port
        in_addr sin_addr

    struct in6_addr:
        char s6_addr[16]

    struct sockaddr_in6:
        int sin6_family
        int sin6_port
        unsigned int sin6_flowinfo
        in6_addr sin6_addr
        unsigned int sin6_scope_id


    unsigned int htons(unsigned int hostshort)
    unsigned int ntohs(unsigned int hostshort)
    unsigned int htonl(unsigned int hostlong)
    unsigned int ntohl(unsigned int hostlong)

cdef int AI_NUMERICSERV = _socket.AI_NUMERICSERV
cdef int AI_CANONNAME = _socket.AI_CANONNAME
cdef int NI_NUMERICHOST = _socket.NI_NUMERICHOST
cdef int NI_NUMERICSERV = _socket.NI_NUMERICSERV
cdef int NI_NOFQDN = _socket.NI_NOFQDN
cdef int NI_NAMEREQD = _socket.NI_NAMEREQD
cdef int NI_DGRAM = _socket.NI_DGRAM


cdef dict _ares_errors = dict([
    (cares.ARES_SUCCESS, 'ARES_SUCCESS'),

    (cares.ARES_EADDRGETNETWORKPARAMS, 'ARES_EADDRGETNETWORKPARAMS'),
    (cares.ARES_EBADFAMILY, 'ARES_EBADFAMILY'),
    (cares.ARES_EBADFLAGS, 'ARES_EBADFLAGS'),
    (cares.ARES_EBADHINTS, 'ARES_EBADHINTS'),
    (cares.ARES_EBADNAME, 'ARES_EBADNAME'),
    (cares.ARES_EBADQUERY, 'ARES_EBADQUERY'),
    (cares.ARES_EBADRESP, 'ARES_EBADRESP'),
    (cares.ARES_EBADSTR, 'ARES_EBADSTR'),
    (cares.ARES_ECANCELLED, 'ARES_ECANCELLED'),
    (cares.ARES_ECONNREFUSED, 'ARES_ECONNREFUSED'),
    (cares.ARES_EDESTRUCTION, 'ARES_EDESTRUCTION'),
    (cares.ARES_EFILE, 'ARES_EFILE'),
    (cares.ARES_EFORMERR, 'ARES_EFORMERR'),
    (cares.ARES_ELOADIPHLPAPI, 'ARES_ELOADIPHLPAPI'),
    (cares.ARES_ENODATA, 'ARES_ENODATA'),
    (cares.ARES_ENOMEM, 'ARES_ENOMEM'),
    (cares.ARES_ENONAME, 'ARES_ENONAME'),
    (cares.ARES_ENOTFOUND, 'ARES_ENOTFOUND'),
    (cares.ARES_ENOTIMP, 'ARES_ENOTIMP'),
    (cares.ARES_ENOTINITIALIZED, 'ARES_ENOTINITIALIZED'),
    (cares.ARES_EOF, 'ARES_EOF'),
    (cares.ARES_EREFUSED, 'ARES_EREFUSED'),
    (cares.ARES_ESERVICE, 'ARES_ESERVICE'),
    (cares.ARES_ESERVFAIL, 'ARES_ESERVFAIL'),
    (cares.ARES_ETIMEOUT, 'ARES_ETIMEOUT'),
])

cdef dict _ares_to_gai_system = {
    cares.ARES_EBADFAMILY: cares.EAI_ADDRFAMILY,
    cares.ARES_EBADFLAGS:  cares.EAI_BADFLAGS,
    cares.ARES_EBADHINTS:  cares.EAI_BADHINTS,
    cares.ARES_ENOMEM:     cares.EAI_MEMORY,
    cares.ARES_ENONAME:    cares.EAI_NONAME,
    cares.ARES_ENOTFOUND:  cares.EAI_NONAME,
    cares.ARES_ENOTIMP:    cares.EAI_FAMILY,
    # While EAI_NODATA ("No address associated with nodename") might
    # seem to be the natural mapping, typical resolvers actually
    # return EAI_NONAME in that same situation; I've yet to find EAI_NODATA
    # in a test.
    cares.ARES_ENODATA:    cares.EAI_NONAME,
    # This one gets raised for unknown port/service names.
    cares.ARES_ESERVICE:   cares.EAI_NONAME if MAC else cares.EAI_SERVICE,
}

cdef _gevent_gai_strerror(code):
    cdef const char* err_str
    cdef object result = None
    cdef int system
    try:
        system = _ares_to_gai_system[code]
    except KeyError:
        err_str = cares.ares_strerror(code)
        result = '%s: %s' % (_ares_errors.get(code) or code, _as_str(err_str))
    else:
        err_str = cares.gai_strerror(system)
        result = _as_str(err_str)
    return result

cdef object _gevent_gaierror_from_status(int ares_status):
    cdef object code = _ares_to_gai_system.get(ares_status, ares_status)
    cdef object message = _gevent_gai_strerror(ares_status)
    return gaierror(code, message)

cdef dict _ares_to_host_system = {
    cares.ARES_ENONAME:    cares.HOST_NOT_FOUND,
    cares.ARES_ENOTFOUND:  cares.HOST_NOT_FOUND,
    cares.ARES_ENODATA:    cares.NO_DATA,
}

cdef _gevent_herror_strerror(code):
    cdef const char* err_str
    cdef object result = None
    cdef int system
    try:
        system = _ares_to_host_system[code]
    except KeyError:
        err_str = cares.ares_strerror(code)
        result = '%s: %s' % (_ares_errors.get(code) or code, _as_str(err_str))
    else:
        err_str = cares.hstrerror(system)
        result = _as_str(err_str)
    return result


cdef object _gevent_herror_from_status(int ares_status):
    cdef object code = _ares_to_host_system.get(ares_status, ares_status)
    cdef object message = _gevent_herror_strerror(ares_status)
    return herror(code, message)


class InvalidIP(ValueError):
    pass


cdef void gevent_sock_state_callback(void *data, int s, int read, int write):
    if not data:
        return
    cdef channel ch = <channel>data
    ch._sock_state_callback(s, read, write)


cdef class Result(object):
    cdef public object value
    cdef public object exception

    def __init__(self, object value=None, object exception=None):
        self.value = value
        self.exception = exception

    def __repr__(self):
        if self.exception is None:
            return '%s(%r)' % (self.__class__.__name__, self.value)
        elif self.value is None:
            return '%s(exception=%r)' % (self.__class__.__name__, self.exception)
        else:
            return '%s(value=%r, exception=%r)' % (self.__class__.__name__, self.value, self.exception)
        # add repr_recursive precaution

    def successful(self):
        return self.exception is None

    def get(self):
        if self.exception is not None:
            raise self.exception
        return self.value


class ares_host_result(tuple):

    def __new__(cls, family, iterable):
        cdef object self = tuple.__new__(cls, iterable)
        self.family = family
        return self

    def __getnewargs__(self):
        return (self.family, tuple(self))


cdef list _parse_h_aliases(hostent* host):
    cdef list result = []
    cdef char** aliases = host.h_aliases

    if not aliases or not aliases[0]:
        return result

    while aliases[0]: # *aliases
        # The old C version of this excluded an alias if
        # it matched the host name. I don't think the stdlib does that?
        result.append(_as_str(aliases[0]))
        aliases += 1
    return result


cdef list _parse_h_addr_list(hostent* host):
    cdef list result = []
    cdef char** addr_list = host.h_addr_list
    cdef int addr_type = host.h_addrtype
    # INET6_ADDRSTRLEN is 46, but we can't use that named constant
    # here; cython doesn't like it.
    cdef char tmpbuf[46]

    if not addr_list or not addr_list[0]:
        return result

    while addr_list[0]:
        if not cares.ares_inet_ntop(host.h_addrtype, addr_list[0], tmpbuf, INET6_ADDRSTRLEN):
            raise _socket.error("Failed in ares_inet_ntop")

        result.append(_as_str(tmpbuf))
        addr_list += 1

    return result


cdef object _as_str(const char* val):
    if not val:
        return None

    if PY_MAJOR_VERSION < 3:
        return <bytes>val
    return val.decode('utf-8')


cdef void gevent_ares_nameinfo_callback(void *arg, int status, int timeouts, char *c_node, char *c_service):
    cdef channel channel
    cdef object callback
    channel, callback = <tuple>arg
    Py_DECREF(<tuple>arg)
    cdef object node
    cdef object service
    try:
        if status:
            callback(Result(None, _gevent_gaierror_from_status(status)))
        else:
            node = _as_str(c_node)
            service = _as_str(c_service)
            callback(Result((node, service)))
    except:
        channel.loop.handle_error(callback, *sys.exc_info())



cdef int _make_sockaddr(const char* hostp, int port, int flowinfo, int scope_id, sockaddr_in6* sa6):
    if cares.ares_inet_pton(AF_INET, hostp, &(<sockaddr_in*>sa6).sin_addr.s_addr) > 0:
        (<sockaddr_in*>sa6).sin_family = AF_INET
        (<sockaddr_in*>sa6).sin_port = htons(port)
        return sizeof(sockaddr_in)

    if cares.ares_inet_pton(AF_INET6, hostp, &(sa6.sin6_addr).s6_addr) > 0:
        sa6.sin6_family = AF_INET6
        sa6.sin6_port = htons(port)
        sa6.sin6_flowinfo = flowinfo
        sa6.sin6_scope_id = scope_id
        return sizeof(sockaddr_in6);

    return -1;


cdef class channel:
    cdef ares_channeldata* channel

    cdef readonly object loop

    cdef dict _watchers
    cdef object _timer

    def __init__(self, object loop, flags=None, timeout=None, tries=None, ndots=None,
                 udp_port=None, tcp_port=None, servers=None):
        cdef ares_channeldata* channel = NULL
        cdef cares.ares_options options
        memset(&options, 0, sizeof(cares.ares_options))
        cdef int optmask = cares.ARES_OPT_SOCK_STATE_CB

        options.sock_state_cb = <void*>gevent_sock_state_callback
        options.sock_state_cb_data = <void*>self

        if flags is not None:
            options.flags = int(flags)
            optmask |= cares.ARES_OPT_FLAGS

        if timeout is not None:
            options.timeout = int(float(timeout) * 1000)
            optmask |= cares.ARES_OPT_TIMEOUTMS

        if tries is not None:
            options.tries = int(tries)
            optmask |= cares.ARES_OPT_TRIES

        if ndots is not None:
            options.ndots = int(ndots)
            optmask |= cares.ARES_OPT_NDOTS

        if udp_port is not None:
            options.udp_port = int(udp_port)
            optmask |= cares.ARES_OPT_UDP_PORT

        if tcp_port is not None:
            options.tcp_port = int(tcp_port)
            optmask |= cares.ARES_OPT_TCP_PORT

        cdef int result = cares.ares_library_init(cares.ARES_LIB_INIT_ALL)  # ARES_LIB_INIT_WIN32 -DUSE_WINSOCK?
        if result:
            raise gaierror(result, _gevent_gai_strerror(result))
        result = cares.ares_init_options(&channel, &options, optmask)
        if result:
            raise gaierror(result, _gevent_gai_strerror(result))
        self._timer = loop.timer(TIMEOUT, TIMEOUT)
        self._watchers = {}
        self.channel = channel
        try:
            if servers is not None:
                self.set_servers(servers)
            self.loop = loop
        except:
            self.destroy()
            raise

    def __repr__(self):
        args = (self.__class__.__name__, id(self), self._timer, len(self._watchers))
        return '<%s at 0x%x _timer=%r _watchers[%s]>' % args

    def destroy(self):
        self.__destroy()

    cdef __destroy(self):
        if self.channel:
            # XXX ares_library_cleanup?
            cares.ares_destroy(self.channel)
            self.channel = NULL
            self._watchers.clear()
            self._timer.stop()
            self.loop = None

    def __dealloc__(self):
        self.__destroy()

    cpdef set_servers(self, servers=None):
        if not self.channel:
            raise gaierror(cares.ARES_EDESTRUCTION, 'this ares channel has been destroyed')
        if not servers:
            servers = []
        if isinstance(servers, string_types):
            servers = servers.split(',')
        cdef int length = len(servers)
        cdef int result, index
        cdef char* string
        cdef cares.ares_addr_node* c_servers
        if length <= 0:
            result = cares.ares_set_servers(self.channel, NULL)
        else:
            c_servers = <cares.ares_addr_node*>PyMem_Malloc(sizeof(cares.ares_addr_node) * length)
            if not c_servers:
                raise MemoryError
            try:
                index = 0
                for server in servers:
                    if isinstance(server, unicode):
                        server = server.encode('ascii')
                    string = <char*?>server
                    if cares.ares_inet_pton(AF_INET, string, &c_servers[index].addr) > 0:
                        c_servers[index].family = AF_INET
                    elif cares.ares_inet_pton(AF_INET6, string, &c_servers[index].addr) > 0:
                        c_servers[index].family = AF_INET6
                    else:
                        raise InvalidIP(repr(string))
                    c_servers[index].next = &c_servers[index] + 1
                    index += 1
                    if index >= length:
                        break
                c_servers[length - 1].next = NULL
                index = cares.ares_set_servers(self.channel, c_servers)
                if index:
                    raise ValueError(_gevent_gai_strerror(index))
            finally:
                PyMem_Free(c_servers)

    # this crashes c-ares
    #def cancel(self):
    #    cares.ares_cancel(self.channel)

    cdef _sock_state_callback(self, int socket, int read, int write):
        if not self.channel:
            return
        cdef object watcher = self._watchers.get(socket)
        cdef int events = 0
        if read:
            events |= EV_READ
        if write:
            events |= EV_WRITE
        if watcher is None:
            if not events:
                return
            watcher = self.loop.io(socket, events)
            self._watchers[socket] = watcher
        elif events:
            if watcher.events == events:
                return
            watcher.stop()
            watcher.events = events
        else:
            watcher.stop()
            watcher.close()
            self._watchers.pop(socket, None)
            if not self._watchers:
                self._timer.stop()
            return
        watcher.start(self._process_fd, watcher, pass_events=True)
        self._timer.again(self._on_timer)

    def _on_timer(self):
        cares.ares_process_fd(self.channel, cares.ARES_SOCKET_BAD, cares.ARES_SOCKET_BAD)

    def _process_fd(self, int events, object watcher):
        if not self.channel:
            return
        cdef int read_fd = watcher.fd
        cdef int write_fd = read_fd
        if not (events & EV_READ):
            read_fd = cares.ARES_SOCKET_BAD
        if not (events & EV_WRITE):
            write_fd = cares.ARES_SOCKET_BAD
        cares.ares_process_fd(self.channel, read_fd, write_fd)

    @staticmethod
    cdef void _gethostbyname_or_byaddr_cb(void *arg, int status, int timeouts, hostent* host):
        cdef channel channel
        cdef object callback
        channel, callback = <tuple>arg
        Py_DECREF(<tuple>arg)
        cdef object host_result
        try:
            if status or not host:
                callback(Result(None, _gevent_herror_from_status(status)))
            else:
                try:
                    host_result = ares_host_result(host.h_addrtype,
                                                   (_as_str(host.h_name),
                                                    _parse_h_aliases(host),
                                                    _parse_h_addr_list(host)))
                except:
                    callback(Result(None, sys.exc_info()[1]))
                else:
                    callback(Result(host_result))
        except:
            channel.loop.handle_error(callback, *sys.exc_info())


    def gethostbyname(self, object callback, char* name, int family=AF_INET):
        if not self.channel:
            raise gaierror(cares.ARES_EDESTRUCTION, 'this ares channel has been destroyed')
        # note that for file lookups still AF_INET can be returned for AF_INET6 request
        cdef object arg = (self, callback)
        Py_INCREF(arg)
        cares.ares_gethostbyname(self.channel, name, family,
                                 <void*>channel._gethostbyname_or_byaddr_cb, <void*>arg)

    def gethostbyaddr(self, object callback, char* addr):
        if not self.channel:
            raise gaierror(cares.ARES_EDESTRUCTION, 'this ares channel has been destroyed')
        # will guess the family
        cdef char addr_packed[16]
        cdef int family
        cdef int length
        if cares.ares_inet_pton(AF_INET, addr, addr_packed) > 0:
            family = AF_INET
            length = 4
        elif cares.ares_inet_pton(AF_INET6, addr, addr_packed) > 0:
            family = AF_INET6
            length = 16
        else:
            raise InvalidIP(repr(addr))
        cdef object arg = (self, callback)
        Py_INCREF(arg)
        cares.ares_gethostbyaddr(self.channel, addr_packed, length, family,
                                 <void*>channel._gethostbyname_or_byaddr_cb, <void*>arg)

    cpdef _getnameinfo(self, object callback, tuple sockaddr, int flags):
        if not self.channel:
            raise gaierror(cares.ARES_EDESTRUCTION, 'this ares channel has been destroyed')
        cdef char* hostp = NULL
        cdef int port = 0
        cdef int flowinfo = 0
        cdef int scope_id = 0
        cdef sockaddr_in6 sa6
        if not PyTuple_Check(sockaddr):
            raise TypeError('expected a tuple, got %r' % (sockaddr, ))
        PyArg_ParseTuple(sockaddr, "si|ii", &hostp, &port, &flowinfo, &scope_id)
        # if port < 0 or port > 65535:
        #     raise gaierror(-8, 'Invalid value for port: %r' % port)
        cdef int length = _make_sockaddr(hostp, port, flowinfo, scope_id, &sa6)
        if length <= 0:
            raise InvalidIP(repr(hostp))
        cdef object arg = (self, callback)
        Py_INCREF(arg)
        cdef sockaddr_t* x = <sockaddr_t*>&sa6
        cares.ares_getnameinfo(self.channel, x, length, flags, <void*>gevent_ares_nameinfo_callback, <void*>arg)

    @staticmethod
    cdef int _convert_cares_ni_flags(int flags):
        cdef int cares_flags = cares.ARES_NI_LOOKUPHOST | cares.ARES_NI_LOOKUPSERVICE
        if flags & NI_NUMERICHOST:
            cares_flags |= cares.ARES_NI_NUMERICHOST
        if flags & NI_NUMERICSERV:
            cares_flags |= cares.ARES_NI_NUMERICSERV
        if flags & NI_NOFQDN:
            cares_flags |= cares.ARES_NI_NOFQDN
        if flags & NI_NAMEREQD:
            cares_flags |= cares.ARES_NI_NAMEREQD
        if flags & NI_DGRAM:
            cares_flags |= cares.ARES_NI_DGRAM
        return cares_flags

    def getnameinfo(self, object callback, tuple sockaddr, int flags):
        flags = channel._convert_cares_ni_flags(flags)
        return self._getnameinfo(callback, sockaddr, flags)

    @staticmethod
    cdef int _convert_cares_ai_flags(int flags):
        # c-ares supports a limited set of flags.
        # We always want NOSORT, because that implies that
        # c-ares will not connect to resolved addresses.
        cdef int cares_flags = cares.ARES_AI_NOSORT
        if flags & AI_CANONNAME:
            cares_flags |= cares.ARES_AI_CANONNAME
        if flags & AI_NUMERICSERV:
            cares_flags |= cares.ARES_AI_NUMERICSERV
        return cares_flags

    @staticmethod
    cdef void _getaddrinfo_cb(void *arg,
                              int status,
                              int timeouts,
                              cares.ares_addrinfo* result):
        cdef cares.ares_addrinfo_node* nodes
        cdef cares.ares_addrinfo_cname* cnames
        cdef sockaddr_in* sadr4
        cdef sockaddr_in6* sadr6
        cdef object canonname = ''

        cdef channel channel
        cdef object callback
        # INET6_ADDRSTRLEN is 46, but we can't use that named constant
        # here; cython doesn't like it.
        cdef char tmpbuf[46]

        channel, callback = <tuple>arg
        Py_DECREF(<tuple>arg)


        # Result is a list of:
        # (family, socktype, proto, canonname, sockaddr)
        # Where sockaddr depends on family; for INET it is
        # (address, port)
        # and INET6 is
        # (address, port, flow info, scope id)
        # TODO: Check the canonnames.
        addrs = []
        try:
            if status != cares.ARES_SUCCESS:
                callback(Result(None, _gevent_gaierror_from_status(status)))
                return
            if result.cnames:
                # These tend to come in pairs:
                #
                # alias: www.gevent.org name: python-gevent.readthedocs.org
                # alias: python-gevent.readthedocs.org name: readthedocs.io
                #
                # The standard library returns the last name so we do too.

                cnames = result.cnames
                while cnames:
                    canonname = _as_str(cnames.name)
                    cnames = cnames.next

            nodes = result.nodes
            while nodes:
                if nodes.ai_family == AF_INET:
                    sadr4 = <sockaddr_in*>nodes.ai_addr
                    cares.ares_inet_ntop(nodes.ai_family, &sadr4.sin_addr, tmpbuf,
                                            INET6_ADDRSTRLEN)
                    sockaddr = (
                        _as_str(tmpbuf),
                        ntohs(sadr4.sin_port),
                    )
                elif nodes.ai_family == AF_INET6:
                    sadr6 = <sockaddr_in6*>nodes.ai_addr
                    cares.ares_inet_ntop(nodes.ai_family, &sadr6.sin6_addr, tmpbuf,
                                            INET6_ADDRSTRLEN)

                    sockaddr = (
                        _as_str(tmpbuf),
                        ntohs(sadr6.sin6_port),
                        sadr6.sin6_flowinfo,
                        sadr6.sin6_scope_id,
                    )
                addrs.append((
                    nodes.ai_family,
                    nodes.ai_socktype,
                    nodes.ai_protocol,
                    canonname,
                    sockaddr,
                ))
                nodes = nodes.ai_next

            callback(Result(addrs, None))
        except:
            channel.loop.handle_error(callback, *sys.exc_info())
        finally:
            if result:
                cares.ares_freeaddrinfo(result)

    def getaddrinfo(self,
                    object callback,
                    const char* name,
                    object service, # AKA port
                    int family=0,
                    int type=0,
                    int proto=0,
                    int flags=0):
        if not self.channel:
            raise gaierror(cares.ARES_EDESTRUCTION, 'this ares channel has been destroyed')

        cdef cares.ares_addrinfo_hints hints
        memset(&hints, 0, sizeof(cares.ares_addrinfo_hints))

        hints.ai_flags = channel._convert_cares_ai_flags(flags)
        hints.ai_family = family
        hints.ai_socktype = type
        hints.ai_protocol = proto
        cdef object arg = (self, callback)
        Py_INCREF(arg)

        cares.ares_getaddrinfo(
            self.channel,
            name,
            NULL if service is None else <char*>service,
            &hints,
            <void*>channel._getaddrinfo_cb,
            <void*>arg
        )
