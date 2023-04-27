"""0MQ Constant names"""

# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.

# dictionaries of constants new or removed in particular versions

new_in = {
    (2,2,0) : [
        'RCVTIMEO',
        'SNDTIMEO',
    ],
    (3,2,2) : [
        # errnos
        'EMSGSIZE',
        'EAFNOSUPPORT',
        'ENETUNREACH',
        'ECONNABORTED',
        'ECONNRESET',
        'ENOTCONN',
        'ETIMEDOUT',
        'EHOSTUNREACH',
        'ENETRESET',

        # ctx opts
        'IO_THREADS',
        'MAX_SOCKETS',
        'IO_THREADS_DFLT',
        'MAX_SOCKETS_DFLT',

        # socket opts
        'IPV4ONLY',
        'LAST_ENDPOINT',
        'ROUTER_BEHAVIOR',
        'ROUTER_MANDATORY',
        'FAIL_UNROUTABLE',
        'TCP_KEEPALIVE',
        'TCP_KEEPALIVE_CNT',
        'TCP_KEEPALIVE_IDLE',
        'TCP_KEEPALIVE_INTVL',
        'DELAY_ATTACH_ON_CONNECT',
        'XPUB_VERBOSE',

        # msg opts
        'MORE',

        'EVENT_CONNECTED',
        'EVENT_CONNECT_DELAYED',
        'EVENT_CONNECT_RETRIED',
        'EVENT_LISTENING',
        'EVENT_BIND_FAILED',
        'EVENT_ACCEPTED',
        'EVENT_ACCEPT_FAILED',
        'EVENT_CLOSED',
        'EVENT_CLOSE_FAILED',
        'EVENT_DISCONNECTED',
        'EVENT_ALL',
    ],
    (4,0,0) : [
        # socket types
        'STREAM',

        # socket opts
        'IMMEDIATE',
        'ROUTER_RAW',
        'IPV6',
        'MECHANISM',
        'PLAIN_SERVER',
        'PLAIN_USERNAME',
        'PLAIN_PASSWORD',
        'CURVE_SERVER',
        'CURVE_PUBLICKEY',
        'CURVE_SECRETKEY',
        'CURVE_SERVERKEY',
        'PROBE_ROUTER',
        'REQ_RELAXED',
        'REQ_CORRELATE',
        'CONFLATE',
        'ZAP_DOMAIN',

        # security
        'NULL',
        'PLAIN',
        'CURVE',

        # events
        'EVENT_MONITOR_STOPPED',
    ],
    (4,1,0) : [
        # ctx opts
        'SOCKET_LIMIT',
        'THREAD_PRIORITY',
        'THREAD_PRIORITY_DFLT',
        'THREAD_SCHED_POLICY',
        'THREAD_SCHED_POLICY_DFLT',

        # socket opts
        'ROUTER_HANDOVER',
        'TOS',
        'IPC_FILTER_PID',
        'IPC_FILTER_UID',
        'IPC_FILTER_GID',
        'CONNECT_RID',
        'GSSAPI_SERVER',
        'GSSAPI_PRINCIPAL',
        'GSSAPI_SERVICE_PRINCIPAL',
        'GSSAPI_PLAINTEXT',
        'HANDSHAKE_IVL',
        'XPUB_NODROP',
        'SOCKS_PROXY',

        # msg opts
        'SRCFD',
        'SHARED',

        # security
        'GSSAPI',
    ],
    (4,2,0): [
        # polling
        'POLLPRI',
    ],
    (4,2,3): [
        'ROUTING_ID',
        'CONNECT_ROUTING_ID',
    ],
    (4,3,0): [
        # context options
        'MSG_T_SIZE',
        'THREAD_AFFINITY_CPU_ADD',
        'THREAD_AFFINITY_CPU_REMOVE',
        'THREAD_NAME_PREFIX',

        # socket options
        'GSSAPI_PRINCIPAL_NAMETYPE',
        'GSSAPI_SERVICE_PRINCIPAL_NAMETYPE',
        'BINDTODEVICE',

        # GSSAPI principal name types
        'GSSAPI_NT_HOSTBASED',
        'GSSAPI_NT_USER_NAME',
        'GSSAPI_NT_KRB5_PRINCIPAL',

        # events
        'EVENT_HANDSHAKE_FAILED_NO_DETAIL',
        'EVENT_HANDSHAKE_SUCCEEDED',
        'EVENT_HANDSHAKE_FAILED_PROTOCOL',
        'EVENT_HANDSHAKE_FAILED_AUTH',

        'PROTOCOL_ERROR_ZMTP_UNSPECIFIED',
        'PROTOCOL_ERROR_ZMTP_UNEXPECTED_COMMAND',
        'PROTOCOL_ERROR_ZMTP_INVALID_SEQUENCE',
        'PROTOCOL_ERROR_ZMTP_KEY_EXCHANGE',
        'PROTOCOL_ERROR_ZMTP_MALFORMED_COMMAND_UNSPECIFIED',
        'PROTOCOL_ERROR_ZMTP_MALFORMED_COMMAND_MESSAGE',
        'PROTOCOL_ERROR_ZMTP_MALFORMED_COMMAND_HELLO',
        'PROTOCOL_ERROR_ZMTP_MALFORMED_COMMAND_INITIATE',
        'PROTOCOL_ERROR_ZMTP_MALFORMED_COMMAND_ERROR',
        'PROTOCOL_ERROR_ZMTP_MALFORMED_COMMAND_READY',
        'PROTOCOL_ERROR_ZMTP_MALFORMED_COMMAND_WELCOME',
        'PROTOCOL_ERROR_ZMTP_INVALID_METADATA',
        'PROTOCOL_ERROR_ZMTP_CRYPTOGRAPHIC',
        'PROTOCOL_ERROR_ZMTP_MECHANISM_MISMATCH',

        'PROTOCOL_ERROR_ZAP_UNSPECIFIED',
        'PROTOCOL_ERROR_ZAP_MALFORMED_REPLY',
        'PROTOCOL_ERROR_ZAP_BAD_REQUEST_ID',
        'PROTOCOL_ERROR_ZAP_BAD_VERSION',
        'PROTOCOL_ERROR_ZAP_INVALID_STATUS_CODE',
        'PROTOCOL_ERROR_ZAP_INVALID_METADATA',
    ]
}

draft_in = {
    (4,2,0): [
        # socket types
        'SERVER',
        'CLIENT',
        'RADIO',
        'DISH',
        'GATHER',
        'SCATTER',
        'DGRAM',

        # ctx options
        'BLOCKY',

        # socket options
        'XPUB_MANUAL',
        'XPUB_WELCOME_MSG',
        'STREAM_NOTIFY',
        'INVERT_MATCHING',
        'HEARTBEAT_IVL',
        'HEARTBEAT_TTL',
        'HEARTBEAT_TIMEOUT',
        'XPUB_VERBOSER',
        'CONNECT_TIMEOUT',
        'TCP_MAXRT',
        'THREAD_SAFE',
        'MULTICAST_MAXTPDU',
        'VMCI_BUFFER_SIZE',
        'VMCI_BUFFER_MIN_SIZE',
        'VMCI_BUFFER_MAX_SIZE',
        'VMCI_CONNECT_TIMEOUT',
        'USE_FD',
    ],
    (4,2,4): [
        # socket options
        'ZAP_ENFORCE_DOMAIN',
        'LOOPBACK_FASTPATH',
        'METADATA',
        'ZERO_COPY_RECV',
    ],
    (4,3,0): [
        # socket options
        'ROUTER_NOTIFY',
        'MULTICAST_LOOP',

        'NOTIFY_CONNECT',
        'NOTIFY_DISCONNECT',
    ],
}


removed_in = {
    (3,2,2) : [
        'UPSTREAM',
        'DOWNSTREAM',

        'HWM',
        'SWAP',
        'MCAST_LOOP',
        'RECOVERY_IVL_MSEC',
    ]
}

# collections of zmq constant names based on their role
# base names have no specific use
# opt names are validated in get/set methods of various objects

base_names = [
    # base
    'VERSION',
    'VERSION_MAJOR',
    'VERSION_MINOR',
    'VERSION_PATCH',
    'NOBLOCK',
    'DONTWAIT',

    'POLLIN',
    'POLLOUT',
    'POLLERR',
    'POLLPRI',

    'SNDMORE',

    'STREAMER',
    'FORWARDER',
    'QUEUE',

    'IO_THREADS_DFLT',
    'MAX_SOCKETS_DFLT',
    'POLLITEMS_DFLT',
    'THREAD_PRIORITY_DFLT',
    'THREAD_SCHED_POLICY_DFLT',

    # socktypes
    'PAIR',
    'PUB',
    'SUB',
    'REQ',
    'REP',
    'DEALER',
    'ROUTER',
    'XREQ',
    'XREP',
    'PULL',
    'PUSH',
    'XPUB',
    'XSUB',
    'UPSTREAM',
    'DOWNSTREAM',
    'STREAM',
    'SERVER',
    'CLIENT',
    'RADIO',
    'DISH',
    'GATHER',
    'SCATTER',
    'DGRAM',

    # events
    'EVENT_CONNECTED',
    'EVENT_CONNECT_DELAYED',
    'EVENT_CONNECT_RETRIED',
    'EVENT_LISTENING',
    'EVENT_BIND_FAILED',
    'EVENT_ACCEPTED',
    'EVENT_ACCEPT_FAILED',
    'EVENT_CLOSED',
    'EVENT_CLOSE_FAILED',
    'EVENT_DISCONNECTED',
    'EVENT_ALL',
    'EVENT_MONITOR_STOPPED',
    'EVENT_HANDSHAKE_FAILED_NO_DETAIL',
    'EVENT_HANDSHAKE_SUCCEEDED',
    'EVENT_HANDSHAKE_FAILED_PROTOCOL',
    'EVENT_HANDSHAKE_FAILED_AUTH',

    'PROTOCOL_ERROR_ZMTP_UNSPECIFIED',
    'PROTOCOL_ERROR_ZMTP_UNEXPECTED_COMMAND',
    'PROTOCOL_ERROR_ZMTP_INVALID_SEQUENCE',
    'PROTOCOL_ERROR_ZMTP_KEY_EXCHANGE',
    'PROTOCOL_ERROR_ZMTP_MALFORMED_COMMAND_UNSPECIFIED',
    'PROTOCOL_ERROR_ZMTP_MALFORMED_COMMAND_MESSAGE',
    'PROTOCOL_ERROR_ZMTP_MALFORMED_COMMAND_HELLO',
    'PROTOCOL_ERROR_ZMTP_MALFORMED_COMMAND_INITIATE',
    'PROTOCOL_ERROR_ZMTP_MALFORMED_COMMAND_ERROR',
    'PROTOCOL_ERROR_ZMTP_MALFORMED_COMMAND_READY',
    'PROTOCOL_ERROR_ZMTP_MALFORMED_COMMAND_WELCOME',
    'PROTOCOL_ERROR_ZMTP_INVALID_METADATA',
    'PROTOCOL_ERROR_ZMTP_CRYPTOGRAPHIC',
    'PROTOCOL_ERROR_ZMTP_MECHANISM_MISMATCH',

    'PROTOCOL_ERROR_ZAP_UNSPECIFIED',
    'PROTOCOL_ERROR_ZAP_MALFORMED_REPLY',
    'PROTOCOL_ERROR_ZAP_BAD_REQUEST_ID',
    'PROTOCOL_ERROR_ZAP_BAD_VERSION',
    'PROTOCOL_ERROR_ZAP_INVALID_STATUS_CODE',
    'PROTOCOL_ERROR_ZAP_INVALID_METADATA',

    'NOTIFY_CONNECT',
    'NOTIFY_DISCONNECT',

    # security
    'NULL',
    'PLAIN',
    'CURVE',
    'GSSAPI',
    'GSSAPI_NT_HOSTBASED',
    'GSSAPI_NT_USER_NAME',
    'GSSAPI_NT_KRB5_PRINCIPAL',

    ## ERRNO
    # Often used (these are else in errno.)
    'EAGAIN',
    'EINVAL',
    'EFAULT',
    'ENOMEM',
    'ENODEV',
    'EMSGSIZE',
    'EAFNOSUPPORT',
    'ENETUNREACH',
    'ECONNABORTED',
    'ECONNRESET',
    'ENOTCONN',
    'ETIMEDOUT',
    'EHOSTUNREACH',
    'ENETRESET',

    # For Windows compatibility
    'HAUSNUMERO',
    'ENOTSUP',
    'EPROTONOSUPPORT',
    'ENOBUFS',
    'ENETDOWN',
    'EADDRINUSE',
    'EADDRNOTAVAIL',
    'ECONNREFUSED',
    'EINPROGRESS',
    'ENOTSOCK',

    # 0MQ Native
    'EFSM',
    'ENOCOMPATPROTO',
    'ETERM',
    'EMTHREAD',
]

int64_sockopt_names = [
    'AFFINITY',
    'MAXMSGSIZE',

    # sockopts removed in 3.0.0
    'HWM',
    'SWAP',
    'MCAST_LOOP',
    'RECOVERY_IVL_MSEC',

    # new in 4.2
    'VMCI_BUFFER_SIZE',
    'VMCI_BUFFER_MIN_SIZE',
    'VMCI_BUFFER_MAX_SIZE',
]

bytes_sockopt_names = [
    'IDENTITY',
    'SUBSCRIBE',
    'UNSUBSCRIBE',
    'LAST_ENDPOINT',
    'TCP_ACCEPT_FILTER',

    'PLAIN_USERNAME',
    'PLAIN_PASSWORD',

    'CURVE_PUBLICKEY',
    'CURVE_SECRETKEY',
    'CURVE_SERVERKEY',
    'ZAP_DOMAIN',
    'CONNECT_RID',
    'GSSAPI_PRINCIPAL',
    'GSSAPI_SERVICE_PRINCIPAL',
    'SOCKS_PROXY',

    'XPUB_WELCOME_MSG',

    # new in 4.2.3
    'ROUTING_ID',
    'CONNECT_ROUTING_ID',

    # new in 4.3.0
    'BINDTODEVICE',
]

fd_sockopt_names = [
    'FD',
]

int_sockopt_names = [
    # sockopts
    'RECONNECT_IVL_MAX',

    # sockopts new in 2.2.0
    'SNDTIMEO',
    'RCVTIMEO',

    # new in 3.x
    'SNDHWM',
    'RCVHWM',
    'MULTICAST_HOPS',
    'IPV4ONLY',

    'ROUTER_BEHAVIOR',
    'TCP_KEEPALIVE',
    'TCP_KEEPALIVE_CNT',
    'TCP_KEEPALIVE_IDLE',
    'TCP_KEEPALIVE_INTVL',
    'DELAY_ATTACH_ON_CONNECT',
    'XPUB_VERBOSE',

    'EVENTS',
    'TYPE',
    'LINGER',
    'RECONNECT_IVL',
    'BACKLOG',

    'ROUTER_MANDATORY',
    'FAIL_UNROUTABLE',

    'ROUTER_RAW',
    'IMMEDIATE',
    'IPV6',
    'MECHANISM',
    'PLAIN_SERVER',
    'CURVE_SERVER',
    'PROBE_ROUTER',
    'REQ_RELAXED',
    'REQ_CORRELATE',
    'CONFLATE',
    'ROUTER_HANDOVER',
    'TOS',
    'IPC_FILTER_PID',
    'IPC_FILTER_UID',
    'IPC_FILTER_GID',
    'GSSAPI_SERVER',
    'GSSAPI_PLAINTEXT',
    'HANDSHAKE_IVL',
    'XPUB_NODROP',

    # new in 4.2
    'XPUB_MANUAL',
    'STREAM_NOTIFY',
    'INVERT_MATCHING',
    'XPUB_VERBOSER',
    'HEARTBEAT_IVL',
    'HEARTBEAT_TTL',
    'HEARTBEAT_TIMEOUT',
    'CONNECT_TIMEOUT',
    'TCP_MAXRT',
    'THREAD_SAFE',
    'MULTICAST_MAXTPDU',
    'VMCI_CONNECT_TIMEOUT',
    'USE_FD',

    # new in 4.3
    'GSSAPI_PRINCIPAL_NAMETYPE',
    'GSSAPI_SERVICE_PRINCIPAL_NAMETYPE',
    'MULTICAST_LOOP',
    'ROUTER_NOTIFY',
    'ZAP_ENFORCE_DOMAIN',
]

switched_sockopt_names = [
    'RATE',
    'RECOVERY_IVL',
    'SNDBUF',
    'RCVBUF',
    'RCVMORE',
]

ctx_opt_names = [
    'IO_THREADS',
    'MAX_SOCKETS',
    'SOCKET_LIMIT',
    'THREAD_PRIORITY',
    'THREAD_SCHED_POLICY',
    'BLOCKY',

    # new in 4.3
    'MSG_T_SIZE',
    'THREAD_AFFINITY_CPU_ADD',
    'THREAD_AFFINITY_CPU_REMOVE',
    'THREAD_NAME_PREFIX',
]

msg_opt_names = [
    'MORE',
    'SRCFD',
    'SHARED',
]

from itertools import chain

all_names = list(chain(
    base_names,
    ctx_opt_names,
    bytes_sockopt_names,
    fd_sockopt_names,
    int_sockopt_names,
    int64_sockopt_names,
    switched_sockopt_names,
    msg_opt_names,
))

del chain

def no_prefix(name):
    """does the given constant have a ZMQ_ prefix?"""
    return name.startswith('E') and not name.startswith('EVENT')

