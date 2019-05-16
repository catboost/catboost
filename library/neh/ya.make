

LIBRARY()

PEERDIR(
    contrib/libs/openssl
    library/containers/intrusive_rb_tree
    library/coroutine/engine
    library/coroutine/listener
    library/dns
    library/http/io
    library/http/misc
    library/http/push_parser
    library/neh/asio
    library/openssl/init
    library/openssl/method
    library/threading/atomic
)

SRCS(
    conn_cache.cpp
    factory.cpp
    https.cpp
    http_common.cpp
    http_headers.cpp
    http2.cpp
    inproc.cpp
    jobqueue.cpp
    location.cpp
    multi.cpp
    multiclient.cpp
    neh.cpp
    pipequeue.cpp
    rpc.cpp
    rq.cpp
    smart_ptr.cpp
    stat.cpp
    tcp.cpp
    tcp2.cpp
    udp.cpp
    utils.cpp
)

IF(NOT OS_ANDROID)
    PEERDIR(
        library/netliba/v6
    )

    SRCS(
        netliba.cpp
        netliba_udp_http.cpp
    )
ENDIF()

GENERATE_ENUM_SERIALIZATION(http_common.h)

END()
