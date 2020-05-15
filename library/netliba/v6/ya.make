LIBRARY()



SRCS(
    stdafx.cpp
    udp_address.cpp
    udp_client_server.cpp
    udp_http.cpp
    net_acks.cpp
    udp_test.cpp
    block_chain.cpp
    net_test.cpp
    udp_debug.cpp
    udp_socket.cpp
    net_queue_stat.h
    ib_low.cpp
    ib_buffers.cpp
    ib_mem.cpp
    ib_cs.cpp
    ib_test.cpp
    cpu_affinity.cpp
    net_request.cpp
    ib_collective.cpp
    ib_memstream.cpp
)

IF (OS_LINUX AND NOT CATBOOST_OPENSOURCE)
    PEERDIR(
        contrib/libs/ibdrv
    )
ENDIF()

IF (CATBOOST_OPENSOURCE)
    CFLAGS(-DCATBOOST_OPENSOURCE=yes)
ENDIF()

PEERDIR(
    library/cpp/binsaver
    library/netliba/socket
    library/cpp/string_utils/url
    library/cpp/threading/atomic
    library/cpp/threading/mux_event
)

END()
