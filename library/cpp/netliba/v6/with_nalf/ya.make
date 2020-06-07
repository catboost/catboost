LIBRARY()



SRCDIR(library/cpp/netliba/v6)

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

IF (OS_LINUX)
    PEERDIR(
        contrib/libs/ibdrv
    )
ENDIF()

PEERDIR(
    library/cpp/threading/mux_event
    library/cpp/binsaver
    library/cpp/netliba/socket/with_nalf
)

END()
