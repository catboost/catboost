LIBRARY()



SRCDIR(library/netliba/v12)

SRCS(
    block_chain.cpp
    circular_pod_buffer.h
    cpu_affinity.cpp
    ib_buffers.cpp
    ib_collective.cpp
    ib_cs.cpp
    ib_low.cpp
    ib_mem.cpp
    ib_memstream.cpp
    ib_test.cpp
    local_ip_params.cpp
    net_acks.cpp
    net_queue_stat.h
    net_request.cpp
    net_test.cpp
    paged_pod_buffer.h
    posix_shared_memory.h
    settings.h
    socket.h
    stdafx.cpp
    udp_address.cpp
    udp_debug.cpp
    udp_host.cpp
    udp_host_connection.h
    udp_host_protocol.h
    udp_host_recv_completed.h
    udp_http.cpp
    udp_recv_packet.h
    udp_socket.cpp
    udp_test.cpp
)

IF (OS_LINUX)
    PEERDIR(
        contrib/libs/ibdrv
    )
ENDIF()

PEERDIR(
    library/cpp/threading/mux_event
    library/cpp/digest/crc32c
    library/cpp/binsaver
    library/netliba/socket/with_nalf
)

END()
