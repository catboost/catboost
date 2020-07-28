

LIBRARY()

PEERDIR(
    library/cpp/coroutine/engine
    library/cpp/dns
)

SRCS(
    asio.cpp
    deadline_timer_impl.cpp
    executor.cpp
    io_service_impl.cpp
    poll_interrupter.cpp
    tcp_acceptor_impl.cpp
    tcp_socket_impl.cpp
)

END()
