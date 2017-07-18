UNITTEST_FOR(util)



PEERDIR(
    library/threading/future
)

SRCS(
    network/address_ut.cpp
    network/endpoint_ut.cpp
    network/ip_ut.cpp
    network/poller_ut.cpp
    network/sock_ut.cpp
    network/socket_ut.cpp
)

END()
