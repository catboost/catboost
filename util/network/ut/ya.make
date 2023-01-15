UNITTEST_FOR(util)

REQUIREMENTS(network:full)



PEERDIR(
    library/cpp/threading/future
)

SRCS(
    network/address_ut.cpp
    network/endpoint_ut.cpp
    network/ip_ut.cpp
    network/poller_ut.cpp
    network/sock_ut.cpp
    network/socket_ut.cpp
)

INCLUDE(${ARCADIA_ROOT}/util/tests/ya_util_tests.inc)

END()
