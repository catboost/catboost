UNITTEST_FOR(util)


SUBSCRIBER(g:util-subscribers)

SRCS(
    thread/factory_ut.cpp
    thread/lfqueue_ut.cpp
    thread/lfstack_ut.cpp
    thread/pool_ut.cpp
    thread/singleton_ut.cpp
)

PEERDIR(
    library/cpp/threading/future
)

END()
