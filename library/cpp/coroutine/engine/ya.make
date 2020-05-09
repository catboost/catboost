LIBRARY()



GENERATE_ENUM_SERIALIZATION(poller.h)
GENERATE_ENUM_SERIALIZATION(trampoline.h)

PEERDIR(
    library/cpp/containers/intrusive_rb_tree
)

SRCS(
    cont_poller.cpp
    impl.cpp
    iostatus.cpp
    network.cpp
    poller.cpp
    sockpool.cpp
    trampoline.cpp
)

END()
