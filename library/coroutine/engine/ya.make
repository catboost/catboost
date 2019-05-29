LIBRARY()



GENERATE_ENUM_SERIALIZATION(poller.h)

PEERDIR(
    library/containers/intrusive_rb_tree
)

SRCS(
    cont_poller.cpp
    impl.cpp
    iostatus.cpp
    poller.cpp
    sockpool.cpp
    stack.cpp
)

END()
