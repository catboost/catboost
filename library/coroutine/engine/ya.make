LIBRARY()



PEERDIR(
    library/containers/intrusive_rb_tree
)

SRCS(
    impl.cpp
    iostatus.cpp
    poller.cpp
    sockpool.cpp
    stack.cpp
)

END()
