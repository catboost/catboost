LIBRARY()



SRCS(
    skiplist.cpp
)

PEERDIR(
    library/cpp/deprecated/atomic
)

END()

RECURSE(
    perf
    ut
)
