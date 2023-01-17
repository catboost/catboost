LIBRARY()



SRCS(
    profiler.cpp
    stackcollect.cpp
)

PEERDIR(
    library/cpp/lfalloc/dbg_info
    library/cpp/cache
    library/cpp/deprecated/atomic
)

END()

RECURSE(
    ut
)
