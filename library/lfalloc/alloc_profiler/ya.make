LIBRARY()



SRCS(
    profiler.cpp
    stackcollect.cpp
)

PEERDIR(
    library/lfalloc/dbg_info
    library/cpp/cache
)

END()

RECURSE(
    ut
)
