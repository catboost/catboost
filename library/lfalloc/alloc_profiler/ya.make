LIBRARY()



SRCS(
    profiler.cpp
    stackcollect.cpp
)

PEERDIR(
    library/lfalloc/dbg_info
    library/cache
)

END()

RECURSE(
    ut
)
