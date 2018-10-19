LIBRARY()



SRCS(
    profiler.cpp
    stackcollect.cpp
)

PEERDIR(
    library/lfalloc/dbg_info
)

END()

RECURSE(
    ut
)
