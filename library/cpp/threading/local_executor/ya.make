

LIBRARY()
PEERDIR(
    contrib/libs/tbb
)

ADDINCL(
    contrib/libs/tbb/include
)

SRCS(
    local_executor.cpp
    tbb_local_executor.cpp
)

PEERDIR(
    contrib/libs/tbb
)

END()
