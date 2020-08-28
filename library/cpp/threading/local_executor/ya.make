

LIBRARY()

SRCS(
    local_executor.cpp
    omp_local_executor.cpp
)

PEERDIR(
    contrib/libs/cxxsupp/openmp
)

END()
