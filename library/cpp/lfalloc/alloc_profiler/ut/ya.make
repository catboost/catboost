UNITTEST_FOR(library/cpp/lfalloc/alloc_profiler)



PEERDIR(
    library/cpp/unittest
)

IF (ARCH_AARCH64)
    PEERDIR(
        contrib/libs/jemalloc
    )
ELSE()
    ALLOCATOR(LF_DBG)
ENDIF()

SRCS(
    profiler_ut.cpp
)

END()
