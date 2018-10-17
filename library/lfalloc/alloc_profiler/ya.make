LIBRARY()



SRCS(
    profiler.cpp
    stackcollect.cpp
)

IF (PROFILE_MEMORY_ALLOCATIONS)
    CFLAGS(
        GLOBAL -DPROFILE_MEMORY_ALLOCATIONS
    )
    PEERDIR(
        library/lfalloc/dbg_info
    )
ENDIF()

END()
