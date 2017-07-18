LIBRARY()


NO_UTIL()
NO_COMPILER_WARNINGS()

IF (ARCH_AARCH64)
    PEERDIR(
        contrib/libs/jemalloc
    )
ELSE()
    IF ("${YMAKE}" MATCHES "devtools")
        CFLAGS(-DYMAKE=1)
    ENDIF()

    CXXFLAGS(-DLFALLOC_DBG -DLFALLOC_YT)
    SRCS(
        ../lf_allocX64.cpp
    )
ENDIF()

PEERDIR(
    library/malloc/api
)

SET(IDE_FOLDER "util")

END()
