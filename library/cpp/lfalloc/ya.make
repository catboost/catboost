LIBRARY()



NO_UTIL()

NO_COMPILER_WARNINGS()

IF (ARCH_AARCH64)
    PEERDIR(
        contrib/libs/jemalloc
    )
ELSE()
    SRCS(
        lf_allocX64.cpp
    )
ENDIF()

PEERDIR(
    library/cpp/malloc/api
)

SET(IDE_FOLDER "util")

END()
