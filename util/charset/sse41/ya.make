LIBRARY()



NO_UTIL()

IF (MSVC)
    CFLAGS(
        /D__SSE41__=1
    )
ELSE()
    CFLAGS(
        -msse4.1
    )
ENDIF()

SRCS(
    wide.cpp
)

END()
