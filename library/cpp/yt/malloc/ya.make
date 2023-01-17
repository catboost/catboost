LIBRARY()

IF(NOT OS_WINDOWS)
    SRCS(
        malloc.cpp
    )
ENDIF()

END()
