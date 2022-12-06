LIBRARY()

SRCS(
    malloc.cpp
)

IF (NOT OS_WINDOWS OR ALLOCATOR != "LF")
    SRCS(
        dummy_malloc.cpp
    )
ENDIF()

END()
