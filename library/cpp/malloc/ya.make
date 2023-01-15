RECURSE(
    api
    api/helpers
    api/ut
    jemalloc
    nalf
    system
)

IF (NOT OS_WINDOWS)
    RECURSE(
    
)
ENDIF()
