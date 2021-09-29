RECURSE(
    api
    api/helpers
    api/ut
    tcmalloc
    jemalloc
    nalf
    system
)

IF (NOT OS_WINDOWS)
    RECURSE(
    
)
ENDIF()
