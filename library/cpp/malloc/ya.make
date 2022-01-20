RECURSE(
    api
    api/helpers
    api/ut
    tcmalloc
    jemalloc
    nalf
    system
    mimalloc
    mimalloc/link_test
)

IF (NOT OS_WINDOWS)
    RECURSE(
    
)
ENDIF()
