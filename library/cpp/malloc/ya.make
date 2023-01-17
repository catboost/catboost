RECURSE(
    api
    tcmalloc
    jemalloc
    nalf
    system
    mimalloc
)

IF (NOT OS_WINDOWS)
    RECURSE(
    
)
ENDIF()
