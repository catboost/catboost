RECURSE(
    api
    api/helpers
    api/ut
    jemalloc
    nalf
    system
)

IF (OS_LINUX OR OS_WINDOWS)
    RECURSE(
    
)
ENDIF ()
