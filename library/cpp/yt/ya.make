RECURSE(
    assert
    coding
    exception
    misc
    string
    yson
    yson_string
)

IF (NOT OS_WINDOWS)
    RECURSE(
    malloc
    memory
    small_containers
)
ENDIF()
