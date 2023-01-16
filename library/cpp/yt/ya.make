RECURSE(
    assert
    misc
    string
    yson_string
)

IF (NOT OS_WINDOWS)
    RECURSE(
    malloc
    memory
    small_containers
)
ENDIF()
