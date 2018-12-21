RECURSE(
    io
    io/fuzz
    io/list_codings
    misc
    misc/ut
    push_parser
    push_parser/ut
)

IF (NOT OS_WINDOWS)
    RECURSE(
    io/ut
    io/ut/medium
)
ENDIF()
