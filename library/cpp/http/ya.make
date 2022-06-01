RECURSE(
    io
    misc
    push_parser
)

IF (NOT OS_WINDOWS)
    RECURSE_FOR_TESTS(
        io/ut
        io/ut/medium
    )
ENDIF()
