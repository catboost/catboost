PROGRAM()


NO_COMPILER_WARNINGS()

PEERDIR(
    contrib/libs/re2/tests/common
)

SRCDIR(
    contrib/libs/re2/re2/testing
)

SRCS(
    parse_test.cc
)

END()
