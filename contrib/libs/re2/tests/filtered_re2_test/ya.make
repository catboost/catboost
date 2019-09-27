PROGRAM()


NO_COMPILER_WARNINGS()

PEERDIR(
    contrib/libs/re2/tests/common
)

SRCDIR(
    contrib/libs/re2/re2/testing
)

SRCS(
    filtered_re2_test.cc
)

END()
