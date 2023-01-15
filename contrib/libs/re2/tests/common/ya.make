LIBRARY()


NO_COMPILER_WARNINGS()

PEERDIR (
    contrib/libs/re2
)

SRCDIR(
    contrib/libs/re2
)

SRCS(
    util/pcre.cc
    util/test.cc

    re2/testing/backtrack.cc
    re2/testing/dump.cc
    re2/testing/exhaustive_tester.cc
    re2/testing/null_walker.cc
    re2/testing/regexp_generator.cc
    re2/testing/string_generator.cc
    re2/testing/tester.cc
)

END()
