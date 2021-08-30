UNITTEST_FOR(util/charset)


SUBSCRIBER(g:util-subscribers)

DATA(arcadia/util/charset/ut/utf8)

SRCS(
    utf8_ut.cpp
    wide_ut.cpp
)

INCLUDE(${ARCADIA_ROOT}/util/tests/ya_util_tests.inc)

END()
