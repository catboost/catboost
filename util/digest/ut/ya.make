UNITTEST_FOR(util)


SUBSCRIBER(g:util-subscribers)

SRCS(
    digest/city_ut.cpp
    digest/fnv_ut.cpp
    digest/multi_ut.cpp
    digest/murmur_ut.cpp
    digest/sequence_ut.cpp
)

INCLUDE(${ARCADIA_ROOT}/util/tests/ya_util_tests.inc)

END()
