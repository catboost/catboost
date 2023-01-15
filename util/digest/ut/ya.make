UNITTEST_FOR(util)



SRCS(
    digest/fnv_ut.cpp
    digest/murmur_ut.cpp
    digest/multi_ut.cpp
    digest/sequence_ut.cpp
)

INCLUDE(${ARCADIA_ROOT}/util/tests/ya_util_tests.inc)

END()
