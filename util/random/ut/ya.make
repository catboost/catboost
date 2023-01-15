UNITTEST_FOR(util)



SRCS(
    random/common_ops_ut.cpp
    random/easy_ut.cpp
    random/entropy_ut.cpp
    random/fast_ut.cpp
    random/normal_ut.cpp
    random/mersenne_ut.cpp
    random/random_ut.cpp
    random/shuffle_ut.cpp
)

INCLUDE(${ARCADIA_ROOT}/util/tests/ya_util_tests.inc)

END()
