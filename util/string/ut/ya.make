UNITTEST_FOR(util)



SRCS(
    string/builder_ut.cpp
    string/cast_ut.cpp
    string/escape_ut.cpp
    string/join_ut.cpp
    string/hex_ut.cpp
    string/printf_ut.cpp
    string/split_ut.cpp
    string/strip_ut.cpp
    string/subst_ut.cpp
    string/type_ut.cpp
    string/util_ut.cpp
    string/vector_ut.cpp
    string/ascii_ut.cpp
)

INCLUDE(${ARCADIA_ROOT}/util/tests/ya_util_tests.inc)

END()
