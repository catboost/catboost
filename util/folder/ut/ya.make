UNITTEST_FOR(util)



SRCS(
    folder/dirut_ut.cpp
    folder/filelist_ut.cpp
    folder/fts_ut.cpp
    folder/iterator_ut.cpp
    folder/path_ut.cpp
    folder/pathsplit_ut.cpp
)

INCLUDE(${ARCADIA_ROOT}/util/tests/ya_util_tests.inc)

END()
