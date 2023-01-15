UNITTEST_FOR(util)



SRCS(
    memory/addstorage_ut.cpp
    memory/blob_ut.cpp
    memory/pool_ut.cpp
    memory/smallobj_ut.cpp
    memory/tempbuf_ut.cpp
)

INCLUDE(${ARCADIA_ROOT}/util/tests/ya_util_tests.inc)

END()
