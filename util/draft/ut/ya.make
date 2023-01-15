UNITTEST()



SRCDIR(util/draft)

PEERDIR(
    util/draft
)

SRCS(
    date_ut.cpp
    datetime_ut.cpp
    holder_vector_ut.cpp
    memory_ut.cpp
)

INCLUDE(${ARCADIA_ROOT}/util/tests/ya_util_tests.inc)

END()
