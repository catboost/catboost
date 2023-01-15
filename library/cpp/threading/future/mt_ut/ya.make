UNITTEST_FOR(library/cpp/threading/future)



SRCS(
    future_mt_ut.cpp
)

IF(NOT SANITIZER_TYPE)
SIZE(SMALL)

ELSE()
SIZE(MEDIUM)

ENDIF()


END()
