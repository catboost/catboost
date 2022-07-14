UNITTEST_FOR(library/cpp/threading/future)



SRCS(
    future_mt_ut.cpp
)

IF(SANITIZER_TYPE)
    SIZE(MEDIUM)
ENDIF()


END()
