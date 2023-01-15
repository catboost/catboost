

UNITTEST_FOR(library/cpp/threading/name_guard)

IF (OS_LINUX)
    SRCS(
        name_guard_ut.cpp
    )
ELSE()
    SRCS(
        dummy_ut.cpp
    )
ENDIF()

END()
