LIBRARY()


SUBSCRIBER(g:util-subscribers)

NO_UTIL()

IF (TSTRING_IS_STD_STRING)
    CFLAGS(GLOBAL -DTSTRING_IS_STD_STRING)
ENDIF()

SRCS(
    date.cpp
    datetime.cpp
    enum.cpp
    holder_vector.cpp
    ip.cpp
    matrix.cpp
    memory.cpp
)

END()

RECURSE_FOR_TESTS(
    ut
)
