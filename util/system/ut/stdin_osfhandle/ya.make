PROGRAM()


SUBSCRIBER(g:util-subscribers)

SRCS(
    main.cpp
)

NO_UTIL()

BUILD_ONLY_IF(OS_WINDOWS)

END()
