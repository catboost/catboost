LIBRARY()

NO_UTIL()



IF (OS_ANDROID)
    PEERDIR(
        library/cpp/malloc/system
    )
ELSE()
    PEERDIR(
        library/cpp/malloc/api
        contrib/libs/jemalloc
    )
    SRCS(
        malloc-info.cpp
    )
ENDIF()

END()
