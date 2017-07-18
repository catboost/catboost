LIBRARY()

NO_UTIL()



IF (OS_ANDROID)
    PEERDIR(
        library/malloc/system
    )
ELSE()
    PEERDIR(
        library/malloc/api
        contrib/libs/jemalloc
    )

    SRCS(
        malloc-info.cpp
    )
ENDIF()

END()
