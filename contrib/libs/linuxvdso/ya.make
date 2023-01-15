LIBRARY()

LICENSE(
    BSD3
)



NO_UTIL()
NO_RUNTIME()

IF (OS_LINUX)
    PEERDIR(
        contrib/libs/linuxvdso/original
    )

    SRCS(
        interface.cpp
    )
ELSE ()
    SRCS(
        fake.cpp
    )
ENDIF ()

END()
