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

IF (MUSL)
    ADDINCL(
        contrib/libs/musl-1.1.20/arch/generic
        contrib/libs/musl-1.1.20/arch/x86_64
        contrib/libs/musl-1.1.20/extra
        contrib/libs/musl-1.1.20/include
    )
ENDIF ()

END()
