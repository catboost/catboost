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
        contrib/libs/musl/arch/generic
        contrib/libs/musl/arch/x86_64
        contrib/libs/musl/extra
        contrib/libs/musl/include
    )
ENDIF ()

END()
