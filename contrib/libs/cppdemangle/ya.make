LIBRARY()

LICENSE(
    MIT
    BSD
)



NO_RUNTIME()

NO_COMPILER_WARNINGS()

ADDINCL(
    contrib/libs/cxxsupp/libcxx/include
)

IF (OS_WINDOWS)
    SRCS(
        demangle_stub.cpp
    )
ELSE ()
    IF (CPPDEMANGLE_DEBUG)
        CFLAGS(
            -DDEBUGGING
        )
    ENDIF()

    CFLAGS(
        -nostdinc++
    )

    SRCS(
        demangle.cpp
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
