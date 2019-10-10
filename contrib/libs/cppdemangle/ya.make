LIBRARY()

LICENSE(
    MIT
    BSD
)



NO_RUNTIME()

NO_COMPILER_WARNINGS()

IF (NOT USE_STL_SYSTEM)
    ADDINCL(
        contrib/libs/cxxsupp/libcxx/include
    )
ENDIF()

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
        contrib/libs/musl/arch/x86_64
        contrib/libs/musl/arch/generic
        contrib/libs/musl/include
        contrib/libs/musl/extra
    )
ENDIF ()

END()
