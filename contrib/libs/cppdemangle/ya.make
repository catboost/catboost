LIBRARY()

LICENSE(
    MIT
    BSD
)



NO_RUNTIME()

NO_COMPILER_WARNINGS()

IF (OS_WINDOWS)
    SRCS(
        demangle_stub.cpp
    )
ELSE ()
    IF (CPPDEMANGLE_DEBUG)
        ADDINCL(
            contrib/libs/cxxsupp/libcxx/include
        )

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

END()
