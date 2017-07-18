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
    CFLAGS(
        -nostdinc++
    )

    SRCS(
        demangle.cpp
    )
ENDIF ()

END()
