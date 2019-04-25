LIBRARY()



NO_RUNTIME()
NO_COMPILER_WARNINGS()
DISABLE(USE_LTO)

IF (SANITIZER_TYPE STREQUAL memory)
    CFLAGS(-fPIC)
ENDIF ()

NO_SANITIZE()
NO_SANITIZE_COVERAGE()

SRCDIR(
    contrib/libs/libunwind_master/src
)

ADDINCL(
    contrib/libs/libunwind_master/include
)

CXXFLAGS(
    -nostdinc++
    -fno-rtti
    -fno-exceptions
    -funwind-tables
)

CONLYFLAGS(
    -std=c99
)

SRCS(
    libunwind.cpp
    Unwind-EHABI.cpp
)

IF (OS_DARWIN)
    SRCS(
        Unwind_AppleExtras.cpp
    )
ENDIF ()

SRCS(
    UnwindLevel1.c
    UnwindLevel1-gcc-ext.c
    Unwind-sjlj.c
)

SRCS(
    UnwindRegistersRestore.S
    UnwindRegistersSave.S
)

IF (MUSL)
    ADDINCL(
        contrib/libs/musl-1.1.20/arch/generic
        contrib/libs/musl-1.1.20/arch/x86_64
        contrib/libs/musl-1.1.20/extra
        contrib/libs/musl-1.1.20/include
    )
ENDIF ()

END()
