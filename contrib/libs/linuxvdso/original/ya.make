LIBRARY()

LICENSE(
    BSD3
)



NO_UTIL()
NO_RUNTIME()
NO_COMPILER_WARNINGS()

SRCS(
    vdso_support.cc
    elf_mem_image.cc
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
