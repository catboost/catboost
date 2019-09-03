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
        contrib/libs/musl/arch/generic
        contrib/libs/musl/arch/x86_64
        contrib/libs/musl/extra
        contrib/libs/musl/include
    )
ENDIF ()

END()
