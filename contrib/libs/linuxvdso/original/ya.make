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

END()
