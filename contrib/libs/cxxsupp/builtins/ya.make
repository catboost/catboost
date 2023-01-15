LIBRARY()

LICENSE(
    MIT
    BSD
)



NO_UTIL()
NO_RUNTIME()
NO_PLATFORM()
NO_COMPILER_WARNINGS()

IF (GCC OR CLANG)
    # Clang (maybe GCC too) LTO code generator leaves the builtin calls unresolved
    # even if they are available. After the code generation pass is done
    # a linker is forced to select original object files from this library again
    # as they contain unresolved symbols. But code generation is already done,
    # object files actually are not ELFs but an LLVM bytecode and we get
    # "member at xxxxx is not an ELF object" errors from the linker.

    # Just generate native code from the beginning.
    DISABLE(USE_LTO)
ENDIF()

SRCS(
    addtf3.c

    clzti2.c
    comparetf2.c

    divdc3.c
    divsc3.c
    divtf3.c
    divti3.c
    divxc3.c

    extenddftf2.c
    extendsftf2.c

    fixdfti.c
    fixsfti.c
    fixtfdi.c
    fixtfsi.c
    fixunsdfti.c
    fixunstfsi.c
    fixunstfdi.c
    fixunsxfti.c
    fixunssfti.c
    floatditf.c
    floatsitf.c
    floattidf.c
    floattisf.c
    floatunditf.c
    floatunsitf.c
    floatuntidf.c
    floatuntisf.c

    gcc_personality_v0.c

    int_util.c

    modti3.c
    muldc3.c
    muloti4.c
    mulsc3.c
    multf3.c
    mulxc3.c

    popcountdi2.c

    subtf3.c

    trunctfdf2.c
    trunctfsf2.c

    udivmodti4.c
    udivti3.c
    umodti3.c
)

IF (OS_DARWIN)
    SRCS(os_version_check.c)
ENDIF()

END()
