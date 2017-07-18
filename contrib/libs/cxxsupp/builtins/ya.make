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
    udivmodti4.c
    udivti3.c
    umodti3.c

    mulsc3.c
    muldc3.c
    mulxc3.c

    divsc3.c
    divxc3.c
    divdc3.c

    clzti2.c
    fixunsdfti.c
    floatuntidf.c

    int_util.c
    gcc_personality_v0.c
)

END()
