RECURSE(
    enum_parser
    fix_elf
    getpid1
    rescompiler
    rescompressor
    rorescompiler
)

IF (NOT OS_WINDOWS)
    RECURSE(
    
)
ENDIF()
