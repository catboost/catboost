RECURSE(
    enum_parser
    fix_elf
    rescompiler
    rescompressor
    rorescompiler
)

IF (NOT OS_WINDOWS)
    RECURSE(
    
)
ENDIF()
