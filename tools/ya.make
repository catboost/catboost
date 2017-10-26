RECURSE(
    enum_parser
    fix_elf
    rescompressor
    rorescompiler
)

IF (NOT OS_WINDOWS)
    RECURSE(
    
)
ENDIF()
