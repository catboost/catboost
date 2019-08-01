RECURSE(
    enum_parser
    fix_elf
    mtime0
    rescompiler
    rescompressor
    rorescompiler
)

IF (NOT OS_WINDOWS)
    RECURSE(
    
)
ENDIF()
