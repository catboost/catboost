RECURSE(
    archiver
    archiver/alignment_test
    archiver/tests
    enum_parser
    fix_elf
    mtime0
    rescompiler
    rescompressor
    rorescompiler
    triecompiler
    triecompiler/build_tool
    triecompiler/lib
)

IF (NOT OS_WINDOWS)
    RECURSE(
    
)
ENDIF()
