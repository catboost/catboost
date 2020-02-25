

RECURSE(
    flatc
    protoc
    python
    python3
    python3/pycc
    python3/src/Lib/lib2to3
    ragel5
    ragel6
    yasm
)

IF (NOT OS_WINDOWS)
    RECURSE(
    
)
ENDIF ()
