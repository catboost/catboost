

RECURSE(
    cython
    flatc
    protoc
    python
    python3
    ragel5
    ragel6
    yasm
)

IF (NOT OS_WINDOWS)
    RECURSE(
    
)
ENDIF ()
