RECURSE(
    flatc
    protoc
    python
    python/src/Modules/expat
    python3
    ragel5
    ragel6
    yasm
)

IF (NOT OS_WINDOWS)
    RECURSE(
    
)
ENDIF ()
