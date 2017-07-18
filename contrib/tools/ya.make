RECURSE(
    protoc
    python
    python/src/Modules/expat
    ragel5
    ragel6
    yasm
)

IF (NOT OS_WINDOWS)
    RECURSE(
    
)
ENDIF ()
