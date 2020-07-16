

INCLUDE(${ARCADIA_ROOT}/build/prebuilt/ragel6/ya.make.prebuilt)

IF (NOT PREBUILT)
    MESSAGE(FATAL_ERROR Unsupported host platform for prebuilt ragel6 tool)
ENDIF()
