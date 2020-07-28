

INCLUDE(${ARCADIA_ROOT}/build/prebuilt/rescompressor/ya.make.prebuilt)

IF (NOT PREBUILT)
    MESSAGE(FATAL_ERROR Unsupported host platform for prebuilt rescompressor tool)
ENDIF()
