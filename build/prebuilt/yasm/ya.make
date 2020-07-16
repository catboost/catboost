

INCLUDE(${ARCADIA_ROOT}/build/prebuilt/yasm/ya.make.prebuilt)

IF (NOT PREBUILT)
    MESSAGE(FATAL_ERROR Unsupported host platform for prebuilt yasm tool)
ENDIF()
