

INCLUDE(${ARCADIA_ROOT}/build/prebuilt/rescompiler/ya.make.prebuilt)

IF (NOT PREBUILT)
    MESSAGE(FATAL_ERROR Unsupported host platform for prebuilt rescompiler tool)
ENDIF()
