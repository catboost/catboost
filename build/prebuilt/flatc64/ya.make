

INCLUDE(${ARCADIA_ROOT}/build/prebuilt/flatc64/ya.make.prebuilt)

IF (NOT PREBUILT)
    MESSAGE(FATAL_ERROR Unsupported host platform for prebuilt flatc64 tool)
ENDIF()
