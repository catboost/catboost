

INCLUDE(${ARCADIA_ROOT}/build/prebuilt/flatc/ya.make.prebuilt)

IF (NOT PREBUILT)
    MESSAGE(FATAL_ERROR Unsupported host platform for prebuilt flatc tool)
ENDIF()
