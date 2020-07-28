

INCLUDE(${ARCADIA_ROOT}/build/prebuilt/rorescompiler/ya.make.prebuilt)

IF (NOT PREBUILT)
    MESSAGE(FATAL_ERROR Unsupported host platform for prebuilt rorescompiler tool)
ENDIF()
