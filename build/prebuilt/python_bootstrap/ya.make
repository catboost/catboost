

INCLUDE(${ARCADIA_ROOT}/build/prebuilt/python_bootstrap/ya.make.prebuilt)

IF (NOT PREBUILT)
    MESSAGE(FATAL_ERROR Unsupported host platform for prebuilt python bootstrap tool)
ENDIF()
