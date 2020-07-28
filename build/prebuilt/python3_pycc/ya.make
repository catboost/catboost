

INCLUDE(${ARCADIA_ROOT}/build/prebuilt/python3_pycc/ya.make.prebuilt)

IF (NOT PREBUILT)
    MESSAGE(FATAL_ERROR Unsupported host platform for prebuilt python3 pycc tool)
ENDIF()
