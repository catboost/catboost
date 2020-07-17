

INCLUDE(${ARCADIA_ROOT}/build/prebuilt/protoc/ya.make.prebuilt)

IF (NOT PREBUILT)
    MESSAGE(FATAL_ERROR Unsupported host platform for prebuilt protoc tool)
ENDIF()
