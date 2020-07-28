

INCLUDE(${ARCADIA_ROOT}/build/prebuilt/protoc_std/ya.make.prebuilt)

IF (NOT PREBUILT)
    MESSAGE(FATAL_ERROR Unsupported host platform for prebuilt protoc_std tool)
ENDIF()
