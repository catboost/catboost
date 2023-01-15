

INCLUDE(${ARCADIA_ROOT}/build/prebuilt/protoc-gen-go/ya.make.prebuilt)

IF (NOT PREBUILT)
    MESSAGE(FATAL_ERROR Unsupported host platform for prebuilt protoc-gen-go tool)
ENDIF()
