

IF (USE_PREBUILT_TOOLS)
    INCLUDE(${ARCADIA_ROOT}/build/prebuilt/contrib/tools/protoc/plugins/grpc_java/ya.make.prebuilt)
ENDIF()

IF (NOT PREBUILT)
    INCLUDE(${ARCADIA_ROOT}/contrib/tools/protoc/plugins/grpc_java/bin/ya.make)
ENDIF()

RECURSE(
    bin
)
