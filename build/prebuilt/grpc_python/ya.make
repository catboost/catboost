

INCLUDE(${ARCADIA_ROOT}/build/prebuilt/grpc_python/ya.make.prebuilt)

IF (NOT PREBUILT)
    MESSAGE(FATAL_ERROR Unsupported host platform for prebuilt grpc_python tool)
ENDIF()
