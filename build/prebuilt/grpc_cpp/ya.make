

INCLUDE(${ARCADIA_ROOT}/build/prebuilt/grpc_cpp/ya.make.prebuilt)

IF (NOT PREBUILT)
    MESSAGE(FATAL_ERROR Unsupported host platform for prebuilt grpc_cpp tool)
ENDIF()
