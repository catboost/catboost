

IF (USE_PREBUILT_TOOLS)
    INCLUDE(${ARCADIA_ROOT}/build/prebuilt/contrib/tools/protoc/plugins/grpc_java/ya.make.prebuilt)
ENDIF()

IF (NOT PREBUILT)
    PROGRAM()

    NO_COMPILER_WARNINGS()

    PEERDIR(
        contrib/libs/protoc
    )

    SRCDIR(contrib/libs/grpc-java/compiler/src/java_plugin/cpp)

    SRCS(
        java_plugin.cpp
        java_generator.cpp
    )

    END()
ENDIF()
