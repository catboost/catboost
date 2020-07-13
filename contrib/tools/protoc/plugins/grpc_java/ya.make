

IF (USE_PREBUILT_TOOLS AND VALID_HOST_PLATFORM_FOR_COMMON_PREBUILT_TOOLS)
    PREBUILT_PROGRAM()

    PEERDIR(build/external_resources/arcadia_grpc_java)

    PRIMARY_OUTPUT(${ARCADIA_GRPC_JAVA_RESOURCE_GLOBAL}/grpc_java${MODULE_SUFFIX})

    END()
ELSE()
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
