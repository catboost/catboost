IF (NOT USE_PREBUILT_PROTOC OR NOT HOST_OS_DARWIN AND NOT HOST_OS_LINUX AND NOT HOST_OS_WINDOWS)
    PROGRAM()
    
    
    
    NO_COMPILER_WARNINGS()
    
    PEERDIR(
        contrib/libs/protobuf
        contrib/libs/protobuf/protoc
    )
    
    SRCDIR(contrib/libs/grpc-java/compiler/src/java_plugin/cpp)
    
    SRCS(
        java_plugin.cpp
        java_generator.cpp
    )
    
    END()
ELSE()
    PREBUILT_PROGRAM()

    PEERDIR(build/external_resources/arcadia_grpc_java)

    PRIMARY_OUTPUT(${ARCADIA_GRPC_JAVA_RESOURCE_GLOBAL}/grpc_java${MODULE_SUFFIX})

    END()
ENDIF()
