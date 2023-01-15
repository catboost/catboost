

IF (NOT USE_PREBUILT_PROTOC OR NOT HOST_OS_DARWIN AND NOT HOST_OS_LINUX AND NOT HOST_OS_WINDOWS)

    PROGRAM()

    LICENSE(
        BSD3
    )

    NO_COMPILER_WARNINGS()

    PEERDIR(
        contrib/libs/protoc
    )
    SRCDIR(
        contrib/libs/protoc
    )

    SRCS(
        src/google/protobuf/compiler/main.cc
    )

    END()
ELSE()
    PREBUILT_PROGRAM()

    PEERDIR(build/external_resources/arcadia_protoc)

    PRIMARY_OUTPUT(${ARCADIA_PROTOC_RESOURCE_GLOBAL}/protoc${MODULE_SUFFIX})

    END()
ENDIF()
