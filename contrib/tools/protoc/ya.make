

IF (USE_PREBUILT_TOOLS AND VALID_HOST_PLATFORM_FOR_COMMON_PREBUILT_TOOLS)
    PREBUILT_PROGRAM()

    PEERDIR(build/external_resources/arcadia_protoc)

    PRIMARY_OUTPUT(${ARCADIA_PROTOC_RESOURCE_GLOBAL}/protoc${MODULE_SUFFIX})

    END()
ELSE()
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
ENDIF()
