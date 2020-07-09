

IF (HOST_OS_DARWIN AND HOST_ARCH_X86_64 OR
    HOST_OS_LINUX AND HOST_ARCH_X86_64 OR
    HOST_OS_WINDOWS AND HOST_ARCH_X86_64)
    ENABLE(VALID_HOST_PLATFORM_FOR_PREBUILT_PROTOC)
ENDIF()

IF (NOT USE_PREBUILT_TOOLS OR NOT VALID_HOST_PLATFORM_FOR_PREBUILT_PROTOC)
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
