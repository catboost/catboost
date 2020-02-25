

IF (NOT USE_PREBUILT_PROTOC OR NOT HOST_OS_DARWIN AND NOT HOST_OS_LINUX AND NOT HOST_OS_WINDOWS)
    PROGRAM()

    NO_COMPILER_WARNINGS()

    PEERDIR(
        ADDINCL contrib/libs/protobuf
        contrib/libs/protobuf/protoc
    )

    SRCDIR(
        contrib/libs/protobuf/compiler
    )

    SRCS(
        main.cc
    )

    SET(IDE_FOLDER "contrib/tools")

    END()
ELSE()
    PREBUILT_PROGRAM()

    PEERDIR(build/external_resources/arcadia_protoc)

    PRIMARY_OUTPUT(${ARCADIA_PROTOC_RESOURCE_GLOBAL}/protoc${MODULE_SUFFIX})

    END()
ENDIF()
