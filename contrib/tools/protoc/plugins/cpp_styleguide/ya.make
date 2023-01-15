

IF (USE_PREBUILT_TOOLS AND VALID_HOST_PLATFORM_FOR_COMMON_PREBUILT_TOOLS)
    PREBUILT_PROGRAM()

    PEERDIR(build/external_resources/arcadia_cpp_styleguide)

    PRIMARY_OUTPUT(${ARCADIA_CPP_STYLEGUIDE_RESOURCE_GLOBAL}/cpp_styleguide${MODULE_SUFFIX})

    END()
ELSE()
    PROGRAM()

    NO_COMPILER_WARNINGS()

    PEERDIR(
        contrib/libs/protoc
    )

    SRCS(
        cpp_styleguide.cpp
    )

    END()
ENDIF()
