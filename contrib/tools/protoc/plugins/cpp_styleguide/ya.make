

IF (NOT USE_PREBUILT_PROTOC OR NOT HOST_OS_DARWIN AND NOT HOST_OS_LINUX AND NOT HOST_OS_WINDOWS)
    PROGRAM()
    
    NO_COMPILER_WARNINGS()
    
    PEERDIR(
        contrib/libs/protobuf
        contrib/libs/protobuf/protoc
    )
    
    ADDINCL(contrib/libs/protobuf)
    
    SRCS(
        cpp_styleguide.cpp
    )
    SET(IDE_FOLDER "contrib/tools")
    
    END()
ELSE()
    PREBUILT_PROGRAM()

    PEERDIR(build/external_resources/arcadia_cpp_styleguide)

    PRIMARY_OUTPUT(${ARCADIA_CPP_STYLEGUIDE_RESOURCE_GLOBAL}/cpp_styleguide${MODULE_SUFFIX})

    END()
ENDIF()
