RESOURCES_LIBRARY()

# protoc-gen-javalite-3.0.0 binaries
# downloaded from https://repo1.maven.org/maven2/com/google/protobuf/protoc-gen-javalite/3.0.0/
IF(HOST_OS_DARWIN)
    # protoc-gen-javalite-3.0.0-osx-x86_64.exe
    DECLARE_EXTERNAL_RESOURCE(PROTOC_GEN_JAVALITE sbr:1325737526)
ELSEIF(HOST_OS_LINUX)
    # protoc-gen-javalite-3.0.0-linux-x86_64.exe
    DECLARE_EXTERNAL_RESOURCE(PROTOC_GEN_JAVALITE sbr:1325730420)
ELSEIF(HOST_OS_WINDOWS)
    # protoc-gen-javalite-3.0.0-windows-x86_64.exe
    DECLARE_EXTERNAL_RESOURCE(PROTOC_GEN_JAVALITE sbr:1325766529)
ELSE()
    MESSAGE(FATAL_ERROR Unsupported host platfrom for protoc-gen-javalite)
ENDIF()
END()
