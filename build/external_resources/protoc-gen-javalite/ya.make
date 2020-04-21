

RESOURCES_LIBRARY()

# protoc-gen-javalite-3.0.0 binaries
# downloaded from https://repo1.maven.org/maven2/com/google/protobuf/protoc-gen-javalite/3.0.0/

IF (NOT HOST_OS_DARWIN AND NOT HOST_OS_LINUX AND NOT HOST_OS_WINDOWS)
    MESSAGE(FATAL_ERROR Unsupported host platform for protoc-gen-javalite)
ENDIF()

DECLARE_EXTERNAL_HOST_RESOURCES_BUNDLE(
    PROTOC_GEN_JAVALITE
    sbr:1325737526 FOR DARWIN
    sbr:1325730420 FOR LINUX
    sbr:1325766529 FOR WIN32
)

END()
