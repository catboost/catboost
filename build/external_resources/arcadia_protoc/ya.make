RESOURCES_LIBRARY()



# Clone this task to rebuild protoc from trunk:
#
# https://sandbox.yandex-team.ru/task/544339004/view

IF (NOT HOST_OS_DARWIN AND NOT HOST_OS_LINUX AND NOT HOST_OS_WINDOWS)
    MESSAGE(FATAL_ERROR Unsupported host platform for prebuilt arcadia protoc)
ENDIF()

DECLARE_EXTERNAL_HOST_RESOURCES_BUNDLE(
    ARCADIA_PROTOC
    sbr:1345410406 FOR DARWIN
    sbr:1345410624 FOR LINUX
    sbr:1345410517 FOR WIN32
)

END()
