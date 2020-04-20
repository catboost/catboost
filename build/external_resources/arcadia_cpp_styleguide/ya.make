RESOURCES_LIBRARY()



# Clone this task to rebuild cpp_styleguide from trunk:
#
# https://sandbox.yandex-team.ru/task/607000354/view

IF (NOT HOST_OS_DARWIN AND NOT HOST_OS_LINUX AND NOT HOST_OS_WINDOWS)
    MESSAGE(FATAL_ERROR Unsupported host platform for prebuilt arcadia cpp_styleguide)
ENDIF()

DECLARE_EXTERNAL_HOST_RESOURCES_BUNDLE(
    ARCADIA_CPP_STYLEGUIDE
    sbr:1345419574 FOR DARWIN
    sbr:1345419942 FOR LINUX
    sbr:1345419804 FOR WIN32
)

END()
