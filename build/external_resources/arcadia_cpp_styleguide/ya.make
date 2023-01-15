RESOURCES_LIBRARY()



# Clone this task to rebuild cpp_styleguide from trunk:
#
# https://sandbox.yandex-team.ru/task/607000354/view

IF (NOT HOST_OS_DARWIN AND NOT HOST_OS_LINUX AND NOT HOST_OS_WINDOWS)
    MESSAGE(FATAL_ERROR Unsupported host platform for prebuilt arcadia cpp_styleguide)
ENDIF()

DECLARE_EXTERNAL_HOST_RESOURCES_BUNDLE(
    ARCADIA_CPP_STYLEGUIDE
    sbr:1546865733 FOR DARWIN
    sbr:1546866028 FOR LINUX
    sbr:1546865971 FOR WIN32
)

END()
