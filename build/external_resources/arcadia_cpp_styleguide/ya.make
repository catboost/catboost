RESOURCES_LIBRARY()



# Clone this task to rebuild cpp_styleguide from trunk:
#
# https://sandbox.yandex-team.ru/task/607000354/view

IF (NOT HOST_OS_DARWIN AND NOT HOST_OS_LINUX AND NOT HOST_OS_WINDOWS)
    MESSAGE(FATAL_ERROR Unsupported host platform for prebuilt arcadia cpp_styleguide)
ENDIF()

DECLARE_EXTERNAL_HOST_RESOURCES_BUNDLE(
    ARCADIA_CPP_STYLEGUIDE
    sbr:1489922153 FOR DARWIN
    sbr:1489922290 FOR LINUX
    sbr:1489922205 FOR WIN32
)

END()
