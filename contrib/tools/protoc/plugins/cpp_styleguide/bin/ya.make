

PROGRAM(cpp_styleguide)

NO_COMPILER_WARNINGS()

PEERDIR(
    contrib/libs/protoc
)

SRCS(
    cpp_styleguide.cpp
)

SRCDIR(
    contrib/tools/protoc/plugins/cpp_styleguide
)

INCLUDE(${ARCADIA_ROOT}/build/prebuilt/contrib/tools/protoc/plugins/cpp_styleguide/ya.make.induced_deps)

END()
