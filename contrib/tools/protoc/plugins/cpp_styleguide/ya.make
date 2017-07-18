TOOL()

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
