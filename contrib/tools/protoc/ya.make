TOOL()

NO_COMPILER_WARNINGS()

PEERDIR(
    ADDINCL contrib/libs/protobuf
    contrib/libs/protobuf/protoc
)

SRCDIR(
    contrib/libs/protobuf/compiler
)

SRCS(
    main.cc
)

SET(IDE_FOLDER "contrib/tools")

END()
