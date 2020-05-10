LIBRARY()



PEERDIR(
    library/cpp/threading/chunk_queue
)

IF (MUSL)
    # TODO
    NO_COMPILER_WARNINGS()
ENDIF()

CFLAGS(GLOBAL -DNETLIBA_WITH_NALF)

SRCDIR(library/netliba/socket)

SRCS(
    creators.cpp
    socket.cpp
    stdafx.cpp
)

END()
