LIBRARY()



IF (MUSL)
    # TODO
    NO_COMPILER_WARNINGS()
ENDIF ()

PEERDIR(
    library/threading/chunk_queue
)

SRCS(
    creators.cpp
    socket.cpp
    stdafx.cpp
)

END()
