LIBRARY()



IF (MUSL)
    # TODO
    NO_COMPILER_WARNINGS()
ENDIF()

PEERDIR(
    library/cpp/threading/chunk_queue
)

SRCS(
    creators.cpp
    socket.cpp
    stdafx.cpp
)

END()
