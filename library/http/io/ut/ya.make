UNITTEST_FOR(library/http/io)



PEERDIR(
    library/http/server
)

SRCS(
    chunk_ut.cpp
    compression_ut.cpp
    headers_ut.cpp
    stream_ut.cpp
)

END()
