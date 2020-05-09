UNITTEST_FOR(library/cpp/http/io)



PEERDIR(
    library/cpp/http/server
)

SRCS(
    chunk_ut.cpp
    compression_ut.cpp
    headers_ut.cpp
    stream_ut.cpp
)

END()
