LIBRARY()



SRCS(
    http_parser.cpp
)

PEERDIR(
    library/http/io
    library/blockcodecs
)

END()

NEED_CHECK()
