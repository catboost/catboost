LIBRARY()



SRCS(
    serialize_char_span.cpp
)

PEERDIR(
    library/cpp/json/writer
    library/cpp/protobuf/util
    library/cpp/token/serialization/protos
    library/cpp/token
)

END()
