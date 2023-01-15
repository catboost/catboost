LIBRARY()



SRCS(
    serialize_char_span.cpp
)

PEERDIR(
    library/cpp/json/writer
    library/cpp/protobuf/util
    library/token/serialization/protos
    library/token
)

END()
