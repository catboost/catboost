LIBRARY()



SRCS(
    serialize_char_span.cpp
)

PEERDIR(
    library/json/writer
    library/protobuf/util
    library/token/serialization/protos
    library/token
)

END()
