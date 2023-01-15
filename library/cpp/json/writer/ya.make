LIBRARY()



PEERDIR(
    library/cpp/json/common
)

SRCS(
    json_value.cpp
    json.cpp
)

GENERATE_ENUM_SERIALIZATION(json_value.h)

END()
