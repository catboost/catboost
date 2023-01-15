LIBRARY()



SRCS(
    json_writer.cpp
    json_reader.cpp
    json_prettifier.cpp
    rapidjson_helpers.cpp
)

PEERDIR(
    contrib/libs/rapidjson
    library/cpp/json/common
    library/cpp/json/fast_sax
    library/cpp/json/writer
    library/cpp/string_utils/relaxed_escaper
)

END()
