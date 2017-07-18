LIBRARY()



SRCS(
    json_writer.cpp
    json_reader.cpp
    json_prettifier.cpp
    rapidjson_helpers.cpp
)

PEERDIR(
    contrib/libs/rapidjson
    library/json/writer
    library/json/common
    library/json/fast_sax
    library/string_utils/relaxed_escaper
)

END()
