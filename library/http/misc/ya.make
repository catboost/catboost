LIBRARY()



GENERATE_ENUM_SERIALIZATION(httpcodes.h)

SRCS(
    httpcodes.cpp
    httpdate.cpp
    httpreqdata.cpp
    parsed_request.cpp
)

PEERDIR(
    library/digest/lower_case
)

END()
