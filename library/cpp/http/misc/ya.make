LIBRARY()



GENERATE_ENUM_SERIALIZATION(httpcodes.h)

SRCS(
    httpcodes.cpp
    httpdate.cpp
    httpreqdata.cpp
    parsed_request.cpp
)

PEERDIR(
    library/cpp/cgiparam
    library/cpp/digest/lower_case
)

END()

RECURSE_FOR_TESTS(ut)
