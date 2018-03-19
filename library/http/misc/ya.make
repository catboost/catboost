LIBRARY()



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
