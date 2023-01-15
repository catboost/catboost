UNITTEST()



PEERDIR(
    ADDINCL library/cpp/json/writer
)

SRCDIR(library/cpp/json/writer)

SRCS(
    json_ut.cpp
    json_value_ut.cpp
)

END()
