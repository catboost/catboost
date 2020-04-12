LIBRARY()



SRCS(
    generated/uniscripts.cpp
    langs.cpp
)

PEERDIR(
    library/cpp/digest/lower_case
)

GENERATE_ENUM_SERIALIZATION(scripts.h)

END()
