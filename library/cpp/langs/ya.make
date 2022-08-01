LIBRARY()



SRCS(
    generated/uniscripts.cpp
    langs.cpp
    scripts.cpp
)

PEERDIR(
    library/cpp/digest/lower_case
)

END()
