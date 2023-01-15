LIBRARY()



SRCS(
    linear_regression.cpp
    unimodal.cpp
    welford.cpp
)

PEERDIR(
    library/cpp/accurate_accumulate
)

END()
