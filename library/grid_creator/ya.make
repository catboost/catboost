LIBRARY()



SRCS(
    binarization.cpp
    median_in_bin_binarization.cpp
)

GENERATE_ENUM_SERIALIZATION(
    binarization.h
)

END()
