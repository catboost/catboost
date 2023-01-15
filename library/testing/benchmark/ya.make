LIBRARY()



SRCS(
    bench.cpp
    dummy.cpp
)

PEERDIR(
    contrib/libs/re2
    library/cpp/colorizer
    library/cpp/getopt/small
    library/json
    library/linear_regression
    library/cpp/threading/poor_man_openmp
)

END()
