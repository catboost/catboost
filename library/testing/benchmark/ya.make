LIBRARY()



SRCS(
    bench.cpp
    dummy.cpp
)

PEERDIR(
    contrib/libs/re2
    library/colorizer
    library/getopt/small
    library/json
    library/linear_regression
    library/threading/poor_man_openmp
)

END()
