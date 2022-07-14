LIBRARY()



SRCS(
    packers.cpp
    proto_packer.cpp
    region_packer.cpp
)

END()

RECURSE_FOR_TESTS(
    ut
)
