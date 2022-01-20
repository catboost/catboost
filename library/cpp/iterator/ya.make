

LIBRARY()

SRCS(
    cartesian_product.cpp
    concatenate.cpp
    enumerate.cpp
    iterate_keys.cpp
    iterate_values.cpp
    filtering.cpp
    functools.cpp
    mapped.cpp
    zip.cpp
)

END()

RECURSE_FOR_TESTS(ut)
