LIBRARY()



SRCS(
    range_ops.cpp
    stack_array.cpp
)

END()

RECURSE_FOR_TESTS(
    ut
)
