LIBRARY()



SRCS(
    chunked_helpers.cpp
    reader.cpp
    writer.cpp
)

END()

RECURSE_FOR_TESTS(
    ut
)
