LIBRARY()



SRCS(
    yarchive.cpp
    yarchive.h
    directory_models_archive_reader.cpp
    directory_models_archive_reader.h
)

END()

RECURSE_FOR_TESTS(
    ut
)
