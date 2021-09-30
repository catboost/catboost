LIBRARY()

WITHOUT_LICENSE_TEXTS()

LICENSE(MIT)

VERSION(2021-04-02-7959d59028dd126416cdf10dbbd22162922e1336)



SRCDIR(contrib/libs/cxxsupp/libcxx)

SRCS(
    src/filesystem/directory_iterator.cpp
    src/filesystem/int128_builtins.cpp
    src/filesystem/operations.cpp
)

END()
