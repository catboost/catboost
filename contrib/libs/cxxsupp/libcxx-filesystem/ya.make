LIBRARY()

WITHOUT_LICENSE_TEXTS()

LICENSE(MIT)



SRCDIR(contrib/libs/cxxsupp/libcxx)

SRCS(
    src/filesystem/directory_iterator.cpp
    src/filesystem/int128_builtins.cpp
    src/filesystem/operations.cpp
)

END()
