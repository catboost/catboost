LIBRARY()

LICENSE(Apache-2.0 MIT)



SRCDIR(contrib/libs/cxxsupp/libcxx)

SRCS(
    src/filesystem/directory_iterator.cpp
    src/filesystem/int128_builtins.cpp
    src/filesystem/operations.cpp
)

END()
