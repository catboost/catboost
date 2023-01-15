LIBRARY()

# Placeholders for new GCC 4.9.2 C++ ABI which is not present on older systems



SRCS(
    cxxabi.cpp
    stdcxx_bits.cpp
)

NO_UTIL()
NO_RUNTIME()

END()
