LIBRARY()

LICENSE(MIT)

# Placeholders for new GCC 4.9.2 C++ ABI which is not present on older systems



IF (NOT OS_WINDOWS)
    SRCS(
        cxxabi.cpp
        stdcxx_bits.cpp
    )
ENDIF()

NO_UTIL()

END()
