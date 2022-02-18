LIBRARY()

WITHOUT_LICENSE_TEXTS()

# Proxy library
LICENSE(Not-Applicable)



NO_PLATFORM()

IF (NOT USE_STL_SYSTEM)
    PEERDIR(
        contrib/libs/cxxsupp/libcxx
    )
ELSE()
    PEERDIR(
        contrib/libs/cxxsupp/system_stl
    )
ENDIF()

END()

RECURSE(
    libcxx
    libcxxabi
    libcxxrt
)
