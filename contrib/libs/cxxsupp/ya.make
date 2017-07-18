LIBRARY()



NO_PLATFORM()

IF (NOT USE_STL_SYSTEM)
    PEERDIR(
        contrib/libs/cxxsupp/libcxx-next
    )
ELSE()
    PEERDIR(
        contrib/libs/cxxsupp/system_stl
    )
ENDIF()

END()
