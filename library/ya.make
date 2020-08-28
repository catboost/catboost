IF (NOT SANITIZER_TYPE)
    RECURSE(
    
)
ENDIF()

PEERDIR(
    contrib/libs/cxxsupp/openmp
)

NEED_CHECK()
