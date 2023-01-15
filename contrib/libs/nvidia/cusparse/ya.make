LIBRARY()

LICENSE(BSD-3-Clause)



NO_PLATFORM()

IF (HAVE_CUDA)
    PEERDIR(build/platform/cuda)

    IF (NOT OS_WINDOWS)
        LDFLAGS(-lcusparse_static)
    ELSE()
        LDFLAGS(cusparse.lib)
    ENDIF()

    IF (OS_LINUX)
        LDFLAGS(-lstdc++)
    ENDIF()

    IF (OS_DARWIN)
        LDFLAGS(-lc++)
    ENDIF()
ENDIF()

END()
