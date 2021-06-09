LIBRARY()

LICENSE(BSD-3-Clause)



NO_PLATFORM()

IF (HAVE_CUDA)
    PEERDIR(build/platform/cuda)

    IF (NOT OS_WINDOWS)
        LDFLAGS(-lcublas_static)
    ELSE()
        LDFLAGS(cublas.lib)
    ENDIF()

    IF (CUDA_VERSION VERSION_GE 10.1)
        IF (NOT OS_WINDOWS)
            LDFLAGS(-lcublasLt_static)
        ELSE()
            LDFLAGS(cublasLt.lib)
        ENDIF()
    ENDIF()

    IF (OS_LINUX)
        LDFLAGS(-lstdc++)
    ENDIF()
ENDIF()

END()
