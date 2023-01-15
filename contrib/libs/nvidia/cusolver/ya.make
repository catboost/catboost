LIBRARY()

LICENSE(BSD-3-Clause)



NO_PLATFORM()

IF (HAVE_CUDA)
    PEERDIR(
        build/platform/cuda
        contrib/libs/nvidia/cublas
        contrib/libs/nvidia/cusparse
    )

    IF (NOT OS_WINDOWS)
        LDFLAGS(-lcusolver_static)
        IF (CUDA_VERSION VERSION_GE "10.0")
            LDFLAGS(-llapack_static)
        ENDIF()
    ELSE()
        LDFLAGS(cusolver.lib)
    ENDIF()
ENDIF()

END()
