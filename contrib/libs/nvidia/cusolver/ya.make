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
        LDFLAGS(
            -lcusolver_static
            -llapack_static
        )
    ELSE()
        LDFLAGS(cusolver.lib)
    ENDIF()
ENDIF()

END()
