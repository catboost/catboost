LIBRARY()

NO_RUNTIME()

LDFLAGS(-fuse-ld=gold)

CFLAGS(GLOBAL -I${CUDA_ROOT}/include)
LDFLAGS(-L${CUDA_ROOT}/lib64)

IF (NOT PIC)
    EXTRALIBS_STATIC(-lcublas_static -lcurand_static -lcudart_static -lculibos -lcusparse_static)
ELSE()
    EXTRALIBS(-lcublas -lcurand -lcudart -lcusparse)
ENDIF()

END()
