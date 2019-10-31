LIBRARY()



IF (ARCH_X86_64)
    PEERDIR(
        library/float16/float16_avx_impl
    )
ELSE()
    PEERDIR(
        library/float16/float16_dummy_impl
    )
ENDIF()

SRCS(
    float16.cpp
)

END()
RECURSE_FOR_TESTS(ut)
