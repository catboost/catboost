LIBRARY()



NO_COMPILER_WARNINGS()

PEERDIR(
    contrib/libs/cblas
    contrib/libs/clapack
)

SRCS(
#   apple_sgemv_fix.c
    wrap_accelerate_c.c
    wrap_accelerate_f.f
#   wrap_dummy_accelerate.f
#   wrap_dummy_g77_abi.f
    wrap_g77_abi_c.c
    wrap_g77_abi_f.f
)

END()
