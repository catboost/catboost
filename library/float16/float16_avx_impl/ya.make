LIBRARY()



SRC_CPP_AVX(
    float16_avx.cpp
)

CFLAGS(
    -mf16c
)

END()

NEED_CHECK()
