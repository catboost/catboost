LIBRARY()



SRC_C_AVX(float16_avx.cpp)

IF (NOT MSVC OR CLANG_CL)
    CFLAGS(-mf16c)
ENDIF()

END()
