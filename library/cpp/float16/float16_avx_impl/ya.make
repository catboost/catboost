LIBRARY()



SRC_CPP_AVX(float16_avx.cpp)

IF (NOT MSVC)
    CFLAGS(-mf16c)
ENDIF()

END()
