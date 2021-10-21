LIBRARY()



SRCS(
    dot_product.cpp
    dot_product_sse.cpp
    dot_product_simple.cpp
)

IF (USE_SSE4 == "yes" AND OS_LINUX == "yes")
    SRC_CPP_AVX2(dot_product_avx2.cpp -mfma)
ELSE()
    SRC(dot_product_avx2.cpp)
ENDIF()

PEERDIR(
    library/cpp/sse
    library/cpp/testing/common
)

END()
