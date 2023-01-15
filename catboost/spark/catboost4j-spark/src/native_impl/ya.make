DLL_JAVA(catboost4j-spark-impl)

NO_WERROR()



SRCS(
    jni_helpers.cpp
    quantization.cpp
    native_impl.swg
)

EXTRADIR(bindings/swiglib)

PEERDIR(
    library/cpp/grid_creator
    catboost/libs/helpers
)

END()
