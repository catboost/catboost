DLL_JAVA(catboost4j-spark-impl)

NO_WERROR()



SRCS(
    data_provider_builders.cpp
    features_layout.cpp
    jni_helpers.cpp
    quantized_features_info.cpp
    quantization.cpp
    native_impl.swg
)

EXTRADIR(bindings/swiglib)

PEERDIR(
    library/cpp/dbg_output
    library/cpp/grid_creator
    library/cpp/json
    library/cpp/threading/local_executor
    catboost/libs/data
    catboost/libs/helpers
)

END()
