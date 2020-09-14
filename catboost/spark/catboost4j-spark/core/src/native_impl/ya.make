DLL_JAVA(catboost4j-spark-impl)

NO_WERROR()



SRCS(
    dataset_rows_reader.cpp
    data_provider_builders.cpp
    features_layout.cpp
    jni_helpers.cpp
    meta_info.cpp
    quantized_features_info.cpp
    quantization.cpp
    string.cpp
    native_impl.swg
)

EXTRADIR(bindings/swiglib)

PEERDIR(
    library/cpp/dbg_output
    library/cpp/grid_creator
    library/cpp/json
    library/cpp/threading/local_executor
    catboost/libs/column_description
    catboost/libs/data
    catboost/libs/helpers
    catboost/libs/logging
    catboost/private/libs/data_util
    catboost/private/libs/options
)

END()
