DLL_JAVA(catboost4j-spark-impl)

NO_WERROR()



SRCS(
    dataset_rows_reader.cpp
    data_provider_builders.cpp
    features_layout.cpp
    jni_helpers.cpp
    master.cpp
    meta_info.cpp
    model.cpp
    quantized_features_info.cpp
    quantization.cpp
    GLOBAL spark_quantized.cpp
    string.cpp
    native_impl.swg
    quantized_pool_serialization.cpp
    vector_output.cpp
    worker.cpp
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
    catboost/libs/model
    catboost/private/libs/algo
    catboost/private/libs/app_helpers
    catboost/private/libs/data_util
    catboost/private/libs/distributed
    catboost/private/libs/options
    catboost/private/libs/quantized_pool
)

IF (ARCH_AARCH64 OR OS_WINDOWS)
    ALLOCATOR(J)
ELSE()
    ALLOCATOR(LF)
ENDIF()

END()
