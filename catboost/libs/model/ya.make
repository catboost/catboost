LIBRARY()



CFLAGS(-DONNX_ML=1 -DONNX_NAMESPACE=onnx)

SRCS(
    coreml_helpers.cpp
    ctr_data.cpp
    ctr_provider.cpp
    ctr_value_table.cpp
    eval_processing.cpp
    features.cpp
    json_model_helpers.cpp
    model.cpp
    online_ctr.cpp
    onnx_helpers.cpp
    pmml_helpers.cpp
    static_ctr_provider.cpp
    model_build_helper.cpp
    feature_calcer.cpp
    cpu/evaluator_impl.cpp
    cpu/formula_evaluator.cpp
    cpu/quantization.cpp
)

IF (HAVE_CUDA AND NOT GCC)
    INCLUDE(${ARCADIA_ROOT}/catboost/libs/cuda_wrappers/default_nvcc_flags.make.inc)

    SRCS(
        cuda/evaluator.cu
        cuda/evaluator.cpp
    )
    PEERDIR(
        catboost/libs/cuda_wrappers
    )
ELSE()
    SRCS(
        cuda/no_cuda_stub.cpp
    )
ENDIF()

PEERDIR(
    catboost/libs/cat_feature
    catboost/libs/ctr_description
    catboost/libs/helpers
    catboost/libs/logging
    catboost/libs/model/flatbuffers
    catboost/libs/options
    catboost/libs/model/model_export
    contrib/libs/coreml
    contrib/libs/flatbuffers
    contrib/libs/onnx
    library/binsaver
    library/containers/dense_hash
    library/dbg_output
    library/fast_exp
    library/json
    library/svnversion
)

GENERATE_ENUM_SERIALIZATION(ctr_provider.h)
GENERATE_ENUM_SERIALIZATION(features.h)
GENERATE_ENUM_SERIALIZATION(fwd.h)
GENERATE_ENUM_SERIALIZATION(split.h)

END()
