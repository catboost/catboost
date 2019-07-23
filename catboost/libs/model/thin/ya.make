LIBRARY()



SRCDIR(catboost/libs/model)

SRCS(
    ctr_data.cpp
    ctr_helpers.cpp
    ctr_provider.cpp
    ctr_value_table.cpp
    eval_processing.cpp
    features.cpp
    GLOBAL model_import_interface.cpp
    model.cpp
    online_ctr.cpp
    static_ctr_provider.cpp
    model_build_helper.cpp
    cpu/evaluator_impl.cpp
    cpu/formula_evaluator.cpp
    cpu/quantization.cpp
)

IF (HAVE_CUDA AND NOT GCC)
    INCLUDE(${ARCADIA_ROOT}/catboost/libs/cuda_wrappers/default_nvcc_flags.make.inc)

    SRC(cuda/evaluator.cu -fno-lto)

    SRCS(
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
    contrib/libs/flatbuffers
    library/binsaver
    library/containers/dense_hash
    library/dbg_output
    library/fast_exp
    library/json
    library/object_factory
    library/svnversion
)

GENERATE_ENUM_SERIALIZATION(ctr_provider.h)
GENERATE_ENUM_SERIALIZATION(enums.h)
GENERATE_ENUM_SERIALIZATION(features.h)
GENERATE_ENUM_SERIALIZATION(split.h)

END()
