

RECURSE(
    algo
    algo/ut
    app_helpers
    data_new
    data_new/ut
    data_types
    data_util
    data_util/ut
    distributed
    documents_importance
    eval_result
    fstr
    gpu_config
    helpers
    helpers/ut
    index_range
    init
    labels
    lapack
    loggers
    logging
    metrics
    metrics/ut
    model
    model/model_export/ut
    model/ut
    model_interface
    options
    options/ut
    overfitting_detector
    quantization
    quantization_schema
    quantization_schema/ut
    quantized_pool
    quantized_pool_analysis
    quantized_pool/ut
    target
    train_lib
    train_lib/ut
    validate_fb
    feature_estimator
    text_features
    text_features/ut
    train_interface
)

IF (HAVE_CUDA)
RECURSE(
    cuda_wrappers
)
ENDIF()

IF (NOT OS_WINDOWS)
    RECURSE(
    model_interface/static
)
ENDIF()
