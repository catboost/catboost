

RECURSE(
    data
    data/ut
    data/benchmarks_ut
    eval_result
    fstr
    gpu_config
    helpers
    helpers/ut
    loggers
    logging
    metrics
    metrics/ut
    model
    model/model_export
    model/model_export/ut
    model/ut
    model_interface
    model_interface/static
    overfitting_detector
    monoforest
    train_lib
    train_lib/ut
    train_interface
)

IF (HAVE_CUDA)
    RECURSE(
    model/cuda
)
ENDIF()
