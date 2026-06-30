#pragma once

namespace NCB { // split due to CUDA-compiler inability to support nested namespace definitions
    namespace NModelEvaluation {
        enum class EPredictionType {
            RawFormulaVal = 0,
            Exponent = 1,
            RMSEWithUncertainty = 2,
            Probability = 3,
            Class = 4,
            MultiProbability = 5
        };
    }
}

enum class EFormulaEvaluatorType {
    CPU,
    GPU
};

// TODO(kirillovs): move inside NCB namespace
enum class EModelType {
    CatboostBinary /* "CatboostBinary", "cbm", "catboost" */,
    AppleCoreML    /* "AppleCoreML", "coreml"     */,
    Cpp            /* "Cpp", "CPP", "cpp" */,
    Python         /* "Python", "python" */,
    Json           /* "Json", "json"       */,
    Onnx           /* "Onnx", "onnx" */,
    Pmml           /* "PMML", "pmml" */,
    CPUSnapshot    /* "CpuSnapshot" */
};
