%{
#include <library/cpp/grid_creator/binarization.h>
#include <catboost/private/libs/options/enums.h>
#include <catboost/libs/logging/logging_level.h>
#include <catboost/libs/model/enums.h>
%}

%include "enums.swg"

%include "defaults.i"

%javaconst(1);

enum class EBorderSelectionType {
    Median = 1,
    GreedyLogSum = 2,
    UniformAndQuantiles = 3,
    MinEntropy = 4,
    MaxLogSum = 5,
    Uniform = 6,
    GreedyMinEntropy = 7,
};

enum class EModelType {
    CatboostBinary = 0,
    AppleCoreML = 1,
    Cpp = 2,
    Python = 3,
    Json = 4,
    Onnx = 5,
    Pmml = 6,
    CPUSnapshot = 7,
};

enum class ECtrTableMergePolicy {
    FailIfCtrIntersects,
    LeaveMostDiversifiedTable,
    IntersectingCountersAverage
};

%include <catboost/private/libs/options/enums.h>
%include <catboost/libs/logging/logging_level.h>