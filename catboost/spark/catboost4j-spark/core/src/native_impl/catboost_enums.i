%{
#include <library/cpp/grid_creator/binarization.h>
#include <catboost/private/libs/options/enums.h>
#include <catboost/libs/logging/logging_level.h>
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

%include <catboost/private/libs/options/enums.h>
%include <catboost/libs/logging/logging_level.h>