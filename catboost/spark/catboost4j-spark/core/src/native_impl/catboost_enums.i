%{
#include <library/cpp/grid_creator/binarization.h>
#include <catboost/private/libs/options/enums.h>
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

%javaconst(1);
enum class ENanMode {
    Min,
    Max,
    Forbidden
};

%javaconst(1);
enum class EFeatureType {
    Float,
    Categorical,
    Text,
    Embedding
};


namespace NCB {

    %javaconst(1);
    enum class ERawTargetType : ui32 {
        Integer,
        Float,
        String,
        None
    };
}
