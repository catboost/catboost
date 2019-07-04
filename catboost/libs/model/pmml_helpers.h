#pragma once

#include "fwd.h"

namespace NJson {
    class TJsonValue;
}

namespace NCatboost {
    namespace NPmml {
        void OutputModel(
            const TFullModel& model,
            const TString& modelFile,
            const NJson::TJsonValue& userParameters,
            const THashMap<ui32, TString>* catFeaturesHashToString = nullptr);
    }
}
