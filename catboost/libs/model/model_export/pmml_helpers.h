#pragma once

#include <catboost/libs/model/fwd.h>

namespace NJson {
    class TJsonValue;
}

namespace NCB {
    namespace NPmml {
        void OutputModel(
            const TFullModel& model,
            const TString& modelFile,
            const NJson::TJsonValue& userParameters,
            const THashMap<ui32, TString>* catFeaturesHashToString = nullptr);
    }
}
