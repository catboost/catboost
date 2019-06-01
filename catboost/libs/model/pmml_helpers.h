#pragma once

#include <util/generic/hash.h>
#include <util/generic/string.h>
#include <util/system/types.h>


namespace NJson {
    class TJsonValue;
}

struct TFullModel;


namespace NCatboost {
    namespace NPmml {
        void OutputModel(
            const TFullModel& model,
            const TString& modelFile,
            const NJson::TJsonValue& userParameters,
            const THashMap<ui32, TString>* catFeaturesHashToString = nullptr);
    }
}
