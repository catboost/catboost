#pragma once

#include <catboost/private/libs/options/enums.h>

#include <util/generic/fwd.h>
#include <util/generic/array_ref.h>
#include <util/generic/string.h>


namespace NJson {
    class TJsonValue;
}


namespace NCB {
    ERawTargetType GetRawTargetType(const NJson::TJsonValue& classLabel);

    TString ClassLabelToString(const NJson::TJsonValue& classLabel);

    TVector<TString> ClassLabelsToStrings(TConstArrayRef<NJson::TJsonValue> classLabels);

}
