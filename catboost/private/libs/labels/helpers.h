#pragma once

#include <catboost/private/libs/options/enums.h>

#include <library/cpp/json/json_value.h>

#include <util/generic/fwd.h>
#include <util/generic/array_ref.h>
#include <util/generic/string.h>


namespace NCB {
    ERawTargetType GetRawTargetType(const NJson::TJsonValue& classLabel);

    TString ClassLabelToString(const NJson::TJsonValue& classLabel);

    TVector<TString> ClassLabelsToStrings(TConstArrayRef<NJson::TJsonValue> classLabels);

    // For AllowConstLabel: Training won't work properly with single-dimensional approx for multiclass, so make a
    // 'phantom' second dimension.
    void MaybeAddPhantomSecondClass(TVector<NJson::TJsonValue>* classLabels);

    void CheckBooleanClassLabels(TConstArrayRef<NJson::TJsonValue> booleanClassLabels);
}
