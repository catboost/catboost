#pragma once

#include "enums.h"
#include "option.h"

#include <library/cpp/json/json_value.h>

#include <util/generic/vector.h>


struct TClassLabelOptions {
public:
    explicit TClassLabelOptions();

    void Save(NJson::TJsonValue* options) const;
    void Load(const NJson::TJsonValue& options);

    void Validate();

    bool operator==(const TClassLabelOptions& rhs) const;
    bool operator!=(const TClassLabelOptions& rhs) const;

    /* It is necessary to save label type explicitly because JSON format does not distinguish between
     * Integers and Floats
     */
    NCatboostOptions::TOption<NCB::ERawTargetType> ClassLabelType;
    NCatboostOptions::TOption<TVector<float>> ClassToLabel;
    NCatboostOptions::TOption<TVector<NJson::TJsonValue>> ClassLabels;  // can be Integers, Floats or Strings
    NCatboostOptions::TOption<int> ClassesCount;
};
