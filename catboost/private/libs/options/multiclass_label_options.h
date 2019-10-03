#pragma once

#include "option.h"

#include <util/generic/vector.h>
#include <util/generic/string.h>

namespace NJson {
    class TJsonValue;
}

struct TMulticlassLabelOptions {
public:
    explicit TMulticlassLabelOptions();

    void Save(NJson::TJsonValue* options) const;
    void Load(const NJson::TJsonValue& options);

    void Validate();

    bool operator==(const TMulticlassLabelOptions& rhs) const;
    bool operator!=(const TMulticlassLabelOptions& rhs) const;

    NCatboostOptions::TOption<TVector<float>> ClassToLabel;
    NCatboostOptions::TOption<TVector<TString>> ClassNames;
    NCatboostOptions::TOption<int> ClassesCount;
};
