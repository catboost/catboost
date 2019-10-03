#pragma once

#include "label_helper_builder.h"

#include <util/generic/hash.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>


class TExternalLabelsHelper {
public:
    TExternalLabelsHelper() : Initialized(false) {};
    void Initialize(const TString& multiclassLabelParams);
    void Initialize(int approxDimension);
    TString GetVisibleClassNameFromClass(int classId) const;
    TString GetVisibleClassNameFromLabel(float label) const;
    int GetExternalIndex(int approxId) const;
    int GetExternalApproxDimension() const;
    bool IsInitialized() const;
private:
    bool Initialized;
    int ExternalApproxDimension;
    TVector<int> SignificantLabelsIds;
    TVector<TString> VisibleClassNames;
    THashMap<float, TString> LabelToName;
};
