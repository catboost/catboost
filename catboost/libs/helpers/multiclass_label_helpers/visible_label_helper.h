#pragma once

#include <util/generic/hash.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>


class TVisibleLabelsHelper {
public:
    TVisibleLabelsHelper() : Initialized(false) {};
    void Initialize(const TString& multiclassLabelParams);
    void Initialize(int approxDimension);
    TString GetVisibleClassName(int classId) const;
    int GetVisibleIndex(int approxId) const;
    int GetVisibleApproxDimension() const;
    bool IsInitialized() const;
private:
    bool Initialized;
    int VisibleApproxDimension;
    TVector<float> ClassToLabel;
    TVector<int> SignificantLabelsIds;
    TVector<TString> VisibleClassNames;
};
