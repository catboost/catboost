#pragma once

#include <util/generic/hash.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>


class TFullModel;


class TExternalLabelsHelper {
public:
    TExternalLabelsHelper() : Initialized(false) {};
    TExternalLabelsHelper(const TFullModel& model);

    TString GetVisibleClassNameFromClass(int classId) const;
    TString GetVisibleClassNameFromLabel(float label) const;
    int GetExternalIndex(int approxId) const;
    int GetExternalApproxDimension() const;
    bool IsInitialized() const;

private:
    void InitializeImpl(const TString& multiclassLabelParams);
    void InitializeImpl(int approxDimension);
    void InitializeImpl(const TVector<TString>& binclassNames);

private:
    bool Initialized;
    int ExternalApproxDimension;
    TVector<int> SignificantLabelsIds;
    TVector<TString> VisibleClassNames;
    THashMap<float, TString> LabelToName;
};
