#pragma once

#include <util/generic/string.h>
#include <util/generic/vector.h>


namespace NJson {
    class TJsonValue;
}

class TFullModel;


class TExternalLabelsHelper {
public:
    TExternalLabelsHelper() : Initialized(false) {};
    TExternalLabelsHelper(const TFullModel& model);

    TString GetVisibleClassNameFromClass(int classId) const;
    int GetExternalIndex(int approxId) const;
    int GetExternalApproxDimension() const;
    bool IsInitialized() const;

private:
    void InitializeImpl(int approxDimension, const TString& classLabelParams);
    void InitializeImpl(int approxDimension);
    void InitializeImpl(const TVector<NJson::TJsonValue>& binclassLabels);

private:
    bool Initialized;
    int ExternalApproxDimension;
    TVector<int> SignificantLabelsIds;
    TVector<TString> VisibleClassNames;
};
