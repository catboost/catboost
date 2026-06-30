#pragma once

#include <library/cpp/binsaver/bin_saver.h>

#include <util/generic/array_ref.h>
#include <util/generic/hash_set.h>
#include <util/generic/hash.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>


namespace NJson {
    class TJsonValue;
}


class TLabelConverter {
public:
    TLabelConverter() : Initialized(false) {};

    bool operator==(const TLabelConverter& rhs) const;

    SAVELOAD(MultiClass, LabelToClass, ClassToLabel, ClassesCount, Initialized);

    void Initialize(bool isMultiClass, const TString& classLabelParams);
    void InitializeBinClass();
    void InitializeMultiClass(int approxDimension);
    void InitializeMultiClass(TConstArrayRef<float> targets, int classesCount, bool allowConstLabel);

    void ValidateLabels(TConstArrayRef<float> labels) const;

    int GetApproxDimension() const;
    int GetClassIdx(float label) const;
    bool IsInitialized() const;
    bool IsMultiClass() const;

    TString SerializeClassParams(int classesCount, const TVector<NJson::TJsonValue>& classLabels) const;
private:
    bool MultiClass;
    THashMap<float, int> LabelToClass;
    TVector<float> ClassToLabel;
    int ClassesCount;
    bool Initialized;
};

void PrepareTargetCompressed(const TLabelConverter& labelConverter, TVector<float>* labels);

int GetClassesCount(int classesCount, const TVector<NJson::TJsonValue>& classLabels);
