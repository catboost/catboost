#pragma once

#include <util/generic/array_ref.h>
#include <util/generic/hash_set.h>
#include <util/generic/hash.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>

class TLabelConverter {
public:
    TLabelConverter() : Initialized(false) {};

    bool operator==(const TLabelConverter& rhs) const;

    void Initialize(int approxDimension);
    void Initialize(const TString& multiclassLabelParams);
    void Initialize(TConstArrayRef<float> targets, int classesCount);

    void ValidateLabels(TConstArrayRef<float> labels) const;

    int GetApproxDimension() const;
    int GetClassIdx(float label) const;
    TVector<float> GetClassLabels() const;
    bool IsInitialized() const;

    TString SerializeMulticlassParams(int classesCount, const TVector<TString>& classNames) const;
private:
    THashMap<float, int> LabelToClass;
    TVector<float> ClassToLabel;
    int ClassesCount;
    bool Initialized;
};

void PrepareTargetCompressed(const TLabelConverter& labelConverter, TVector<float>* labels);

int GetClassesCount(int classesCount, const TVector<TString>& classNames);
