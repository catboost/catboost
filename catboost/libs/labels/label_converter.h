#pragma once

#include <util/generic/hash_set.h>
#include <util/generic/hash.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>

class TLabelConverter {
public:
    TLabelConverter() : Initialized(false) {};

    void Initialize(int approxDimension);
    void Initialize(const TString& multiclassLabelParams);
    void Initialize(const TVector<float>& targets, int classesCount);

    void ValidateLabels(const TVector<float>& labels) const;

    int GetApproxDimension() const;
    int GetClassIdx(float label) const;
    bool IsInitialized() const;

    TString SerializeMulticlassParams(int classesCount, const TVector<TString>& classNames) const;
private:
    THashMap<float, int> LabelToClass;
    TVector<float> ClassToLabel;
    int ClassesCount;
    bool Initialized;
};

void PrepareTargetCompressed(const TLabelConverter& labelConverter, TVector<float>* labels);

THashMap<float, int> CalcLabelToClassMap(TVector<float> targets, int classesCount);

int GetClassesCount(int classesCount, const TVector<TString>& classNames);
