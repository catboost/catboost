#pragma once

#include <util/generic/array_ref.h>
#include <util/generic/vector.h>

class TLeafStatistics {
private:
    int ApproxDimension;
    int AllObjectsCount;
    double AllObjectsSumWeight;

    TVector<float> Weights;
    TVector<float> SampleWeights; // used only double for Quantile regression
    TVector<float> Labels;
    TVector<TVector<double>> Approx;
    TVector<double> LeafValues;

    int ObjectsCount;
    int LeafIdx;

public:
    TLeafStatistics() = default;

    TLeafStatistics(int approxDimension, int allObjectsCount, double sumWeight)
        : ApproxDimension(approxDimension)
        , AllObjectsCount(allObjectsCount)
        , AllObjectsSumWeight(sumWeight)
        , LeafValues(TVector<double>(approxDimension, 0))
        , ObjectsCount(0)
    {
    }

    void SetLeafIdx(int leafIdx) {
        LeafIdx = leafIdx;
    }

    int GetLeafIdx() const {
        return LeafIdx;
    }

    void Resize(int objectsCount, bool needSampleWeights, bool hasWeights) {
        ObjectsCount = objectsCount;
        Labels.yresize(objectsCount);
        if (!needSampleWeights && hasWeights) {
            Weights.yresize(objectsCount);
        }
        if (needSampleWeights) {
            SampleWeights.yresize(objectsCount);
        }

        Approx.yresize(ApproxDimension);
        for (int dimIdx = 0; dimIdx < ApproxDimension; ++dimIdx) {
            Approx[dimIdx].yresize(objectsCount);
        }
    }

    int GetApproxDimension() const {
        return ApproxDimension;
    }

    int GetLearnObjectsCount() const {
        return AllObjectsCount;
    }

    double GetAllObjectsSumWeight() const {
        return AllObjectsSumWeight;
    }

    int GetObjectsCountInLeaf() const {
        return ObjectsCount;
    }

    TConstArrayRef<float> GetWeights() const {
        return Weights;
    }

    TArrayRef<float> GetWeights() {
        return Weights;
    }

    TConstArrayRef<float> GetSampleWeights() const {
        return SampleWeights;
    }

    TArrayRef<float> GetSampleWeights() {
        return SampleWeights;
    }

    TConstArrayRef<float> GetLabels() const {
        return Labels;
    }

    TArrayRef<float> GetLabels() {
        return Labels;
    }

    TVector<double>* GetLeafValuesRef() {
        return &(LeafValues);
    }

    TConstArrayRef<double> GetLeafValues() const {
        return LeafValues;
    }

    TVector<TVector<double>>* GetApproxRef() {
        return &(Approx);
    }

    TVector<TVector<double>>& GetApprox() {
        return Approx;
    }

    TArrayRef<double> GetApprox(int dim) {
        return Approx[dim];
    }
};
