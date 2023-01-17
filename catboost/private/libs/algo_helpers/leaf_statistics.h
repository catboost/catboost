#pragma once

#include <util/generic/array_ref.h>
#include <util/generic/vector.h>

class TLeafStatistics {
private:
    int TargetDimension;
    int ApproxDimension;
    int AllObjectsCount;
    double AllObjectsSumWeight;

    TVector<float> Weights;
    TVector<float> SampleWeights; // used only double for Quantile regression
    TVector<TVector<float>> Labels;
    TVector<TVector<double>> Approx; // [approxDim][sampleIdx]
    TVector<double> LeafValues; // [approxDim]

    int ObjectsCount;
    int LeafIdx;

    TVector<TArrayRef<float>> LabelsView;
    TVector<TConstArrayRef<float>> ConstLabelsView;

public:
    TLeafStatistics() = default;

    TLeafStatistics(int targetDimension, int approxDimension, int allObjectsCount, double sumWeight)
        : TargetDimension(targetDimension)
        , ApproxDimension(approxDimension)
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
        Labels.resize(TargetDimension);
        LabelsView.resize(TargetDimension);
        ConstLabelsView.resize(TargetDimension);
        for (int dimIdx = 0; dimIdx < TargetDimension; ++dimIdx) {
            Labels[dimIdx].yresize(objectsCount);
            LabelsView[dimIdx] = Labels[dimIdx];
            ConstLabelsView[dimIdx] = Labels[dimIdx];
        }

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

    TConstArrayRef<TConstArrayRef<float>> GetLabels() const {
        return ConstLabelsView;
    }

    TArrayRef<TArrayRef<float>> GetLabels() {
        return LabelsView;
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
