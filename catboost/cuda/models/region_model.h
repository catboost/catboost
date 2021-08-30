#pragma once

#include "bin_optimized_model.h"
#include <catboost/cuda/data/feature.h>
#include <catboost/cuda/data/binarizations_manager.h>
#include <util/generic/vector.h>

namespace NCatboostCuda {
    struct TRegionStructure {
        TVector<TBinarySplit> Splits;
        TVector<ESplitValue> Directions;

        ui64 GetHash() const {
            return static_cast<ui64>(TVecHash<TBinarySplit>()(Splits));
        }

        ui32 LeavesCount() const {
            return static_cast<ui32>(Splits.size()) + 1;
        }

        bool HasSplit(const TBinarySplit& candidate) {
            for (const auto& split : Splits) {
                if (split == candidate) {
                    return true;
                }
            }
            return false;
        }

        bool operator==(const TRegionStructure& other) const {
            return Splits == other.Splits && Directions == other.Directions;
        }

        bool operator!=(const TRegionStructure& other) const {
            return !(*this == other);
        }

        Y_SAVELOAD_DEFINE(Splits, Directions);
    };

    class TRegionModel: public IBinOptimizedModel {
    public:
        TRegionModel(TRegionStructure&& modelStructure,
                     const TVector<float>& values,
                     const TVector<double>& weights,
                     ui32 dim)
            : ModelStructure(std::move(modelStructure))
            , LeafValues(values)
            , LeafWeights(weights)
            , Dim(dim)
        {
        }

        TRegionModel() = default;

        ~TRegionModel() {
        }

        const TRegionStructure& GetStructure() const {
            return ModelStructure;
        }

        void Rescale(double scale) override final {
            for (ui32 i = 0; i < LeafValues.size(); ++i) {
                LeafValues[i] *= scale;
            }
        }

        void ShiftLeafValues(double shift) override final {
            for (ui32 i = 0; i < LeafValues.size(); ++i) {
                LeafValues[i] += shift;
            }
        }

        void UpdateLeaves(const TVector<float>& newValues) final {
            LeafValues = newValues;
        }

        void UpdateWeights(const TVector<double>& newWeights) final {
            LeafWeights = newWeights;
        }

        const TVector<float>& GetValues() const {
            return LeafValues;
        }

        const TVector<double>& GetWeights() const {
            return LeafWeights;
        }

        void ComputeBins(const TDocParallelDataSet& dataSet,
                         TStripeBuffer<ui32>* dst) const override;

        ui32 OutputDim() const final {
            Y_ASSERT(Dim);
            return Dim;
        }

        ui32 BinCount() const final {
            return ModelStructure.LeavesCount();
        }

        Y_SAVELOAD_DEFINE(ModelStructure, LeafValues, LeafWeights, Dim);

    private:
        TRegionStructure ModelStructure;
        TVector<float> LeafValues;
        TVector<double> LeafWeights;
        ui32 Dim = 0;
    };
}

template <>
struct THash<NCatboostCuda::TRegionStructure> {
    inline size_t operator()(const NCatboostCuda::TRegionStructure& value) const {
        return value.GetHash();
    }
};
