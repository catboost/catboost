#pragma once

#include "bin_optimized_model.h"
#include <catboost/cuda/data/feature.h>
#include <catboost/cuda/data/binarizations_manager.h>
#include <util/generic/vector.h>

namespace NCatboostCuda {
    struct TObliviousTreeStructure {
        TVector<TBinarySplit> Splits;

        ui64 GetHash() const {
            return static_cast<ui64>(TVecHash<TBinarySplit>()(Splits));
        }

        ui32 GetDepth() const {
            return static_cast<ui32>(Splits.size());
        }

        ui32 LeavesCount() const {
            return static_cast<ui32>(1 << GetDepth());
        }

        bool HasSplit(const TBinarySplit& candidate) {
            for (const auto& split : Splits) {
                if (split == candidate) {
                    return true;
                }
            }
            return false;
        }

        bool operator==(const TObliviousTreeStructure& other) const {
            return Splits == other.Splits;
        }

        bool operator!=(const TObliviousTreeStructure& other) const {
            return !(*this == other);
        }

        bool operator<(const TObliviousTreeStructure& rhs) const {
            return Splits < rhs.Splits;
        }
        bool operator>(const TObliviousTreeStructure& rhs) const {
            return rhs < *this;
        }
        bool operator<=(const TObliviousTreeStructure& rhs) const {
            return !(rhs < *this);
        }
        bool operator>=(const TObliviousTreeStructure& rhs) const {
            return !(*this < rhs);
        }

        Y_SAVELOAD_DEFINE(Splits);
    };

    class TObliviousTreeModel: public IBinOptimizedModel {
    public:
        TObliviousTreeModel(
            TObliviousTreeStructure&& modelStructure,
            const TVector<float>& values,
            const TVector<double>& weights,
            ui32 dim)
            : ModelStructure(std::move(modelStructure))
            , LeafValues(values)
            , LeafWeights(weights)
            , Dim(dim)
        {
        }

        TObliviousTreeModel(
            const TObliviousTreeStructure& modelStructure,
            const TVector<float>& values,
            ui32 dim)
            : ModelStructure(modelStructure)
            , LeafValues(values)
            , Dim(dim)
        {
        }

        TObliviousTreeModel() = default;

        TObliviousTreeModel(const TObliviousTreeStructure& modelStructure)
            : ModelStructure(modelStructure)
            , LeafValues(modelStructure.LeavesCount())
            , LeafWeights(modelStructure.LeavesCount())
            , Dim(1)
        {
        }

        ~TObliviousTreeModel() {
        }

        const TObliviousTreeStructure& GetStructure() const {
            return ModelStructure;
        }

        inline void Rescale(double scale) override final {
            for (ui32 i = 0; i < LeafValues.size(); ++i) {
                LeafValues[i] *= scale;
            }
        }

        inline void ShiftLeafValues(double shift) override final {
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

        void ComputeBins(
            const TDocParallelDataSet& dataSet,
            TStripeBuffer<ui32>* dst) const override;

        ui32 OutputDim() const final {
            Y_ASSERT(Dim);
            return Dim;
        }

        ui32 BinCount() const final {
            return ModelStructure.LeavesCount();
        }

        TObliviousTreeModel SortedBySplitsModel() const {
            TObliviousTreeStructure sortedStructure = ModelStructure;
            TVector<ui32> order;
            order.resize(sortedStructure.Splits.size());
            Iota(order.begin(), order.end(), 0);

            const ui32 leafCount = BinCount();

            Sort(order.begin(), order.end(), [&](const ui32 i, const ui32 j) -> bool {
                return ModelStructure.Splits[i] < ModelStructure.Splits[j];
            });

            Sort(sortedStructure.Splits.begin(), sortedStructure.Splits.end());

            TVector<float> values(leafCount * Dim);

            for (ui64 leafId = 0; leafId < leafCount; ++leafId) {
                ui32 sourceId = 0;
                for (ui32 bit = 0; bit < ModelStructure.GetDepth(); ++bit) {
                    ui32 bitValue = (leafId >> bit) & 1;
                    const ui32 srcBit = order[bit];
                    sourceId |= bitValue << srcBit;
                }
                for (ui32 dim = 0; dim < Dim; ++dim) {
                    values[leafId * Dim + dim] = LeafValues[sourceId * Dim + dim];
                }
            }
            return TObliviousTreeModel(std::move(sortedStructure),
                                       values,
                                       TVector<double>(),
                                       Dim);
        }

        TMaybe<float> GetL1LeavesSum() const {
            if (LeafValues.empty()) {
                return Nothing();
            }
            const auto numLeaves = LeafValues.size() / Dim;
            double sumOverLeaves = 0;
            for (auto leaf : xrange(numLeaves)) {
                double w2 = 0;
                for (auto dim : xrange(Dim)) {
                    const double leafValue = LeafValues[Dim * leaf + dim];
                    w2 += leafValue * leafValue;
                }
                sumOverLeaves += sqrt(w2);
            }
            return Sqr(sumOverLeaves / numLeaves);
        }

        Y_SAVELOAD_DEFINE(ModelStructure, LeafValues, LeafWeights, Dim);

    private:
        TObliviousTreeStructure ModelStructure;
        TVector<float> LeafValues;
        TVector<double> LeafWeights;
        ui32 Dim = 0;
    };
}

template <>
struct THash<NCatboostCuda::TObliviousTreeStructure> {
    inline size_t operator()(const NCatboostCuda::TObliviousTreeStructure& value) const {
        return value.GetHash();
    }
};
