#pragma once

#include "split.h"

#include <catboost/libs/model/hash.h>

#include <util/ysaveload.h>
#include <util/generic/vector.h>

namespace NMonoForest {
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

    class TObliviousTree {
    public:
        TObliviousTree() = default;

        TObliviousTree(
            TObliviousTreeStructure&& treeStructure,
            TVector<double>&& values,
            TVector<double>&& weights,
            ui32 dim)
            : TreeStructure(std::move(treeStructure))
            , LeafValues(std::move(values))
            , LeafWeights(std::move(weights))
            , Dim(dim)
        {
        }

        TObliviousTree(
            TObliviousTreeStructure&& treeStructure,
            TVector<double>&& values,
            ui32 dim)
            : TreeStructure(std::move(treeStructure))
            , LeafValues(std::move(values))
            , Dim(dim)
        {
        }

        explicit TObliviousTree(const TObliviousTreeStructure& treeStructure)
            : TreeStructure(treeStructure)
            , LeafValues(treeStructure.LeavesCount())
            , LeafWeights(treeStructure.LeavesCount())
            , Dim(1)
        {
        }

        ~TObliviousTree() = default;

        const TObliviousTreeStructure& GetStructure() const {
            return TreeStructure;
        }

        const TVector<double>& GetValues() const {
            return LeafValues;
        }

        const TVector<double>& GetWeights() const {
            return LeafWeights;
        }

        ui32 OutputDim() const {
            return Dim;
        }

        ui32 LeavesCount() const {
            return 1 << TreeStructure.GetDepth();
        }

        Y_SAVELOAD_DEFINE(TreeStructure, LeafValues, LeafWeights, Dim);

    private:
        TObliviousTreeStructure TreeStructure;
        TVector<double> LeafValues;
        TVector<double> LeafWeights;
        ui32 Dim = 0;
    };
}

template <>
struct THash<NMonoForest::TObliviousTreeStructure> {
    inline size_t operator()(const NMonoForest::TObliviousTreeStructure& value) const {
        return value.GetHash();
    }
};
