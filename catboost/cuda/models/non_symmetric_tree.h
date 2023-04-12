#pragma once

#include "bin_optimized_model.h"
#include <catboost/cuda/data/feature.h>
#include <catboost/cuda/data/binarizations_manager.h>
#include <catboost/cuda/data/leaf_path.h>
#include <util/generic/vector.h>

namespace NCatboostCuda {
    struct TNonSymmetricTreeStructure {
        const TVector<TTreeNode>& GetNodes() const {
            return Nodes;
        }

        TVector<TTreeNode>& GetNodes() {
            Hash.Clear();
            return Nodes;
        }

        TVector<EBinSplitType>& GetSplitTypes() {
            Hash.Clear();
            return SplitTypes;
        }

        const TVector<EBinSplitType>& GetSplitTypes() const {
            return SplitTypes;
        }

        ui64 GetHash() const {
            if (!Hash) {
                Hash = static_cast<ui64>(MultiHash(TVecHash<TTreeNode>()(Nodes), VecCityHash(SplitTypes)));
            }
            return *Hash;
        }

        ui32 LeavesCount() const {
            return static_cast<ui32>(Nodes.size()) + 1;
        }

        bool operator==(const TNonSymmetricTreeStructure& other) const {
            return Nodes == other.Nodes && SplitTypes == other.SplitTypes;
        }

        bool operator!=(const TNonSymmetricTreeStructure& other) const {
            return !(*this == other);
        }

        template <class TVisitor>
        void VisitBins(TVisitor&& visitor) const {
            VisitBinsImpl(visitor);
        }

        Y_SAVELOAD_DEFINE(Nodes, SplitTypes, Hash);

    private:
        template <class TVisitor>
        void VisitBinsImpl(TVisitor&& visitor) const {
            TVector<i64> nodes;
            nodes.reserve(Nodes.size());
            nodes.push_back(0);

            TLeafPath currentPath;
            i64 binCursor = 0;
            // Single leaf (constant value) tree
            if (Nodes.empty()) {
                visitor(currentPath, binCursor);
                return;
            }

            bool unwind = false;

            i64 prev = 0;

            while (!nodes.empty()) {
                const auto current = nodes.back();
                const auto& currentNode = Nodes[nodes.back()];
                CB_ENSURE(currentNode.LeftSubtree >= 1 && currentNode.RightSubtree >= 1, "Left and/or right subtree is missing");

                if (unwind) {
                    if (current + currentNode.LeftSubtree != prev) {
                        currentPath.Splits.resize(nodes.size());
                        currentPath.Directions.resize(nodes.size());
                        currentPath.Directions.back() = ESplitValue::One;

                        if (currentNode.RightSubtree == 1) {
                            visitor(currentPath, binCursor++);
                        } else {
                            nodes.push_back(current + currentNode.LeftSubtree);
                            unwind = false;
                            continue;
                        }
                    }
                    prev = current;
                    nodes.pop_back();
                } else {
                    currentPath.Splits.push_back(TBinarySplit(currentNode.FeatureId,
                                                              currentNode.Bin,
                                                              SplitTypes[nodes.back()]));
                    currentPath.Directions.push_back(ESplitValue::Zero);

                    if (currentNode.LeftSubtree != 1) {
                        nodes.push_back(nodes.back() + 1);
                    } else {
                        visitor(currentPath, binCursor++);
                        currentPath.Directions.back() = ESplitValue::One;

                        if (currentNode.RightSubtree == 1) {
                            visitor(currentPath, binCursor++);
                            unwind = true;
                            prev = current;
                            nodes.pop_back();
                        } else {
                            nodes.push_back(nodes.back() + 1);
                        }
                    }
                }
            }
        }

    private:
        mutable TMaybe<ui64> Hash;
        TVector<TTreeNode> Nodes;
        TVector<EBinSplitType> SplitTypes; //used only for conversion
    };

    class TNonSymmetricTree: public IBinOptimizedModel {
    public:
        TNonSymmetricTree(TNonSymmetricTreeStructure&& modelStructure,
                          const TVector<float>& values,
                          const TVector<double>& weights,
                          ui32 dim)
            : ModelStructure(std::move(modelStructure))
            , LeafValues(values)
            , LeafWeights(weights)
            , Dim(dim)
        {
        }

        TNonSymmetricTree() = default;

        ~TNonSymmetricTree() {
        }

        const TNonSymmetricTreeStructure& GetStructure() const {
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

        template <class TVisitor>
        void VisitLeaves(TVisitor&& visitor) const {
            Y_ASSERT(Dim);
            ModelStructure.VisitBins([&](const TLeafPath& path, ui32 bin) {
                auto values = TConstArrayRef<float>(LeafValues.data() + bin * Dim, Dim);
                visitor(path, values);
            });
        }

        template <class TVisitor>
        void VisitLeavesAndWeights(TVisitor&& visitor) const {
            Y_ASSERT(Dim);
            ModelStructure.VisitBins([&](const TLeafPath& path, ui32 bin) {
                auto values = TConstArrayRef<float>(LeafValues.data() + bin * Dim, Dim);
                visitor(path, values, LeafWeights[bin]);
            });
        }
    private:
        TNonSymmetricTreeStructure ModelStructure;
        TVector<float> LeafValues;
        TVector<double> LeafWeights;
        ui32 Dim = 0;
    };
}

template <>
struct THash<NCatboostCuda::TNonSymmetricTreeStructure> {
    inline size_t operator()(const NCatboostCuda::TNonSymmetricTreeStructure& value) const {
        return value.GetHash();
    }
};
