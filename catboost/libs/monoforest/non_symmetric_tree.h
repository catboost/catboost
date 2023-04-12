#pragma once

#include "split.h"
#include "leaf_path.h"
#include <catboost/libs/model/hash.h>

#include <util/ysaveload.h>
#include <util/generic/vector.h>
#include <util/generic/maybe.h>
#include <util/generic/array_ref.h>
#include <util/digest/multi.h>

namespace NMonoForest {
    struct TTreeNode {
        ui16 FeatureId = 0;
        ui16 Bin = 0;

        ui16 LeftSubtree = 0;
        ui16 RightSubtree = 0;

        ui64 GetHash() const {
            return MultiHash(FeatureId, Bin, LeftSubtree, RightSubtree);
        }

        bool operator==(const TTreeNode& rhs) const {
            return std::tie(FeatureId, Bin, LeftSubtree, RightSubtree) == std::tie(rhs.FeatureId, rhs.Bin, rhs.LeftSubtree, rhs.RightSubtree);
        }
        bool operator!=(const TTreeNode& rhs) const {
            return !(rhs == *this);
        }

        Y_SAVELOAD_DEFINE(FeatureId, Bin, LeftSubtree, RightSubtree);
    };

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

            bool unwind = false;

            i64 prev = 0;

            while (!nodes.empty()) {
                const auto current = nodes.back();
                const auto& currentNode = Nodes[nodes.back()];
                CB_ENSURE(currentNode.LeftSubtree >= 1 && currentNode.RightSubtree >= 1, "Left and/or right subtrees are missing");

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

    class TNonSymmetricTree {
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

        const TNonSymmetricTreeStructure& GetStructure() const {
            return ModelStructure;
        }

        const TVector<float>& GetValues() const {
            return LeafValues;
        }

        const TVector<double>& GetWeights() const {
            return LeafWeights;
        }

        ui32 OutputDim() const {
            return Dim;
        }

        ui32 BinCount() const {
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
struct THash<NMonoForest::TNonSymmetricTreeStructure> {
    inline size_t operator()(const NMonoForest::TNonSymmetricTreeStructure& value) const {
        return value.GetHash();
    }
};

