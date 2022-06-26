#include "model_builder.h"
#include <catboost/cuda/models/oblivious_model.h>
#include <catboost/cuda/models/region_model.h>
#include <catboost/cuda/models/non_symmetric_tree.h>

#include <catboost/libs/helpers/exception.h>

#include <util/stream/labeled.h>
#include <util/generic/array_ref.h>
#include <util/generic/maybe.h>

using NCatboostCuda::TLeafPath;

static void ValidateParameters(
    const TConstArrayRef<TLeafPath> leaves,
    const TConstArrayRef<double> leafWeights,
    const TConstArrayRef<TVector<float>> leafValues) {
    CB_ENSURE(!leaves.empty(), "Error: empty tree");

    const auto depth = leaves.front().Splits.size();
    const auto expectedLeavesCount = size_t(1) << depth;
    CB_ENSURE(leaves.size() == expectedLeavesCount, LabeledOutput(leaves.size(), expectedLeavesCount));

    CB_ENSURE(leaves.size() == leafValues.size(), LabeledOutput(leaves.size(), leafValues.size()));
    CB_ENSURE(leaves.size() == leafWeights.size(), LabeledOutput(leaves.size(), leafWeights.size()));

    for (size_t i = 1; i < leaves.size(); ++i) {
        CB_ENSURE(leaves[i].Splits == leaves.front().Splits, LabeledOutput(i));
    }

    for (size_t i = 1; i < leafValues.size(); ++i) {
        CB_ENSURE(leafValues[i].size() == leafValues.front().size(), LabeledOutput(i));
    }
}

namespace NCatboostCuda {
    template <>
    TObliviousTreeModel BuildTreeLikeModel<TObliviousTreeModel>(const TVector<TLeafPath>& leaves,
                                                                const TVector<double>& leafWeights,
                                                                const TVector<TVector<float>>& leafValues) {
        ValidateParameters(leaves, leafWeights, leafValues);

        const auto depth = leaves.front().Splits.size();
        const auto leavesCount = size_t(1) << depth;
        const auto outputDimention = leafValues.front().size();

        TVector<ui32> binIds(leavesCount);

        ui32 checkSum = 0;
        for (size_t i = 0; i < leavesCount; ++i) {
            ui32 bin = 0;
            for (size_t level = 0; level < depth; ++level) {
                const auto direction = leaves[i].Directions[level];
                bin |= ((direction == ESplitValue::Zero) ? 0 : 1) << level;
            }

            CB_ENSURE(bin < leavesCount, "Bin id is too large");
            binIds[i] = bin;
            checkSum += bin;
        }

        const auto expectedCheckSum = leavesCount * (leavesCount - 1) / 2;
        CB_ENSURE(checkSum == expectedCheckSum, LabeledOutput(checkSum, expectedCheckSum, leavesCount));

        TVector<double> resultWeights(leavesCount);
        TVector<float> resultValues(outputDimention * leavesCount);

        for (size_t i = 0; i < leavesCount; ++i) {
            ui32 bin = binIds[i];
            resultWeights[bin] = leafWeights[i];
            for (size_t dim = 0; dim < outputDimention; ++dim) {
                resultValues[bin * outputDimention + dim] = leafValues[i][dim];
            }
        }

        TObliviousTreeStructure structure;
        structure.Splits = leaves[0].Splits;

        return TObliviousTreeModel(
            std::move(structure),
            std::move(resultValues),
            std::move(resultWeights),
            outputDimention);
    }

    inline bool IsPrefix(const TLeafPath& path, const TLeafPath& of) {
        if (path.Directions.size() > of.Directions.size()) {
            return false;
        }
        for (ui32 i = 0; i < path.Directions.size(); ++i) {
            if (path.Splits[i] != of.Splits[i]) {
                return false;
            }
        }
        return true;
    }

    template <>
    TRegionModel BuildTreeLikeModel<TRegionModel>(const TVector<TLeafPath>& leaves,
                                                  const TVector<double>& leafWeights,
                                                  const TVector<TVector<float>>& leafValues) {
        CB_ENSURE(leaves.size(), "Error: empty region");
        CB_ENSURE(leaves.size() == leafValues.size());
        CB_ENSURE(leaves.size() == leafWeights.size());

        std::vector<ui32> sortedByDepthIds(leaves.size());
        Iota(sortedByDepthIds.begin(), sortedByDepthIds.end(), 0);
        Sort(sortedByDepthIds.begin(), sortedByDepthIds.end(), [&](const ui32 left, const ui32 right) -> bool {
            return leaves[left].GetDepth() < leaves[right].GetDepth() || (leaves[left].GetDepth() == leaves[right].GetDepth() && leaves[left].Directions.back() < leaves[right].Directions.back());
        });

        TLeafPath regionPath = leaves[sortedByDepthIds.back()];

        for (ui32 i = 0; i < leaves.size(); ++i) {
            const auto& currentDepthPath = leaves[sortedByDepthIds[i]];
            CB_ENSURE(IsPrefix(currentDepthPath, regionPath), "This leaves structure is not region, can't build model");
        }

        const ui32 outputDim = leafValues[0].size();
        TRegionStructure structure;
        structure.Splits = regionPath.Splits;
        structure.Directions = regionPath.Directions;
        const ui32 depth = regionPath.GetDepth();
        CB_ENSURE(leaves.size() == depth + 1);

        TVector<double> regionWeights(leaves.size());
        TVector<float> regionValues(leaves.size() * outputDim);
        for (ui32 i = 0; i < leaves.size(); ++i) {
            const ui32 srcIdx = sortedByDepthIds[i];
            regionWeights[i] = leafWeights[srcIdx];

            for (ui32 dim = 0; dim < outputDim; ++dim) {
                regionValues[i * outputDim + dim] = leafValues[srcIdx][dim];
            }
        }
        return TRegionModel(std::move(structure), regionValues, regionWeights, outputDim);
    }

    class TFlatTreeBuilder {
    public:
        enum class EDuplicateTerminalLeavesPolicy {
            Combine,
            Exception
        };

        explicit TFlatTreeBuilder(EDuplicateTerminalLeavesPolicy policy)
            : Policy(policy)
        {
        }

        struct TLeaf {
            double Weight;
            TVector<float> Values;

            TLeaf(double weight, const TVector<float>& values)
                : Weight(weight)
                , Values(values)
            {
            }
        };

        struct TNode {
            TSimpleSharedPtr<TNode> Left;
            TSimpleSharedPtr<TNode> Right;

            explicit TNode(TBinarySplit split)
                : Value(split)
            {
            }

            TNode(double weight, const TVector<float>& values)
                : Value(TLeaf(weight, values))
            {
            }

            TLeaf& GetLeaf() {
                return std::get<TLeaf>(Value);
            }

            TBinarySplit& GetSplit() {
                return std::get<TBinarySplit>(Value);
            }

            bool IsTerminal() const {
                return std::holds_alternative<TLeaf>(Value);
            };

            std::variant<TLeaf, TBinarySplit> Value;
        };

        using TNodePtr = TSimpleSharedPtr<TNode>;

        void Add(const TLeafPath& path, const TVector<float>& values, double weight) {
            TNodePtr* cursor = &Root;

            for (ui64 i = 0; i < path.Splits.size(); ++i) {
                TBinarySplit split = path.Splits[i];

                if ((*cursor) == nullptr) {
                    (*cursor) = new TNode(split);
                } else {
                    CB_ENSURE(!(*cursor)->IsTerminal() && (*cursor)->GetSplit() == split, "Error: path is not from current tree.");
                }

                ESplitValue direction = path.Directions[i];
                switch (direction) {
                    case ESplitValue::Zero: {
                        cursor = &(*cursor)->Left;
                        break;
                    }
                    case ESplitValue::One: {
                        cursor = &(*cursor)->Right;
                        break;
                    }
                }
            }

            if (*cursor) {
                CB_ENSURE((*cursor)->IsTerminal());
                if (Policy == EDuplicateTerminalLeavesPolicy::Exception) {
                    ythrow TCatBoostException() << "Can't add terminal leaf twice";
                } else {
                    CB_ENSURE(Policy == EDuplicateTerminalLeavesPolicy::Combine);
                    auto& leaf = (*cursor)->GetLeaf();
                    CB_ENSURE(leaf.Values.size() == values.size());

                    leaf.Weight += weight;
                    for (ui64 i = 0; i < values.size(); ++i) {
                        leaf.Values[i] += values[i];
                    }
                }
            } else {
                (*cursor) = new TNode(weight, values);
            }
        }

        void BuildFlat(TVector<TTreeNode>* nodes,
                       TVector<EBinSplitType>* splitTypes,
                       TVector<float>* leavesValues,
                       TVector<double>* weights) {
            nodes->clear();
            splitTypes->clear();
            leavesValues->clear();
            weights->clear();
            Visit(Root, nodes, splitTypes, leavesValues, weights);
        }

    private:
        ui64 Visit(TNodePtr cursor,
                   TVector<TTreeNode>* flatNodes,
                   TVector<EBinSplitType>* flatSplitTypes,
                   TVector<float>* leavesValues,
                   TVector<double>* weights) {
            CB_ENSURE(cursor, "Tree is empty (cursor is nullptr)");
            const bool isTerminal = cursor->IsTerminal();
            if (isTerminal) {
                const auto& leaf = cursor->GetLeaf();
                for (const auto& val : leaf.Values) {
                    leavesValues->push_back(val);
                }
                weights->push_back(leaf.Weight);
                return 1;
            } else {
                const auto& split = cursor->GetSplit();
                TTreeNode node;
                node.FeatureId = split.FeatureId;
                node.Bin = split.BinIdx;
                flatNodes->push_back(node);
                flatSplitTypes->push_back(split.SplitType);
                ui64 idx = flatNodes->size() - 1;

                const ui64 leftSubtree = Visit(cursor->Left, flatNodes, flatSplitTypes, leavesValues, weights);
                const ui64 rightSubtree = Visit(cursor->Right, flatNodes, flatSplitTypes, leavesValues, weights);

                (*flatNodes)[idx].LeftSubtree = leftSubtree;
                (*flatNodes)[idx].RightSubtree = rightSubtree;
                return leftSubtree + rightSubtree;
            }
        }

    private:
        TSimpleSharedPtr<TNode> Root;
        EDuplicateTerminalLeavesPolicy Policy;
    };

    template <>
    TNonSymmetricTree BuildTreeLikeModel<TNonSymmetricTree>(const TVector<TLeafPath>& leaves,
                                                            const TVector<double>& leavesWeight,
                                                            const TVector<TVector<float>>& leavesValues) {
        CB_ENSURE(leaves.size(), "Error: empty region");
        CB_ENSURE(leaves.size() == leavesValues.size());
        CB_ENSURE(leaves.size() == leavesWeight.size());

        TFlatTreeBuilder treeBuilder(TFlatTreeBuilder::EDuplicateTerminalLeavesPolicy::Exception);
        for (ui64 leaf = 0; leaf < leaves.size(); ++leaf) {
            treeBuilder.Add(leaves[leaf], leavesValues[leaf], leavesWeight[leaf]);
        }

        TNonSymmetricTreeStructure structure;
        TVector<double> weights;
        TVector<float> values;
        treeBuilder.BuildFlat(&structure.GetNodes(), &structure.GetSplitTypes(), &values, &weights);
        const ui32 outputDim = leavesValues[0].size();
        return TNonSymmetricTree(std::move(structure), values, weights, outputDim);
    }
}
