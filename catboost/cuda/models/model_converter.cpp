#include "model_converter.h"
#include "region_model.h"
#include "non_symmetric_tree.h"
#include "compact_model.h"
#include <catboost/cuda/data/leaf_path.h>

namespace NCatboostCuda {
    template <>
    THolder<TAdditiveModel<TObliviousTreeModel>> MakeObliviousModel(THolder<TAdditiveModel<TObliviousTreeModel>>&& model, NPar::ILocalExecutor*) {
        return std::move(model);
    }

    struct TBinarySplits {
        TVector<TBinarySplit> Splits;

        ui64 GetHash() const {
            return static_cast<ui64>(TVecHash<TBinarySplit>()(Splits));
        }

        bool operator==(const TRegionStructure& other) const {
            return Splits == other.Splits;
        }

        bool operator!=(const TRegionStructure& other) const {
            return !(*this == other);
        }

        Y_SAVELOAD_DEFINE(Splits);
    };

    struct TNode;

    struct TOneWaySplit {
        THolder<TNode> Left;
        THolder<TNode> Right;
    };

    struct TNode {
        THashMap<TBinarySplit, TOneWaySplit> Splits;
        TVector<double> NodeValues;

        explicit TNode(ui32 dim)
            : NodeValues(dim)
        {
        }
    };

    void AddNode(const TLeafPath& leafPath, size_t position, TVector<double>& values, THolder<TNode>* rootPtr) {
        auto& root = *rootPtr;
        if (!root) {
            root = MakeHolder<TNode>(values.size());
        } else {
            Y_ASSERT(root->NodeValues.size() == values.size());
        }

        Y_ASSERT(position <= leafPath.Splits.size());

        if (position == leafPath.Splits.size()) {
            for (ui64 dim = 0; dim < values.size(); ++dim) {
                root->NodeValues[dim] += values[dim];
            }
        } else {
            TBinarySplit split = leafPath.Splits[position];
            ESplitValue splitValue = leafPath.Directions[position];

            auto& oneWaySplit = root->Splits[split];
            if (splitValue == ESplitValue::Zero) {
                AddNode(leafPath, position + 1, values, &oneWaySplit.Left);
            } else {
                AddNode(leafPath, position + 1, values, &oneWaySplit.Right);
            }
        }
    }

    inline ESplitValue InverseDirection(ESplitValue value) {
        if (value == ESplitValue::Zero) {
            return ESplitValue::One;
        } else {
            return ESplitValue::Zero;
        }
    }

    void AddRegionNode(const TRegionModel& region, THolder<TNode>* rootPtr) {
        TLeafPath leafPath;
        const auto& splits = region.GetStructure().Splits;
        const auto& directions = region.GetStructure().Directions;
        const auto& values = region.GetValues();

        const auto outputDim = region.OutputDim();
        TVector<double> outputValues(outputDim);

        for (ui64 i = 0; i < splits.size(); ++i) {
            leafPath.Splits.push_back(splits[i]);
            leafPath.Directions.push_back(InverseDirection(directions[i]));
            for (ui64 dim = 0; dim < outputDim; ++dim) {
                outputValues[dim] = values[i * outputDim + dim];
            }
            AddNode(leafPath, 0, outputValues, rootPtr);
            leafPath.Directions.back() = InverseDirection(leafPath.Directions.back());
        }

        for (ui64 dim = 0; dim < outputDim; ++dim) {
            outputValues[dim] = values[splits.size() * outputDim + dim];
        }
        AddNode(leafPath, 0, outputValues, rootPtr);
    }

    template <class T, class T2>
    inline void CopyCastVector(const TVector<T>& source, TVector<T2>* output) {
        output->resize(source.size());
        for (ui64 i = 0; i < source.size(); ++i) {
            (*output)[i] = source[i];
        }
    }

    struct TBranchState {
        ui32 BinIdx = 0;
        TObliviousTreeStructure Structure;
    };

    enum class EBuildType {
        Subtrees,
        Ensemble
    };
    template <EBuildType Type>
    void BuildObliviousEnsemble(const TNode& root,
                                const TBranchState& state,
                                THashMap<TObliviousTreeStructure, TObliviousTreeStructure>* knownSubtrees,
                                THashMap<TObliviousTreeStructure, TVector<double>>* ensemble) {
        for (const auto& split : root.Splits) {
            TBranchState nextState = state;
            nextState.Structure.Splits.push_back(split.first);

            if (split.second.Left) {
                BuildObliviousEnsemble<Type>(*split.second.Left, nextState, knownSubtrees, ensemble);
            }
            if (split.second.Right) {
                nextState.BinIdx |= (1 << (nextState.Structure.GetDepth() - 1));
                BuildObliviousEnsemble<Type>(*split.second.Right, nextState, knownSubtrees, ensemble);
            }
        }

        if (Type == EBuildType::Subtrees) {
            if (root.Splits.size() == 0) {
                auto prefixStructure = state.Structure;
                if (!knownSubtrees->contains(prefixStructure) || (*knownSubtrees)[prefixStructure].GetDepth() < prefixStructure.GetDepth()) {
                    for (i64 prefixSize = state.Structure.Splits.size(); prefixSize > 0; --prefixSize) {
                        prefixStructure.Splits.resize(prefixSize);
                        (*knownSubtrees)[prefixStructure] = state.Structure;
                    }
                }
            }
            return;
        }

        const auto outputDim = root.NodeValues.size();
        bool needUpdateEnsemble = false;

        for (ui64 dim = 0; dim < outputDim; ++dim) {
            //TOOD(noxoomo): merge prefix to most deep path
            if (Abs(root.NodeValues[dim]) > 1e-20) {
                needUpdateEnsemble = true;
                break;
            }
        }

        if (!needUpdateEnsemble) {
            return;
        }

        TObliviousTreeStructure dstStructure;
        if (knownSubtrees->contains(state.Structure)) {
            dstStructure = (*knownSubtrees)[state.Structure];
        } else {
            dstStructure = state.Structure;
        }

        const ui32 scatterBitsOffset = static_cast<const ui32>(state.Structure.Splits.size());
        const ui32 scatterBitsCount = static_cast<const ui32>(dstStructure.Splits.size() - state.Structure.Splits.size());

        for (ui64 dim = 0; dim < outputDim; ++dim) {
            //TOOD(noxoomo): merge prefix to most deep path
            if (Abs(root.NodeValues[dim])) {
                auto& values = (*ensemble)[dstStructure];
                values.resize(dstStructure.LeavesCount() * outputDim);

                const ui32 scatterBins = 1u << scatterBitsCount;
                for (ui32 upperBin = 0; upperBin < scatterBins; ++upperBin) {
                    const ui32 bin = state.BinIdx | (upperBin << scatterBitsOffset);
                    values[bin * outputDim + dim] += root.NodeValues[dim];
                }
            }
        }
    }

    THashMap<TBinarySplit, double> BuildFeatureUsageCounts(const TAdditiveModel<TRegionModel>& ensemble) {
        THashMap<TBinarySplit, double> result;
        for (const auto& model : ensemble.WeakModels) {
            for (auto& split : model.GetStructure().Splits) {
                result[split]++;
            }
        }
        return result;
    }

    template <>
    THolder<TAdditiveModel<TObliviousTreeModel>> MakeObliviousModel(THolder<TAdditiveModel<TRegionModel>>&& model, NPar::ILocalExecutor*) {
        auto approxDim = model->OutputDim();

        THolder<TNode> root = MakeHolder<TNode>(approxDim);
        for (const auto& region : model->WeakModels) {
            AddRegionNode(region, &root);
        }

        CATBOOST_DEBUG_LOG << "Single decision tree from regions was built" << Endl;

        TBranchState rootState;

        THashMap<TObliviousTreeStructure, TObliviousTreeStructure> knownStructureCache;
        THashMap<TObliviousTreeStructure, TVector<double>> otEnsemble;

        BuildObliviousEnsemble<EBuildType::Subtrees>(*root,
                                                     rootState,
                                                     &knownStructureCache,
                                                     &otEnsemble);

        BuildObliviousEnsemble<EBuildType::Ensemble>(*root,
                                                     rootState,
                                                     &knownStructureCache,
                                                     &otEnsemble);
        CATBOOST_DEBUG_LOG << "Build ot ensemble with " << otEnsemble.size() << " trees from  #" << model->Size() << " regions" << Endl;

        auto result = MakeHolder<TAdditiveModel<TObliviousTreeModel>>();
        for (const auto& otModel : otEnsemble) {
            TObliviousTreeStructure structure = otModel.first;
            TVector<float> values(otModel.second.size());
            TVector<double> weights(otModel.second.size());
            CopyCastVector(otModel.second, &values);

            TObliviousTreeModel tree(std::move(structure),
                                     values,
                                     weights,
                                     approxDim);

            result->AddWeakModel(std::move(tree));
        }
        return result;
    }

    template <>
    THolder<TAdditiveModel<TObliviousTreeModel>> MakeObliviousModel(THolder<TAdditiveModel<TNonSymmetricTree>>&& model, NPar::ILocalExecutor* executor) {
        THolder<TAdditiveModel<TObliviousTreeModel>> result = MakeHolder<TAdditiveModel<TObliviousTreeModel>>();
        (*result) = MakeOTEnsemble(*model, executor);
        return result;
    }
}
