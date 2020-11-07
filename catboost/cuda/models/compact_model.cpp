#include "compact_model.h"

#include <catboost/libs/helpers/exception.h>
#include <catboost/cuda/models/additive_model.h>

#include <util/stream/labeled.h>
#include <util/generic/array_ref.h>
#include <util/ysaveload.h>

using NCatboostCuda::TLeafPath;

namespace NCatboostCuda {

    static inline bool IsSubset(const TObliviousTreeStructure& structure, const TObliviousTreeStructure& of) {
        return NCB::IsSubset(structure.Splits, of.Splits);
    }

    //stupid O(n^2)
    //but for most models should be sufficient
    //TODO: if'll be bottleneck, stupid optimization: map feature -> structures with this feature, loop through structures with only this feature. for most application should be fast enough
    static TVector<ui64> FindDeepestSupstructure(const TVector<TObliviousTreeStructure>& structures, NPar::ILocalExecutor* executor) {
        TVector<ui64> indicesOfDeepestSupstructure;
        indicesOfDeepestSupstructure.resize(structures.size());

        THashMap<TBinarySplit, THashSet<ui64>> lookupHints;
        for (ui32 i = 0; i < structures.size(); ++i) {
            const auto& structure = structures[i];
            for (const auto& split : structure.Splits) {
                lookupHints[split].insert(i);
            }
        }
        NPar::ParallelFor(*executor, 0, static_cast<ui32>(structures.size()), [&](int i) {
            indicesOfDeepestSupstructure[i] = i;

            TBinarySplit lookUpKey = structures[i].Splits.back();
            ui64 size = structures.size();
            for (const auto& split : structures[i].Splits) {
                if (lookupHints[split].size() < size) {
                    size = lookupHints[split].size();
                    lookUpKey = split;
                }
            }

            for (ui64 j : lookupHints[lookUpKey]) {
                if (IsSubset(structures[i], structures[j])) {
                    if (structures[j].GetDepth() > structures[indicesOfDeepestSupstructure[i]].GetDepth()) {
                        indicesOfDeepestSupstructure[i] = j;
                    }
                }
            }
        });
        return indicesOfDeepestSupstructure;
    }

    //paths are sorted
    static THashMap<TObliviousTreeStructure, TVector<TLeafPath>> MapPathToStructures(const TVector<TLeafPath>& paths, NPar::ILocalExecutor* executor) {
        THashMap<TObliviousTreeStructure, TVector<TLeafPath>> structuresToPath;
        for (const auto& path: paths) {
            Y_ASSERT(path.IsSorted());
            TObliviousTreeStructure structure = {path.Splits};
            structuresToPath[structure].push_back(path);
        }

        TVector<TObliviousTreeStructure> uniqueStructures;
        uniqueStructures.reserve(structuresToPath.size());
        for (const auto& [structure, pathsWithStructure] : structuresToPath) {
            Y_UNUSED(pathsWithStructure);
            uniqueStructures.push_back(structure);
        }

        TVector<ui64> structureToDeepest = FindDeepestSupstructure(uniqueStructures, executor);
        THashMap<TObliviousTreeStructure, TVector<TLeafPath>> result;

        for (ui64 i = 0; i < uniqueStructures.size(); ++i) {
            const TObliviousTreeStructure& srcStructure = uniqueStructures[i];
            const TObliviousTreeStructure& supStructure = uniqueStructures[structureToDeepest[i]];

            for (auto& leaf : structuresToPath[srcStructure]) {
                result[supStructure].push_back(std::move(leaf));
            }
        }
        return result;
    }

    //for each leaf path search for deepest OT structure, for which this path is subset
    static THashMap<TObliviousTreeStructure, TVector<TLeafPath>> MapPathToStructures(const THashMap<TLeafPath, TVector<float>>& leaves,
        NPar::ILocalExecutor* executor) {
        TVector<TLeafPath> pathOnly;
        for (const auto& [path, values] : leaves) {
            Y_UNUSED(values);
            pathOnly.push_back(path);
        }
        return MapPathToStructures(pathOnly, executor);
    }

    class TObliviousTreeBuilder {
    public:
        TObliviousTreeBuilder(const TObliviousTreeStructure& structure, ui32 dim)
        : Structure(structure)
        , Dim(dim)
        , Leaves(dim * structure.LeavesCount()) {

        }

        void AddLeaf(const TLeafPath& leafPath, TConstArrayRef<float> values) {
            Y_ASSERT(NCB::IsSubset(leafPath.Splits, Structure.Splits));

            TVector<ui32> leafFeatures;
            TVector<ESplitValue> splitTypes;
            TVector<ui32> restFeatures;
            {
                ui32 leafCursor = 0;
                for (ui32 i = 0; i < Structure.Splits.size(); ++i) {
                    if (leafCursor < leafPath.Splits.size() &&  leafPath.Splits[leafCursor] == Structure.Splits[i]) {
                        leafFeatures.push_back(i);
                        splitTypes.push_back(leafPath.Directions[leafCursor]);
                        ++leafCursor;
                    } else {
                        restFeatures.push_back(i);
                    }
                }
            }
            Y_ASSERT(leafFeatures.size() + restFeatures.size() == Structure.GetDepth());

            const ui32 leavesToUpdate = 1 << restFeatures.size();

            ui64 baseLeaf = 0;
            for (ui32 i = 0; i < leafFeatures.size(); ++i) {
                if (splitTypes[i] == ESplitValue::One) {
                    baseLeaf |= 1ULL << leafFeatures[i];
                }
            }

            for (ui32 idx = 0; idx < leavesToUpdate; ++idx) {
                ui64 leaf = baseLeaf;
                for (ui32 i = 0; i < restFeatures.size(); ++i) {
                    ui32 srcBit = (idx >> i) & 1;
                    leaf |= srcBit << restFeatures[i];
                }
                for (ui32 dim = 0; dim < Dim; ++dim) {
                    Leaves[leaf * Dim + dim] += values[dim];
                }
            }
        }

        TObliviousTreeModel Build() const {
            return TObliviousTreeModel(Structure, Leaves, Dim);
        }

    private:
        TObliviousTreeStructure Structure;
        ui32 Dim;
        TVector<float> Leaves;

    };

    //leaf paths are sorted by binary splits
    static THashMap<TLeafPath, TVector<float>> ExtractLeaves(const TAdditiveModel<TNonSymmetricTree>& ensemble, NPar::ILocalExecutor* executor) {
        THashMap<TLeafPath, TVector<float>> leaves;
        TAdaptiveLock lock;

        const int blockCount = executor->GetThreadCount() + 1;
        const int blockSize = (ensemble.WeakModels.size() + blockCount - 1) / blockCount;

        NPar::ParallelFor(*executor, 0, blockCount, [&](int blockId) {
            THashMap<TLeafPath, TVector<float>> leavesLocal;
            const ui32 start = blockId * blockSize;
            const ui32 end = Min<ui32>((blockId + 1) * blockSize, ensemble.WeakModels.size());

            auto cmp = [](const TBinarySplit& left, const TBinarySplit& right) -> bool {
                return left < right;
            };

            for (ui32 id = start; id < end; ++id) {
                const auto& tree = ensemble.WeakModels[id];
                tree.VisitLeaves([&](const TLeafPath& leafPath, TConstArrayRef<float> pathValues) {
                    if (!AllOf(pathValues, [&](float x) { return x == 0; })) {
                        auto path = SortUniquePath(leafPath, cmp);

                        auto& dst = leavesLocal[path];
                        if (dst.size() == 0) {
                            dst.resize(pathValues.size());
                        }

                        for (ui64 i = 0; i < pathValues.size(); ++i) {
                            dst[i] += pathValues[i];
                        }
                    }
                });
            }

            with_lock(lock) {
                for (auto& [path, value] : leavesLocal) {
                    auto& dst = leaves[path];
                    if (dst.empty()) {
                        dst = std::move(value);
                    } else {
                        for (ui64 i = 0; i < value.size(); ++i) {
                            dst[i] += value[i];
                        }
                    }
                }
            }
        });
        return leaves;
    }

    template <class K, class V>
    static TVector<K> GetKeys(const THashMap<K, V>& map) {
        TVector<K> result;
        result.reserve(map.size());

        for (const auto& [key, value] : map) {
            Y_UNUSED(value);
            result.push_back(key);
        }
        return result;
    }

    TAdditiveModel<TObliviousTreeModel> MakeOTEnsemble(
        const TAdditiveModel<TNonSymmetricTree>& ensemble,
        NPar::ILocalExecutor* executor
        ) {
        const ui64 outputDim = ensemble.OutputDim();
        THashMap<TLeafPath, TVector<float>> leaves = ExtractLeaves(ensemble, executor);
        CATBOOST_DEBUG_LOG << "Extract #" << leaves.size() << " leaves from ensemble with " << ensemble.Size() << " trees" << Endl;
        //we'll obtain map from leaf path to deepest OT structure candidate
        //we must to include each deepest structure candidate in result ensemble
        //so we could compress just structures from this function
        //all others structures will automatically have tree to assign
        auto leafToOTStructure = MapPathToStructures(leaves, executor);
        TVector<TObliviousTreeStructure> structures = GetKeys(leafToOTStructure);
        CATBOOST_DEBUG_LOG << "Found #" << structures.size() << " unique OT structures" << Endl;


        TVector<TObliviousTreeStructure> resultStructures;

        TVector<TVector<ui32>> resultCover;
        resultStructures = structures;

        resultCover.resize(resultStructures.size());
        for (ui32 i = 0; i < resultCover.size(); ++i) {
            resultCover[i] = {i};
        }

        ui64 totalLeaves = 0;
        for (auto& structure : resultStructures) {
            totalLeaves += structure.LeavesCount();
        }

        ui64 modelSize = sizeof(double) * totalLeaves * outputDim;
        TAdditiveModel<TObliviousTreeModel> result;

        if (modelSize > 1024 * 1024 * 1024) {
            CATBOOST_WARNING_LOG
                << "Can't convert non-symmetric tree to symmetric ones, model size is too big (" << modelSize * 1.0 / 1024 / 1024
                << "), result model.bin will be empty" << Endl;
        } else {
            result.WeakModels.resize(resultStructures.size());

            NPar::ParallelFor(*executor, 0, resultStructures.size(), [&](ui32 i) {
                TObliviousTreeBuilder builder(resultStructures[i], outputDim);
                for (auto coveredIdx : resultCover[i]) {
                    const auto& coveredStructure = structures[coveredIdx];
                    for (const auto& leaf : leafToOTStructure[coveredStructure]) {
                        builder.AddLeaf(leaf, leaves[leaf]);
                    }
                }
                result.WeakModels[i] = builder.Build();
            });
        }
        return result;
    }
}
