#pragma once

#include "final_mean_ctr.h"
#include <catboost/libs/model/model.h>
#include <catboost/libs/data/pool.h>
#include <catboost/libs/algo/online_ctr.h>
#include <catboost/libs/model/formula_evaluator.h>
#include <catboost/libs/algo/index_hash_calcer.h>

//TODO(noxoomo): correct decomposition of logic and move from app to proper place
class TCoreModelToFullModelConverter {
public:
    TCoreModelToFullModelConverter(const TCoreModel& coreModel,
                                   const TPool& cpuPool)
        : CoreModel(coreModel)
        , Pool(cpuPool)
        , LocalExecutor(NPar::LocalExecutor())
    {
    }

    TCoreModelToFullModelConverter& SetCtrLeafCount(ui32 count) {
        CtrLeafCountLimit = count;
        return *this;
    }

    TCoreModelToFullModelConverter& SetStoreAllSimpleCtrsFlag(bool flag) {
        StoreAllSimpleCtrsFlag = flag;
        return *this;
    }

    void Save(const TString& outputPath) {
        auto usedCtrs = GetUsedCtrs(CoreModel);
        auto oneHotFeaturesInfo = CreateOneHotFeaturesInfo(CoreModel, Pool);
        auto catFeatures = GetCatFeaturesSet();
        auto targets = ExtractTargetsFromPool(Pool);
        const auto sampleCount = static_cast<ui32>(targets.size());

        TStreamedFullModelSaver saver(outputPath,
                                      usedCtrs.size(),
                                      CoreModel,
                                      oneHotFeaturesInfo);
        if (usedCtrs.empty()) {
            return;
        }
        TAllFeatures allFeatures;

        PrepareAllFeatures(
            Pool.Docs,
            catFeatures,
            CoreModel.Borders,
            CoreModel.HasNans,
            yvector<int>(), //TODO(noxoomo): extract used features from model
            sampleCount,
            GetOneHotMaxSize(CoreModel),
            NanMode,
            LocalExecutor,
            &allFeatures);

        yvector<int> directPermutation(targets.size());
        std::iota(directPermutation.begin(), directPermutation.end(), 0);

        yvector<yvector<int>> learnTargetClasses;
        yvector<int> targetClassesCount;
        CreateTargetClasses(targets,
                            CoreModel.TargetClassifiers,
                            learnTargetClasses,
                            targetClassesCount);

        LocalExecutor.ExecRange([&](int i) {
            auto& ctr = usedCtrs[i];
            TCtrValueTable resTable;
            if (ctr.CtrType == ECtrType::FloatTargetMeanValue) {
                CalcTargetMeanFinalCtrs(ctr,
                                        allFeatures,
                                        sampleCount,
                                        targets,
                                        directPermutation,
                                        &resTable);

            } else {
                CalcFinalCtrs(ctr,
                              allFeatures,
                              sampleCount,
                              directPermutation,
                              learnTargetClasses[ctr.TargetBorderClassifierIdx],
                              targetClassesCount[ctr.TargetBorderClassifierIdx],
                              CtrLeafCountLimit,
                              StoreAllSimpleCtrsFlag,
                              &resTable);
            }
            saver.SaveOneCtr(ctr, resTable);
        },
                                0, usedCtrs.ysize(), NPar::TLocalExecutor::WAIT_COMPLETE);
    }

    inline static TOneHotFeaturesInfo CreateOneHotFeaturesInfo(const TCoreModel& coreModel,
                                                               const TPool& pool) {
        TOneHotFeaturesInfo oneHotFeaturesInfo;
        for (const auto& tree : coreModel.TreeStruct) {
            for (const auto& split : tree.SelectedSplits) {
                if (split.Type != ESplitType::OneHotFeature) {
                    continue;
                }
                oneHotFeaturesInfo.FeatureHashToOrigString[split.OneHotFeature.Value] = pool.CatFeaturesHashToString.at(split.OneHotFeature.Value);
            }
        }
        return oneHotFeaturesInfo;
    }

    yhash_set<int> GetCatFeaturesSet() {
        return yhash_set<int>(Pool.CatFeatures.begin(), Pool.CatFeatures.end());
    }

    inline void CreateTargetClasses(const yvector<float>& targets,
                                    const yvector<TTargetClassifier>& targetClassifiers,
                                    yvector<yvector<int>>& learnTargetClasses,
                                    yvector<int>& targetClassesCount) {
        ui64 ctrCount = targetClassifiers.size();
        const int sampleCount = targets.size();

        learnTargetClasses.assign(ctrCount, yvector<int>(sampleCount));
        targetClassesCount.resize(ctrCount);

        for (ui32 ctrIdx = 0; ctrIdx < ctrCount; ++ctrIdx) {
            NPar::ParallelFor(0, (ui32)sampleCount, [&](int sample) {
                learnTargetClasses[ctrIdx][sample] = targetClassifiers[ctrIdx].GetTargetClass(targets[sample]);
            });

            targetClassesCount[ctrIdx] = targetClassifiers[ctrIdx].GetClassesCount();
        }
    }

    inline static yvector<TModelCtrBase> GetUsedCtrs(const TCoreModel& coreModel) {
        yvector<TModelCtrBase> usedCtrs;
        yhash_set<TModelCtrBase> ctrsSet;
        for (const auto& bestTree : coreModel.TreeStruct) {
            for (const auto& split : bestTree.SelectedSplits) {
                if (split.Type != ESplitType::OnlineCtr) {
                    continue;
                }
                ctrsSet.insert(split.OnlineCtr.Ctr);
            }
        }
        usedCtrs.assign(ctrsSet.begin(), ctrsSet.end());
        return usedCtrs;
    }

    inline static ui32 GetOneHotMaxSize(const TCoreModel& coreModel) {
        ui32 max = 0;
        for (const auto& bestTree : coreModel.TreeStruct) {
            for (const auto& split : bestTree.SelectedSplits) {
                if (split.Type == ESplitType::OneHotFeature) {
                    max = std::max<ui32>(static_cast<ui32>(split.OneHotFeature.Value), max);
                }
            }
        }
        return max;
    }

    inline static yvector<float> ExtractTargetsFromPool(const TPool& pool) {
        yvector<float> target;
        target.resize(pool.Docs.size());
        for (ui32 i = 0; i < pool.Docs.size(); ++i) {
            target[i] = pool.Docs[i].Target;
        }
        return target;
    }

private:
    const TCoreModel& CoreModel;
    const TPool& Pool;
    NPar::TLocalExecutor& LocalExecutor;
    ENanMode NanMode = ENanMode::Forbidden;
    ui32 CtrLeafCountLimit = std::numeric_limits<ui32>::max();
    bool StoreAllSimpleCtrsFlag = false;
};
