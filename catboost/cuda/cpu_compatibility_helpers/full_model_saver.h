#pragma once

#include "final_mean_ctr.h"
#include <catboost/libs/model/model.h>
#include <catboost/libs/data/pool.h>
#include <catboost/libs/algo/online_ctr.h>
#include <catboost/libs/model/formula_evaluator.h>
#include <catboost/libs/algo/index_hash_calcer.h>
namespace NCatboostCuda
{
    class TFileSaver : public TMoveOnly {
    public:
        TFileSaver(const TString& fileName,
                   const ui32 ctrCount,
                   const TCoreModel& coreModel,
                   const TOneHotFeaturesInfo& oneHotInfo)
                : StreamSaver(MakeHolder<TStreamedFullModelSaver>(fileName, ctrCount, coreModel, oneHotInfo)) {
        }

        void SaveOneCtr(const TModelCtrBase& base,
                        TCtrValueTable&& table) {
            StreamSaver->SaveOneCtr(base, table);
        }

    private:
        THolder<TStreamedFullModelSaver> StreamSaver;
    };

    class TFullModelSaver {
    public:
        explicit TFullModelSaver(TFullModel& dst)
               : Model(dst) {

       }

        void SaveOneCtr(const TModelCtrBase& base,
                        TCtrValueTable&& table) {
            CB_ENSURE(Model.CtrCalcerData.LearnCtrs.has(base));
            Model.CtrCalcerData.LearnCtrs[base] = std::move(table);
        }

    private:
        TFullModel& Model;
    };

    class TCoreModelToFullModelConverter {

    public:
        TCoreModelToFullModelConverter(const TCoreModel& coreModel,
                                       const TPool& cpuPool,
                                       NPar::TLocalExecutor& localExecutor)
            : CoreModel(coreModel)
            , Pool(cpuPool)
            , LocalExecutor(localExecutor)
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


        void SaveToFile(const TString& output) {
            TString path = output;
            Save([path](const yvector<TModelCtrBase>& ctrs,
                        const TCoreModel& coreModel,
                        const TOneHotFeaturesInfo& oneHotFeaturesInfo) {
                return TFileSaver(path, ctrs.size(), coreModel, oneHotFeaturesInfo);

            });
        }

        void SaveToModel(TFullModel* dst) {
            CB_ENSURE(dst);
            Save([=](const yvector<TModelCtrBase>& ctrs,
                     const TCoreModel& coreModel,
                     const TOneHotFeaturesInfo& oneHotFeaturesInfo) {
                TCoreModel& coreDst = (*dst);
                TFullModel& fullDst = (*dst);
                coreDst = coreModel;
                fullDst.OneHotFeaturesInfo = oneHotFeaturesInfo;
                for (auto& ctrBase : ctrs) {
                    fullDst.CtrCalcerData.LearnCtrs[ctrBase] = TCtrValueTable();
                }
                return TFullModelSaver(fullDst);
            });
        }
    private:

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
            const int sampleCount = static_cast<const int>(targets.size());

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
            target.resize(pool.Docs.GetDocCount());
            for (ui32 i = 0; i < pool.Docs.GetDocCount(); ++i) {
                target[i] = pool.Docs.Target[i];
            }
            return target;
        }

        template <class TSaverFactory>
        void Save(TSaverFactory&& factory) {
            auto usedCtrs = GetUsedCtrs(CoreModel);
            auto oneHotFeaturesInfo = CreateOneHotFeaturesInfo(CoreModel, Pool);
            auto catFeatures = GetCatFeaturesSet();
            auto targets = ExtractTargetsFromPool(Pool);
            const auto sampleCount = static_cast<ui32>(targets.size());
            auto saver = factory(usedCtrs, CoreModel, oneHotFeaturesInfo);

            if (usedCtrs.empty()) {
                return;
            }

            TAllFeatures allFeatures;
            PrepareAllFeatures(
                    catFeatures,
                    CoreModel.Borders,
                    CoreModel.HasNans,
                    yvector<int>(), //TODO(noxoomo): extract used features from model
                    sampleCount,
                    GetOneHotMaxSize(CoreModel),
                    NanMode,
                    /*allowClearPool*/ false,
                    LocalExecutor,
                    &Pool.Docs,
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
                saver.SaveOneCtr(ctr, std::move(resTable));
            }, 0, usedCtrs.ysize(), NPar::TLocalExecutor::WAIT_COMPLETE);
        }

    private:
        const TCoreModel& CoreModel;
        const TPool& Pool;
        NPar::TLocalExecutor& LocalExecutor;
        ENanMode NanMode = ENanMode::Forbidden;
        ui32 CtrLeafCountLimit = std::numeric_limits<ui32>::max();
        bool StoreAllSimpleCtrsFlag = false;
    };
}
