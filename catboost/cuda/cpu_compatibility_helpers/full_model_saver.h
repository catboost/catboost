#pragma once

#include <catboost/libs/model/model.h>
#include <catboost/libs/data/pool.h>
#include <catboost/libs/algo/online_ctr.h>
#include <catboost/libs/algo/index_hash_calcer.h>

namespace NCatboostCuda {
    class TCoreModelToFullModelConverter {
    public:
        TCoreModelToFullModelConverter(TFullModel&& model,
                                       const TPool& cpuPool,
                                       const TVector<TTargetClassifier>& targetClassifiers,
                                       NPar::TLocalExecutor& localExecutor)
            : ModelBase(std::move(model))
            , Pool(cpuPool)
            , TargetClassifiers(targetClassifiers)
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
            ModelBase.CtrProvider = new TStaticCtrOnFlightSerializationProvider(
                ModelBase.ObliviousTrees.GetUsedModelCtrBases(),
                GetCtrTableGenerator(),
                LocalExecutor);
            {
                TOFStream fileOutput(output);
                ModelBase.Save(&fileOutput);
            }
            ModelBase.CtrProvider.Reset();
        }

        void SaveToModel(TFullModel* dst) {
            CB_ENSURE(dst);
            auto ctrTableGenerator = GetCtrTableGenerator();
            *dst = ModelBase;
            if (dst->HasValidCtrProvider()) {
                // ModelBase apparently has valid ctrs table
                // TODO(kirillovs): add here smart check for ctrprovider serialization ability
                // after implementing non-storing ctr providers
                return;
            }
            dst->CtrProvider = new TStaticCtrProvider;
            auto usedCtrBases = dst->ObliviousTrees.GetUsedModelCtrBases();
            TMutex lock;
            LocalExecutor.ExecRange([&](int i) {
                auto& ctr = usedCtrBases[i];
                auto table = ctrTableGenerator(ctr);
                with_lock (lock) {
                    dst->CtrProvider->AddCtrCalcerData(std::move(table));
                }
            },
                                    0, usedCtrBases.ysize(), NPar::TLocalExecutor::WAIT_COMPLETE);
            dst->UpdateDynamicData();
        }

    private:
        inline void CreateTargetClasses(const TVector<float>& targets,
                                        const TVector<TTargetClassifier>& targetClassifiers,
                                        TVector<TVector<int>>& learnTargetClasses,
                                        TVector<int>& targetClassesCount) {
            ui64 ctrCount = targetClassifiers.size();
            const int sampleCount = static_cast<const int>(targets.size());

            learnTargetClasses.assign(ctrCount, TVector<int>(sampleCount));
            targetClassesCount.resize(ctrCount);

            for (ui32 ctrIdx = 0; ctrIdx < ctrCount; ++ctrIdx) {
                NPar::ParallelFor(0, (ui32)sampleCount, [&](int sample) {
                    learnTargetClasses[ctrIdx][sample] = TargetClassifiers[ctrIdx].GetTargetClass(targets[sample]);
                });

                targetClassesCount[ctrIdx] = TargetClassifiers[ctrIdx].GetClassesCount();
            }
        }

        inline static TVector<float> ExtractTargetsFromPool(const TPool& pool) {
            TVector<float> target;
            target.resize(pool.Docs.GetDocCount());
            for (ui32 i = 0; i < pool.Docs.GetDocCount(); ++i) {
                target[i] = pool.Docs.Target[i];
            }
            return target;
        }

        std::function<TCtrValueTable(const TModelCtrBase& ctr)> GetCtrTableGenerator() {
            auto usedCtrs = ModelBase.ObliviousTrees.GetUsedModelCtrBases();
            auto targets = ExtractTargetsFromPool(Pool);
            const auto sampleCount = static_cast<ui32>(targets.size());

            TVector<TVector<int>> learnTargetClasses;
            TVector<int> targetClassesCount;
            CreateTargetClasses(targets,
                                TargetClassifiers,
                                learnTargetClasses,
                                targetClassesCount);

            return [this, sampleCount, learnTargetClasses, targets, targetClassesCount](const TModelCtrBase& ctr) -> TCtrValueTable {
                TCtrValueTable resTable;
                CalcFinalCtrs(
                    ctr.CtrType,
                    ctr.Projection,
                    Pool,
                    sampleCount,
                    learnTargetClasses[ctr.TargetBorderClassifierIdx],
                    targets,
                    targetClassesCount[ctr.TargetBorderClassifierIdx],
                    CtrLeafCountLimit,
                    StoreAllSimpleCtrsFlag,
                    &resTable);
                resTable.ModelCtrBase = ctr;
                return resTable;
            };
        }

    private:
        TFullModel ModelBase;
        const TPool& Pool;
        const TVector<TTargetClassifier>& TargetClassifiers;
        NPar::TLocalExecutor& LocalExecutor;
        ui32 CtrLeafCountLimit = std::numeric_limits<ui32>::max();
        bool StoreAllSimpleCtrsFlag = false;
    };
}
