#include "full_model_saver.h"
#include <catboost/libs/algo/online_ctr.h>
#include <catboost/libs/algo/index_hash_calcer.h>

void NCatboostCuda::TCoreModelToFullModelConverter::SaveToModel(EFinalCtrComputationMode finalCtrComputationMode,
                                                                TFullModel* dst) {
    if (finalCtrComputationMode == EFinalCtrComputationMode::Skip) {
        return;
    }
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

void NCatboostCuda::TCoreModelToFullModelConverter::CreateTargetClasses(const TVector<float>& targets,
                                                                        const TVector<TTargetClassifier>& targetClassifiers,
                                                                        TVector<TVector<int>>& learnTargetClasses,
                                                                        TVector<int>& targetClassesCount)  {
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

void NCatboostCuda::TCoreModelToFullModelConverter::SaveToFile(EFinalCtrComputationMode finalCtrComputationMode,
                                                               const TString& output)  {
    if (finalCtrComputationMode == EFinalCtrComputationMode::Default) {
        ModelBase.CtrProvider = new TStaticCtrOnFlightSerializationProvider(
                ModelBase.ObliviousTrees.GetUsedModelCtrBases(),
                GetCtrTableGenerator(),
                LocalExecutor);
    }
    TOFStream fileOutput(output);
    ModelBase.Save(&fileOutput);
    ModelBase.CtrProvider.Reset();
}

TVector<float> NCatboostCuda::TCoreModelToFullModelConverter::ExtractTargetsFromPool(const TPool& pool) {
    TVector<float> target;
    target.resize(pool.Docs.GetDocCount());
    for (ui32 i = 0; i < pool.Docs.GetDocCount(); ++i) {
        target[i] = pool.Docs.Target[i];
    }
    return target;
}

std::function<TCtrValueTable(const TModelCtrBase& ctr)> NCatboostCuda::TCoreModelToFullModelConverter::GetCtrTableGenerator()  {
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


