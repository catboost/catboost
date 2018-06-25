#pragma once

#include <catboost/libs/model/model.h>
#include <catboost/libs/data/pool.h>
#include <catboost/libs/model/target_classifier.h>

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

        void SaveToFile(EFinalCtrComputationMode finalCtrComputationMode, const TString& output);
        void SaveToModel(EFinalCtrComputationMode finalCtrComputationMode, TFullModel* dst);

    private:
        void CreateTargetClasses(const TVector<float>& targets,
                                        const TVector<TTargetClassifier>& targetClassifiers,
                                        TVector<TVector<int>>& learnTargetClasses,
                                        TVector<int>& targetClassesCount);

        static TVector<float> ExtractTargetsFromPool(const TPool& pool);

        std::function<TCtrValueTable(const TModelCtrBase& ctr)> GetCtrTableGenerator();
    private:
        TFullModel ModelBase;
        const TPool& Pool;
        const TVector<TTargetClassifier>& TargetClassifiers;
        NPar::TLocalExecutor& LocalExecutor;
        ui32 CtrLeafCountLimit = std::numeric_limits<ui32>::max();
        bool StoreAllSimpleCtrsFlag = false;
    };
}
