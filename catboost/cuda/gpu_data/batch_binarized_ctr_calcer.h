#pragma once

#include "ctr_helper.h"
#include "feature_parallel_dataset.h"

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_util/sort.h>
#include <catboost/cuda/data/feature.h>
#include <catboost/cuda/cuda_lib/device_subtasks_helper.h>

#include <catboost/private/libs/ctr_description/ctr_config.h>

namespace NCatboostCuda {
    /*
     * Warning: this class doesn't guarantee optimal performance
     * Preprocessing stage is not critical and we have some GPU/CPU-memory tradeoff with some possible duplicate copies
     * during featureTensor index construction
     */
    class TBatchedBinarizedCtrsCalcer {
    public:
        struct TBinarizedCtr {
            ui32 BinCount = 0;
            TVector<ui8> BinarizedCtr;
        };

        template <class TUi32>
        TBatchedBinarizedCtrsCalcer(TBinarizedFeaturesManager& featuresManager,
                                    const TCtrTargets<NCudaLib::TMirrorMapping>& ctrTargets,
                                    const NCB::TTrainingDataProvider& dataProvider,
                                    const TMirrorBuffer<TUi32>& ctrPermutation,
                                    const NCB::TTrainingDataProvider* linkedTest,
                                    const TMirrorBuffer<TUi32>* testIndices,
                                    NPar::ILocalExecutor* localExecutor)
            : FeaturesManager(featuresManager)
            , CtrTargets(ctrTargets)
            , DataProvider(dataProvider)
            , LearnIndices(ctrPermutation.ConstCopyView())
            , LinkedTest(linkedTest)
            , LocalExecutor(localExecutor)
        {
            if (LinkedTest) {
                CB_ENSURE(testIndices);
                TestIndices = testIndices->ConstCopyView();
            }
        }

        void ComputeBinarizedCtrs(const TVector<ui32>& ctrs,
                                  TVector<TBinarizedCtr>* learnCtrs,
                                  TVector<TBinarizedCtr>* testCtrs);

    private:
        TVector<TVector<NCB::TCtrConfig>> CreateGrouppedConfigs(const TVector<ui32>& ctrIds);

        TCtrBinBuilder<NCudaLib::TSingleMapping> BuildFeatureTensorBins(const TFeatureTensor& tensor,
                                                                        int devId);

        TSingleBuffer<ui64> BuildCompressedBins(const NCB::TTrainingDataProvider& dataProvider,
                                                ui32 featureManagerFeatureId,
                                                ui32 devId);

    private:
        TBinarizedFeaturesManager& FeaturesManager;
        const TCtrTargets<NCudaLib::TMirrorMapping>& CtrTargets;
        const NCB::TTrainingDataProvider& DataProvider;
        TMirrorBuffer<const ui32> LearnIndices;

        const NCB::TTrainingDataProvider* LinkedTest = nullptr;
        TMirrorBuffer<const ui32> TestIndices;

        NPar::ILocalExecutor* LocalExecutor;
    };
}
