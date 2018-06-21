#pragma once

#include "ctr_helper.h"
#include "feature_parallel_dataset.h"

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/data/grid_creator.h>
#include <catboost/cuda/cuda_util/sort.h>
#include <catboost/cuda/data/feature.h>
#include <catboost/cuda/cuda_lib/device_subtasks_helper.h>

namespace NCatboostCuda {
    /*
     * Warning: this class doesn't guarantee optimal performance
     * Preprocessing stage is not critical and we have some gpu/cpu-memory tradeoff with some possible duplicate copies
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
                                    const TDataProvider& dataProvider,
                                    const TMirrorBuffer<TUi32>& ctrPermutation,
                                    const TDataProvider* linkedTest,
                                    const TMirrorBuffer<TUi32>* testIndices)
            : FeaturesManager(featuresManager)
            , CtrTargets(ctrTargets)
            , DataProvider(dataProvider)
            , LearnIndices(ctrPermutation.ConstCopyView())
            , LinkedTest(linkedTest)
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
        TVector<TVector<TCtrConfig>> CreateGrouppedConfigs(const TVector<ui32>& ctrIds);

        TCtrBinBuilder<NCudaLib::TSingleMapping> BuildFeatureTensorBins(const TFeatureTensor& tensor,
                                                                        int devId);

        TSingleBuffer<ui64> BuildCompressedBins(const TDataProvider& dataProvider,
                                                ui32 featureManagerFeatureId,
                                                ui32 devId);

    private:
        TBinarizedFeaturesManager& FeaturesManager;
        const TCtrTargets<NCudaLib::TMirrorMapping>& CtrTargets;
        const TDataProvider& DataProvider;
        TMirrorBuffer<const ui32> LearnIndices;

        const TDataProvider* LinkedTest = nullptr;
        TMirrorBuffer<const ui32> TestIndices;
    };
}
