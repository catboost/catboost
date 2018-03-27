#pragma once

#include "doc_parallel_dataset.h"
#include "compressed_index_builder.h"
#include "gpu_grid_creator.h"
#include "ctr_helper.h"
#include "batch_binarized_ctr_calcer.h"
#include "dataset_helpers.h"

#include <catboost/cuda/ctrs/ctr_calcers.h>
#include <catboost/cuda/ctrs/ctr.h>
#include <catboost/libs/helpers/interrupt.h>
#include <catboost/cuda/cuda_lib/device_subtasks_helper.h>

namespace NCatboostCuda {
    class TDocParallelDataSetBuilder {
    public:
        using TDataSetLayout = TDocParallelLayout;

        TDocParallelDataSetBuilder(TBinarizedFeaturesManager& featuresManager,
                                   const TDataProvider& dataProvider,
                                   const TDataProvider* linkedTest = nullptr)
            : FeaturesManager(featuresManager)
            , DataProvider(dataProvider)
            , LinkedTest(linkedTest)
        {
        }

        inline TDocParallelDataSetsHolder BuildDataSet(const ui32 permutationCount) {
            TDocParallelDataSetsHolder dataSetsHolder(DataProvider,
                                                      FeaturesManager,
                                                      LinkedTest);

            TSharedCompressedIndexBuilder<TDataSetLayout> compressedIndexBuilder(*dataSetsHolder.CompressedIndex);

            auto ctrsTarget = BuildCtrTarget(FeaturesManager,
                                             DataProvider,
                                             LinkedTest);

            dataSetsHolder.PermutationDataSets.resize(permutationCount);

            //
            TDataPermutation learnLoadBalancingPermutation = dataSetsHolder.LearnDocPerDevicesSplit->Permutation;

            TCudaBuffer<float, NCudaLib::TStripeMapping> targets;
            TCudaBuffer<float, NCudaLib::TStripeMapping> weights;

            targets.Reset(dataSetsHolder.LearnDocPerDevicesSplit->Mapping);
            weights.Reset(dataSetsHolder.LearnDocPerDevicesSplit->Mapping);

            targets.Write(learnLoadBalancingPermutation.Gather(DataProvider.GetTargets()));
            weights.Write(learnLoadBalancingPermutation.Gather(DataProvider.GetWeights()));

            for (ui32 permutationId = 0; permutationId < permutationCount; ++permutationId) {
                dataSetsHolder.PermutationDataSets[permutationId] = new TDocParallelDataSet(DataProvider,
                                                                                            dataSetsHolder.CompressedIndex,
                                                                                            GetPermutation(DataProvider, permutationId),
                                                                                            learnLoadBalancingPermutation,
                                                                                            dataSetsHolder.LearnDocPerDevicesSplit->SamplesGrouping,
                                                                                            TTarget<NCudaLib::TStripeMapping>(targets.ConstCopyView(),
                                                                                                                              weights.ConstCopyView()));
            }

            if (LinkedTest != nullptr) {
                TCudaBuffer<float, NCudaLib::TStripeMapping> testTargets;
                TCudaBuffer<float, NCudaLib::TStripeMapping> testWeights;

                TDataPermutation testLoadBalancingPermutation = dataSetsHolder.TestDocPerDevicesSplit->Permutation;

                testTargets.Reset(dataSetsHolder.TestDocPerDevicesSplit->Mapping);
                testWeights.Reset(dataSetsHolder.TestDocPerDevicesSplit->Mapping);

                testTargets.Write(testLoadBalancingPermutation.Gather(LinkedTest->GetTargets()));
                testWeights.Write(testLoadBalancingPermutation.Gather(LinkedTest->GetWeights()));

                dataSetsHolder.TestDataSet = new TDocParallelDataSet(*LinkedTest,
                                                                     dataSetsHolder.CompressedIndex,
                                                                     GetIdentityPermutation(*LinkedTest),
                                                                     testLoadBalancingPermutation,
                                                                     dataSetsHolder.TestDocPerDevicesSplit->SamplesGrouping,
                                                                     TTarget<NCudaLib::TStripeMapping>(testTargets.ConstCopyView(),
                                                                                                       testWeights.ConstCopyView()));
            }

            auto allFeatures = GetLearnFeatureIds(FeaturesManager);
            TVector<ui32> permutationIndependent;
            TVector<ui32> permutationDependent;
            {
                SplitByPermutationDependence(FeaturesManager,
                                             allFeatures,
                                             permutationCount,
                                             &permutationIndependent,
                                             &permutationDependent);
            }

            auto learnMapping = targets.GetMapping();
            TAtomicSharedPtr<TVector<ui32>> learnGatherIndices = new TVector<ui32>;
            learnLoadBalancingPermutation.FillOrder(*learnGatherIndices);

            TBinarizationInfoProvider learnBinarizationInfo(FeaturesManager,
                                                            &DataProvider);

            const ui32 permutationIndependentCompressedDataSetId = compressedIndexBuilder.AddDataSet(
                learnBinarizationInfo,
                {"Learn permutation independent features"},
                learnMapping,
                permutationIndependent,
                learnGatherIndices);

            for (ui32 permutationId = 0; permutationId < permutationCount; ++permutationId) {
                auto& dataSet = *dataSetsHolder.PermutationDataSets[permutationId];

                if (permutationDependent.size()) {
                    TDataSetDescription description;
                    description.Name = TStringBuilder() << "Learn permutation dependent features #" << permutationId;
                    dataSet.PermutationDependentFeatures = compressedIndexBuilder.AddDataSet(learnBinarizationInfo,
                                                                                             description,
                                                                                             learnMapping,
                                                                                             permutationDependent,
                                                                                             learnGatherIndices);
                }
                dataSet.PermutationIndependentFeatures = permutationIndependentCompressedDataSetId;
            }

            ui32 testDataSetId = -1;
            if (LinkedTest) {
                TDataSetDescription description;
                description.Name = "Test dataset";
                TAtomicSharedPtr<TVector<ui32>> testIndices = new TVector<ui32>;
                dataSetsHolder.TestDocPerDevicesSplit->Permutation.FillOrder(*testIndices);

                TBinarizationInfoProvider testBinarizationInfo(FeaturesManager,
                                                               LinkedTest);

                auto testMapping = dataSetsHolder.TestDataSet->GetSamplesMapping();
                testDataSetId = compressedIndexBuilder.AddDataSet(testBinarizationInfo,
                                                                  description,
                                                                  testMapping,
                                                                  allFeatures,
                                                                  testIndices);

                dataSetsHolder.TestDataSet->PermutationIndependentFeatures = testDataSetId;
            }

            compressedIndexBuilder.PrepareToWrite();

            {
                TFloatAndOneHotFeaturesWriter<TDocParallelLayout> floatFeaturesWriter(FeaturesManager,
                                                                                      compressedIndexBuilder,
                                                                                      DataProvider,
                                                                                      permutationIndependentCompressedDataSetId);
                floatFeaturesWriter.Write(permutationIndependent);
            }

            if (LinkedTest) {
                TFloatAndOneHotFeaturesWriter<TDocParallelLayout> floatFeaturesWriter(FeaturesManager,
                                                                                      compressedIndexBuilder,
                                                                                      *LinkedTest,
                                                                                      testDataSetId);
                floatFeaturesWriter.Write(permutationIndependent);
            }

            TMirrorBuffer<ui32> ctrEstimationOrder;
            TMirrorBuffer<ui32> testIndices;
            ctrEstimationOrder.Reset(NCudaLib::TMirrorMapping(DataProvider.GetSampleCount()));
            if (LinkedTest) {
                testIndices.Reset(NCudaLib::TMirrorMapping(LinkedTest->GetSampleCount()));
            }

            {
                MakeSequence(ctrEstimationOrder);
                if (LinkedTest) {
                    MakeSequence(testIndices);
                }

                TBatchedBinarizedCtrsCalcer ctrsCalcer(FeaturesManager,
                                                       *ctrsTarget,
                                                       DataProvider,
                                                       ctrEstimationOrder,
                                                       LinkedTest,
                                                       LinkedTest ? &testIndices : nullptr);

                TCtrsWriter<TDocParallelLayout> ctrsWriter(FeaturesManager,
                                                           compressedIndexBuilder,
                                                           ctrsCalcer,
                                                           permutationIndependentCompressedDataSetId,
                                                           testDataSetId);
                ctrsWriter.Write(permutationIndependent);
            }

            if (permutationDependent.size()) {
                for (ui32 permutationId = 0; permutationId < permutationCount; ++permutationId) {
                    auto& ds = *dataSetsHolder.PermutationDataSets[permutationId];
                    const TDataPermutation& ctrsEstimationPermutation = ds.GetCtrsEstimationPermutation();
                    ctrsEstimationPermutation.WriteOrder(ctrEstimationOrder);

                    {
                        const TDataProvider* linkedTest = permutationId == 0 ? LinkedTest : nullptr;
                        const TMirrorBuffer<ui32>* testIndicesPtr = (permutationId == 0 && linkedTest)
                                                                        ? &testIndices
                                                                        : nullptr;

                        TBatchedBinarizedCtrsCalcer ctrsCalcer(FeaturesManager,
                                                               *ctrsTarget,
                                                               DataProvider,
                                                               ctrEstimationOrder,
                                                               linkedTest,
                                                               testIndicesPtr);

                        TCtrsWriter<TDocParallelLayout> ctrsWriter(FeaturesManager,
                                                                   compressedIndexBuilder,
                                                                   ctrsCalcer,
                                                                   ds.PermutationDependentFeatures,
                                                                   testDataSetId);
                        ctrsWriter.Write(permutationDependent);
                    }
                    MATRIXNET_INFO_LOG << "Ctr computation for " << permutationId << " is finished" << Endl;
                }
            }
            compressedIndexBuilder.Finish();

            return dataSetsHolder;
        }

    private:
        TBinarizedFeaturesManager& FeaturesManager;
        const TDataProvider& DataProvider;
        const TDataProvider* LinkedTest;
    };
}
