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
                                  TVector<TBinarizedCtr>* testCtrs) {
            THashMap<TFeatureTensor, TVector<ui32>> groupedByTensorFeatures;
            //from ctrs[i] to i
            TMap<ui32, ui32> inverseIndex;

            TVector<TFeatureTensor> featureTensors;

            for (ui32 i = 0; i < ctrs.size(); ++i) {
                CB_ENSURE(FeaturesManager.IsCtr(ctrs[i]));
                TCtr ctr = FeaturesManager.GetCtr(ctrs[i]);
                groupedByTensorFeatures[ctr.FeatureTensor].push_back(ctrs[i]);
                inverseIndex[ctrs[i]] = i;
            }
            for (const auto& group : groupedByTensorFeatures) {
                featureTensors.push_back(group.first);
            }

            learnCtrs->clear();
            learnCtrs->resize(ctrs.size());

            if (LinkedTest) {
                CB_ENSURE(testCtrs);
                testCtrs->clear();
                testCtrs->resize(ctrs.size());
            }

            TAdaptiveLock lock;

            NCudaLib::RunPerDeviceSubtasks([&](ui32 devId) {
                auto ctrSingleDevTargetView = DeviceView(CtrTargets, devId);

                while (true) {
                    TFeatureTensor featureTensor;

                    with_lock (lock) {
                        if (featureTensors.size()) {
                            featureTensor = featureTensors.back();
                            featureTensors.pop_back();
                        } else {
                            return;
                        }
                    }

                    auto ctrVisitor = [&](const TCtrConfig& config, TSingleBuffer<const float> floatCtr, ui32 stream) {
                        TCtr ctr;
                        ctr.FeatureTensor = featureTensor;
                        ctr.Configuration = config;
                        const ui32 featureId = FeaturesManager.GetId(ctr);
                        auto binarizationDescription = FeaturesManager.GetBinarizationDescription(ctr);

                        TVector<float> borders;
                        bool hasBorders = false;

                        with_lock (lock) {
                            if (FeaturesManager.HasBorders(featureId)) {
                                borders = FeaturesManager.GetBorders(featureId);
                                hasBorders = true;
                            }
                        }

                        if (!hasBorders) {
                            TOnCpuGridBuilderFactory gridBuilderFactory;
                            TSingleBuffer<float> sortedFeature = TSingleBuffer<float>::CopyMapping(floatCtr);
                            sortedFeature.Copy(floatCtr, stream);
                            RadixSort(sortedFeature, false, stream);
                            TVector<float> sortedFeatureCpu;
                            sortedFeature.Read(sortedFeatureCpu,
                                               stream);

                            borders = gridBuilderFactory
                                          .Create(binarizationDescription.BorderSelectionType)
                                          ->BuildBorders(sortedFeatureCpu,
                                                         binarizationDescription.BorderCount);


                            //hack to work with constant ctr's
                            if (borders.size() == 0) {
                                borders.push_back(0.5);
                            }

                            with_lock (lock) {
                                if (FeaturesManager.HasBorders(featureId)) {
                                    borders = FeaturesManager.GetBorders(featureId);
                                } else {
                                    FeaturesManager.SetBorders(featureId, borders);
                                }
                            }
                        }

                        TVector<float> ctrValues;
                        floatCtr
                            .CreateReader()
                            .SetCustomReadingStream(stream)
                            .SetReadSlice(CtrTargets.LearnSlice)
                            .Read(ctrValues);

                        ui32 writeIndex = inverseIndex[featureId];
                        auto& dst = (*learnCtrs)[writeIndex];

                        dst.BinarizedCtr = BinarizeLine<ui8>(~ctrValues,
                                                             ctrValues.size(),
                                                             ENanMode::Forbidden,
                                                             borders);

                        dst.BinCount = borders.size() + 1;


                        if (LinkedTest) {
                            auto& testDst = (*testCtrs)[writeIndex];
                            TVector<float> testCtrValues;
                            floatCtr.CreateReader()
                                .SetCustomReadingStream(stream)
                                .SetReadSlice(CtrTargets.TestSlice)
                                .Read(testCtrValues);

                            testDst.BinarizedCtr = BinarizeLine<ui8>(~testCtrValues,
                                                                     testCtrValues.size(),
                                                                     ENanMode::Forbidden,
                                                                     borders);
                            testDst.BinCount = borders.size() + 1;
                        }
                    };

                    using TCtrHelper = TCalcCtrHelper<NCudaLib::TSingleMapping>;
                    THolder<TCtrHelper> ctrHelper;
                    {
                        auto binBuilder = BuildFeatureTensorBins(featureTensor,
                                                                 devId);

                        ctrHelper.Reset(new TCalcCtrHelper<NCudaLib::TSingleMapping>(ctrSingleDevTargetView, binBuilder.GetIndices()));

                        const bool isFeatureFreqOnFullSet = FeaturesManager.UseFullSetForCatFeatureStatCtrs();
                        ctrHelper->UseFullDataForCatFeatureStats(isFeatureFreqOnFullSet);
                    }

                    auto grouppedConfigs = CreateGrouppedConfigs(groupedByTensorFeatures[featureTensor]);
                    for (auto& group : grouppedConfigs) {
                        ctrHelper->VisitEqualUpToPriorCtrs(group, ctrVisitor);
                    }
                }
            },
                                           false /* only local, TODO(noxoomo) test infiniband and enable if gives profit*/);
        }

    private:
        TVector<TVector<TCtrConfig>> CreateGrouppedConfigs(const TVector<ui32>& ctrIds) {
            TVector<TCtrConfig> configs;
            TFeatureTensor tensor;

            for (ui32 i = 0; i < ctrIds.size(); ++i) {
                const ui32 featureId = ctrIds[i];
                TCtr ctr = FeaturesManager.GetCtr(featureId);
                if (i > 0) {
                    CB_ENSURE(ctr.FeatureTensor == tensor);
                } else {
                    tensor = ctr.FeatureTensor;
                }
                configs.push_back(ctr.Configuration);
            }
            auto groupped = CreateEqualUpToPriorAndBinarizationCtrsGroupping(configs);
            TVector<TVector<TCtrConfig>> result;
            for (auto& entry : groupped) {
                result.push_back(entry.second);
            }
            return result;
        }

        TCtrBinBuilder<NCudaLib::TSingleMapping> BuildFeatureTensorBins(const TFeatureTensor& tensor,
                                                                        int devId) {
            CB_ENSURE(tensor.GetSplits().size() == 0, "Unimplemented here yet");
            TCtrBinBuilder<NCudaLib::TSingleMapping> ctrBinBuilder;

            {
                TSingleBuffer<const ui32> learnIndices = LearnIndices.DeviceView(devId);
                TSingleBuffer<const ui32> testIndices;

                if (LinkedTest) {
                    testIndices = TestIndices.DeviceView(devId);
                }

                ctrBinBuilder
                    .SetIndices(learnIndices,
                                LinkedTest ? &testIndices : nullptr);
            }

            for (const ui32 feature : tensor.GetCatFeatures()) {
                const auto learnBins = BuildCompressedBins(DataProvider,
                                                           feature,
                                                           devId);
                const ui32 uniqueValues = FeaturesManager.GetBinCount(feature);
                CB_ENSURE(uniqueValues > 1, "Error: useless catFeature found");

                if (LinkedTest) {
                    auto testBins = BuildCompressedBins(*LinkedTest, feature, devId);

                    ctrBinBuilder.AddCompressedBins(learnBins,
                                                    testBins,
                                                    uniqueValues);
                } else {
                    ctrBinBuilder.AddCompressedBins(learnBins,
                                                    uniqueValues);
                }
            }
            if (tensor.GetSplits().size()) {
                ythrow TCatboostException() << "Precompute for combination ctrs with float splits is unimplemented yet";
            }
            return ctrBinBuilder;
        }

        TSingleBuffer<ui64> BuildCompressedBins(const TDataProvider& dataProvider,
                                                ui32 featureManagerFeatureId,
                                                ui32 devId) {
            const ui32 featureId = FeaturesManager.GetDataProviderId(featureManagerFeatureId);
            const auto& catFeature = dynamic_cast<const ICatFeatureValuesHolder&>(dataProvider.GetFeatureById(featureId));
            auto binsGpu = TSingleBuffer<ui32>::Create(NCudaLib::TSingleMapping(devId, catFeature.GetSize()));
            const ui32 uniqueValues = FeaturesManager.GetBinCount(featureManagerFeatureId);
            auto compressedBinsGpu = TSingleBuffer<ui64>::Create(CompressedSize<ui64>(binsGpu, uniqueValues));
            binsGpu.Write(catFeature.ExtractValues());
            Compress(binsGpu, compressedBinsGpu, uniqueValues);
            return compressedBinsGpu;
        }

    private:
        TBinarizedFeaturesManager& FeaturesManager;
        const TCtrTargets<NCudaLib::TMirrorMapping>& CtrTargets;
        const TDataProvider& DataProvider;
        TMirrorBuffer<const ui32> LearnIndices;

        const TDataProvider* LinkedTest = nullptr;
        TMirrorBuffer<const ui32> TestIndices;
    };
}
