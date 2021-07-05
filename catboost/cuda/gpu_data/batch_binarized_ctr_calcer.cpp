#include "batch_binarized_ctr_calcer.h"

#include "gpu_binarization_helpers.h"
#include <catboost/private/libs/quantization/grid_creator.h>
#include <catboost/private/libs/quantization/utils.h>

void NCatboostCuda::TBatchedBinarizedCtrsCalcer::ComputeBinarizedCtrs(const TVector<ui32>& ctrs,
                                                                      TVector<NCatboostCuda::TBatchedBinarizedCtrsCalcer::TBinarizedCtr>* learnCtrs,
                                                                      TVector<NCatboostCuda::TBatchedBinarizedCtrsCalcer::TBinarizedCtr>* testCtrs) {
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
    TGpuBordersBuilder bordersBuilder(FeaturesManager);

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
            using TVisitor = TCtrVisitor<NCudaLib::TSingleMapping>;
            TVisitor ctrVisitor = [&](const NCB::TCtrConfig& config, const TSingleBuffer<float>& floatCtr, ui32 stream) {
                TCtr ctr;
                ctr.FeatureTensor = featureTensor;
                ctr.Configuration = config;
                const ui32 featureId = FeaturesManager.GetId(ctr);
                auto binarizationDescription = FeaturesManager.GetBinarizationDescription(ctr);

                TVector<float> borders = bordersBuilder.GetOrComputeBorders(featureId,
                    binarizationDescription,
                    floatCtr.SliceView(CtrTargets.LearnSlice),
                    stream);


                TVector<float> ctrValues;
                floatCtr
                    .CreateReader()
                    .SetCustomReadingStream(stream)
                    .SetReadSlice(CtrTargets.LearnSlice)
                    .Read(ctrValues);

                ui32 writeIndex = inverseIndex[featureId];
                auto& dst = (*learnCtrs)[writeIndex];

                dst.BinarizedCtr = NCB::BinarizeLine<ui8>(ctrValues,
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

                    testDst.BinarizedCtr = NCB::BinarizeLine<ui8>(testCtrValues,
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

NCatboostCuda::TCtrBinBuilder<NCudaLib::TSingleMapping> NCatboostCuda::TBatchedBinarizedCtrsCalcer::BuildFeatureTensorBins(const NCatboostCuda::TFeatureTensor& tensor,
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
        ythrow TCatBoostException() << "Precompute for combination ctrs with float splits is unimplemented yet";
    }
    return ctrBinBuilder;
}

TVector<TVector<NCB::TCtrConfig>> NCatboostCuda::TBatchedBinarizedCtrsCalcer::CreateGrouppedConfigs(const TVector<ui32>& ctrIds) {
    TVector<NCB::TCtrConfig> configs;
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
    TVector<TVector<NCB::TCtrConfig>> result;
    for (auto& entry : groupped) {
        result.push_back(entry.second);
    }
    return result;
}

TSingleBuffer<ui64> NCatboostCuda::TBatchedBinarizedCtrsCalcer::BuildCompressedBins(const NCB::TTrainingDataProvider& dataProvider,
                                                                                    ui32 featureManagerFeatureId,
                                                                                    ui32 devId) {
    const ui32 featureId = FeaturesManager.GetDataProviderId(featureManagerFeatureId);

    const auto& catFeature = **(dataProvider.ObjectsData->GetCatFeature(
        dataProvider.MetaInfo.FeaturesLayout->GetInternalFeatureIdx(featureId)));
    auto binsGpu = TSingleBuffer<ui32>::Create(NCudaLib::TSingleMapping(devId, catFeature.GetSize()));
    const ui32 uniqueValues = FeaturesManager.GetBinCount(featureManagerFeatureId);
    auto compressedBinsGpu = TSingleBuffer<ui64>::Create(CompressedSize<ui64>(binsGpu, uniqueValues));

    binsGpu.Write(catFeature.ExtractValues<ui32>(LocalExecutor));
    Compress(binsGpu, compressedBinsGpu, uniqueValues);
    return compressedBinsGpu;
}
