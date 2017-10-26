#pragma once

#include "fold_based_dataset.h"
#include "binarized_dataset_builder.h"
#include "gpu_grid_creator.h"
#include "ctr_helper.h"

#include <catboost/cuda/ctrs/ctr_calcers.h>
#include <catboost/cuda/ctrs/ctr.h>

namespace NCatboostCuda
{
    inline TMirrorBuffer<ui8> BuildBinarizedTarget(const TBinarizedFeaturesManager& featuresManager,
                                                   const yvector<float>& targets)
    {
        CB_ENSURE(featuresManager.HasTargetBinarization(),
                  "Error: target binarization should be set beforedataSet build");
        auto& borders = featuresManager.GetTargetBorders();

        auto binarizedTarget = BinarizeLine<ui8>(~targets, targets.size(),
                                                 borders);

        TMirrorBuffer<ui8> binarizedTargetGpu = TMirrorBuffer<ui8>::Create(
                NCudaLib::TMirrorMapping(binarizedTarget.size()));
        binarizedTargetGpu.Write(binarizedTarget);
        return binarizedTargetGpu;
    }

    template<class T>
    inline yvector<T> Join(const yvector<T>& left,
                           const yvector<T>* right)
    {
        yvector<float> result(left.begin(), left.end());
        if (right)
        {
            for (auto& element : *right)
            {
                result.push_back(element);
            }
        }
        return result;
    }

    inline bool IsTrivialWeights(const yvector<float>& weights)
    {
        for (auto& weight : weights)
        {
            if (weight != weights[0])
            {
                return false;
            }
        }
        CB_ENSURE(weights[0] == 1.0f, "Error: we expect constant weights to be equal to 1.0f");
        return true;
    }

    template<class TGridPolicy>
    class TPermutationDataSetBuilder
    {
    public:
        TPermutationDataSetBuilder(TBinarizedFeaturesManager& featuresManager,
                                   const TDataProvider& dataProvider,
                                   const TDataPermutation& permutation,
                                   const TMirrorBuffer<ui32>& indices,
                                   const TCtrTargets<NCudaLib::TMirrorMapping>& ctrTargets)
                : FeaturesManager(featuresManager)
                  , DataProvider(dataProvider)
                  , LearnIndices(indices)
                  , Permutation(permutation)
                  , CtrTargets(ctrTargets)
        {
        }

        TPermutationDataSetBuilder& ShuffleFeaturesFlag(bool flag)
        {
            ShuffleFeatures = flag;
            return *this;
        }

        TPermutationDataSetBuilder& UseOneHot(bool flag)
        {
            AddOneHotFeatures = flag;
            return *this;
        }

        TPermutationDataSetBuilder& UseFloatFeatures(bool flag)
        {
            AddFloatFeatures = flag;
            return *this;
        }

        template<class TContainer>
        TPermutationDataSetBuilder& SetIgnoredFeatures(const TContainer& ids)
        {
            IgnoredFeatures.insert(ids.begin(), ids.end());
            return *this;
        }

        TPermutationDataSetBuilder& UseCtrs(const yset<ECtrType>& ctrs)
        {
            CtrTypes.insert(ctrs.begin(), ctrs.end());
            return *this;
        }

        TPermutationDataSetBuilder& BuildLinkedTest(const TDataProvider& linkedTest,
                                                    const TMirrorBuffer<ui32>& testIndices)
        {
            LinkedTest = &linkedTest;
            TestIndices = &testIndices;
            return *this;
        }

        TPermutationDataSetBuilder& SetFloatFeatures(const yvector<ui32>& featureIds)
        {
            if (AddFloatFeatures)
            {
                const auto& policyFeatures = TFeaturesSplitter<NCudaLib::TStripeMapping>::ExtractFeaturesForPolicy<TGridPolicy>(
                        FeaturesManager, featureIds);
                for (ui32 feature : policyFeatures.PolicyFeatures)
                {
                    AddFeatureIdIfNotInIgnoredSet(feature);
                }
            }
            return *this;
        }

        TPermutationDataSetBuilder& SetCatFeatures(const yvector<ui32>& featureIds)
        {
            for (ui32 catFeature : featureIds)
            {
                if (AddOneHotFeatures && FeaturesManager.UseForOneHotEncoding(catFeature))
                {
                    if (TGridPolicy::CanProceed(FeaturesManager.GetBinCount(catFeature)))
                    {
                        AddFeatureIdIfNotInIgnoredSet(catFeature);
                    }
                }

                if (FeaturesManager.UseForCtr(catFeature))
                {
                    for (auto& ctr : CtrTypes)
                    {
                        const auto simpleCtrsForType = FeaturesManager.CreateSimpleCtrsForType(catFeature,
                                                                                               ctr);
                        for (auto ctrFeatureId : simpleCtrsForType)
                        {
                            if (TGridPolicy::CanProceed(FeaturesManager.GetBinCount(ctrFeatureId)))
                            {
                                AddFeatureIdIfNotInIgnoredSet(ctrFeatureId);
                            }
                        }
                    }
                }
            }
            return *this;
        }

        const yvector<ui32>& GetFeatureIds() const
        {
            return FeatureIds;
        }


        bool IsTestGpuDataSetRequired() const
        {
            return LinkedTest != nullptr;
        }

        void Build(TGpuBinarizedDataSet<TGridPolicy>& learn,
                   TGpuBinarizedDataSet<TGridPolicy>* test)
        {
            if (LinkedTest)
            {
                CB_ENSURE(test, "Error: provider test set ptr to build");
            }
            const ui64 docCount = Permutation.ObservationsSlice().Size();
            auto docsMapping = NCudaLib::TMirrorMapping(docCount);

            //sort and split between devices
            auto splittedFeatures = TFeaturesSplitter<NCudaLib::TStripeMapping>::Split(FeaturesManager,
                                                                                       FeatureIds,
                                                                                       ShuffleFeatures);

            yvector<ui32> gatherIndices;
            Permutation.FillOrder(gatherIndices);

            TGpuBinarizedDataSetBuilder<TGridPolicy> learnBuilder(splittedFeatures.Layout,
                                                                  docsMapping,
                                                                  &gatherIndices);

            THolder<TGpuBinarizedDataSetBuilder<TGridPolicy>> testBuilderPtr;

            const auto& featuresToBuild = splittedFeatures.FeatureIds;
            learnBuilder
                    .SetFeatureIds(featuresToBuild)
                    .UseForOneHotIds(FeaturesManager.GetOneHotIds(featuresToBuild));

            if (LinkedTest)
            {
                auto testMapping = NCudaLib::TMirrorMapping(LinkedTest->GetSampleCount());

                testBuilderPtr.Reset(new TGpuBinarizedDataSetBuilder<TGridPolicy>(splittedFeatures.Layout,
                                                                                  testMapping));

                testBuilderPtr->SetFeatureIds(featuresToBuild)
                        .UseForOneHotIds(FeaturesManager.GetOneHotIds(featuresToBuild));
            }

            yvector<ui32> ctrs;

            for (auto feature : featuresToBuild)
            {
                if (FeaturesManager.IsCtr(feature))
                {
                    ctrs.push_back(feature);
                } else if (FeaturesManager.IsFloat(feature))
                {
                    WriteFloatFeature(feature,
                                      DataProvider,
                                      learnBuilder);
                    if (LinkedTest)
                    {
                        WriteFloatFeature(feature,
                                          *LinkedTest,
                                          *testBuilderPtr);
                    }
                } else if (FeaturesManager.IsCat(feature))
                {
                    CB_ENSURE(FeaturesManager.UseForOneHotEncoding(feature));
                    WriteCatFeature(feature, DataProvider, learnBuilder);
                    if (LinkedTest)
                    {
                        WriteCatFeature(feature, *LinkedTest, *testBuilderPtr);
                    }
                }
            }

            std::sort(ctrs.begin(), ctrs.end(), [&](ui32 left, ui32 right) -> bool
            {
                return FeaturesManager.GetCtr(left).FeatureTensor < FeaturesManager.GetCtr(right).FeatureTensor;
            });

            if (ctrs.size())
            {
                MATRIXNET_INFO_LOG << "Start building ctrs " << ctrs.size() << Endl;
                for (auto& ctrId : ctrs)
                {
                    AddCtr(ctrId, learnBuilder, test ? testBuilderPtr.Get() : nullptr);
                }
            }

            learn = std::move(learnBuilder.Finish());
            if (test)
            {
                (*test) = std::move(testBuilderPtr->Finish());
            }
            MATRIXNET_INFO_LOG << "Build is done" << Endl;
        }

    private:
        void AddFeatureIdIfNotInIgnoredSet(ui32 featureId)
        {
            if (IgnoredFeatures.count(featureId) == 0)
            {
                FeatureIds.push_back(featureId);
            }
        }

        void WriteFloatFeature(const ui32 feature,
                               const TDataProvider& dataProvider,
                               TGpuBinarizedDataSetBuilder<TGridPolicy>& builder)
        {
            const auto& featureStorage = dataProvider.GetFeatureById(FeaturesManager.GetDataProviderId(feature));

            if (featureStorage.GetType() == EFeatureValuesType::BinarizedFloat)
            {
                const auto& featuresHolder = dynamic_cast<const TBinarizedFloatValuesHolder&>(featureStorage);
                CB_ENSURE(featuresHolder.GetBorders() == FeaturesManager.GetBorders(feature),
                          "Error: unconsistent borders for feature #" << feature);
                builder.Write(feature,
                              static_cast<const ui32>(featuresHolder.GetBorders().size() + 1),
                              featuresHolder.ExtractValues());

            } else
            {
                CB_ENSURE(featureStorage.GetType() == EFeatureValuesType::Float);

                const auto& holder = dynamic_cast<const TFloatValuesHolder&>(featureStorage);
                const auto& borders = FeaturesManager.GetBorders(feature);

                auto bins = BinarizeLine<ui32>(holder.GetValuesPtr(),
                                               holder.GetSize(),
                                               borders);

                builder.Write(feature,
                              static_cast<const ui32>(borders.size() + 1),
                              bins);
            }
        }

        void WriteCatFeature(const ui32 feature,
                             const TDataProvider& dataProvider,
                             TGpuBinarizedDataSetBuilder<TGridPolicy>& builder)
        {
            const auto& featureStorage = dataProvider.GetFeatureById(FeaturesManager.GetDataProviderId(feature));

            CB_ENSURE(featureStorage.GetType() == EFeatureValuesType::Categorical);

            const auto& featuresHolder = dynamic_cast<const ICatFeatureValuesHolder&>(featureStorage);
            builder.Write(feature,
                    /*hack for using only learn catFeatures as one-hot splits. test set storage should now about unseen catFeatures from learn. binCount is totalUniqueValues (learn + test)*/
                          Min<ui32>(FeaturesManager.GetBinCount(feature), featuresHolder.GetUniqueValues()),
                          featuresHolder.ExtractValues());
        }

        void AddCtr(const ui32 feature,
                    TGpuBinarizedDataSetBuilder<TGridPolicy>& learnBuilder,
                    TGpuBinarizedDataSetBuilder<TGridPolicy>* testBuilder)
        {
            ui32 devId = 0;

            TCtr ctr = FeaturesManager.GetCtr(feature);
            CB_ENSURE(ctr.IsSimple(), "Error: non-simple baseCtrs not implemented yet");
            auto binarizationDescription = FeaturesManager.GetBinarizationDescription(ctr);

            if (CurrentTensor != ctr.FeatureTensor)
            {
                BuildCtrBinIndices(ctr.FeatureTensor.GetCatFeatures()[0], devId);
                CurrentTensor = ctr.FeatureTensor;
            }
            TSingleBuffer<float> floatCtr = CtrHelper->ComputeCtr(ctr.Configuration);

            auto borders = FeaturesManager.GetOrBuildCtrBorders(ctr, [&]() -> yvector<float>
            {
                TOnCpuGridBuilderFactory gridBuilderFactory;
                TSingleBuffer<float> sortedFeature = TSingleBuffer<float>::CopyMapping(floatCtr);
                sortedFeature.Copy(floatCtr);
                RadixSort(sortedFeature);
                yvector<float> sortedFeatureCpu;
                sortedFeature.Read(sortedFeatureCpu);

                return gridBuilderFactory
                        .Create(binarizationDescription.BorderSelectionType)
                        ->BuildBorders(sortedFeatureCpu,
                                       binarizationDescription.Discretization);
            });

            yvector<float> ctrValues;
            floatCtr
                    .CreateReader()
                    .SetReadSlice(CtrTargets.LearnSlice)
                    .Read(ctrValues);

            auto binarizedValues = BinarizeLine<ui32>(~ctrValues,
                                                      ctrValues.size(),
                                                      borders);

            learnBuilder.Write(feature,
                               borders.size() + 1,
                               binarizedValues);

            if (LinkedTest)
            {
                CB_ENSURE(testBuilder);

                yvector<float> testCtrValues;
                floatCtr.CreateReader()
                        .SetReadSlice(CtrTargets.TestSlice)
                        .Read(testCtrValues);

                auto testBinValues = BinarizeLine<ui32>(~testCtrValues,
                                                        testCtrValues.size(),
                                                        borders);

                testBuilder->Write(feature, borders.size() + 1, testBinValues);
            }
        }

        void BuildCtrBinIndices(const ui32 feature,
                                const ui32 devId = 0)
        {
            const auto learnBins = BuildCompressedBins(DataProvider,
                                                       feature,
                                                       devId);

            TSingleBuffer<ui32> testIndices;
            if (LinkedTest)
            {
                testIndices = TestIndices->DeviceView(devId);
            }

            CtrSingleDevTargetView = DeviceView(CtrTargets, devId);

            CtrBinBuilder
                    .SetIndices(LearnIndices.DeviceView(devId),
                                LinkedTest
                                ? &testIndices
                                : nullptr);

            const ui32 uniqueValues = FeaturesManager.GetBinCount(feature);
            CB_ENSURE(uniqueValues > 1, "Error: useless catFeature found");

            if (LinkedTest)
            {
                auto testBins = BuildCompressedBins(*LinkedTest, feature, devId);
                CtrBinBuilder.AddCompressedBins(learnBins,
                                                testBins,
                                                uniqueValues);
            } else
            {
                CtrBinBuilder.AddCompressedBins(learnBins, uniqueValues);
            }

            CtrHelper.Reset(new TCalcCtrHelper<NCudaLib::TSingleMapping>(CtrSingleDevTargetView,
                                                                         CtrBinBuilder.GetIndices()));
        }

        TSingleBuffer<ui64> BuildCompressedBins(const TDataProvider& dataProvider,
                                                ui32 featureManagerFeatureId,
                                                ui32 devId = 0)
        {
            const ui32 featureId = FeaturesManager.GetDataProviderId(featureManagerFeatureId);
            const ICatFeatureValuesHolder& catFeature = dynamic_cast<const ICatFeatureValuesHolder&>(dataProvider.GetFeatureById(
                    featureId));
            auto binsGpu = TSingleBuffer<ui32>::Create(NCudaLib::TSingleMapping(devId, catFeature.GetSize()));
            const ui32 uniqueValues = FeaturesManager.GetBinCount(featureManagerFeatureId);
            auto compressedBinsGpu = TSingleBuffer<ui64>::Create(CompressedSize<ui64>(binsGpu, uniqueValues));
            binsGpu.Write(catFeature.ExtractValues());
            Compress(binsGpu, compressedBinsGpu, uniqueValues);
            return compressedBinsGpu;
        }

        yset<ECtrType> CtrTypes;
        bool AddOneHotFeatures = true;
        bool AddFloatFeatures = true;

        yvector<ui32> FeatureIds;
        yset<ui32> IgnoredFeatures;
        TBinarizedFeaturesManager& FeaturesManager;

        const TDataProvider& DataProvider;
        const TMirrorBuffer<ui32>& LearnIndices;

        const TDataProvider* LinkedTest = nullptr;
        const TMirrorBuffer<ui32>* TestIndices = nullptr;

        const TDataPermutation& Permutation;
        const TCtrTargets<NCudaLib::TMirrorMapping>& CtrTargets;

        //context for building ctr_description
        TCtrBinBuilder<NCudaLib::TSingleMapping> CtrBinBuilder;
        THolder<TCalcCtrHelper<NCudaLib::TSingleMapping>> CtrHelper;
        TCtrTargets<NCudaLib::TSingleMapping> CtrSingleDevTargetView;
        TFeatureTensor CurrentTensor;
        bool ShuffleFeatures = true;
    };

//Test dataset will be linked on first permutation (direct indexing)
    template<NCudaLib::EPtrType CatFeaturesStoragePtrType = NCudaLib::CudaDevice>
    class TDataSetHoldersBuilder
    {
    public:
        TDataSetHoldersBuilder(TBinarizedFeaturesManager& featuresManager,
                               const TDataProvider& dataProvider,
                               const TDataProvider* linkedTest = nullptr,
                               bool shuffleFeatures = true,
                               ui32 blockSize = 1)
                : FeaturesManager(featuresManager)
                  , DataProvider(dataProvider)
                  , LinkedTest(linkedTest)
                  , ShuffleFeatures(shuffleFeatures)
                  , DataProviderPermutationBlockSize(blockSize)
        {
        }

        inline TDataSetsHolder<CatFeaturesStoragePtrType> BuildDataSet(const ui32 permutationCount)
        {
            TDataSetsHolder<CatFeaturesStoragePtrType> dataSetsHolder(DataProvider, FeaturesManager);

            dataSetsHolder.CtrTargets.Reset(new TCtrTargets<NCudaLib::TMirrorMapping>());
            auto& ctrsTarget = *dataSetsHolder.CtrTargets;
            {
                yvector<float> joinedTarget = Join(DataProvider.GetTargets(),
                                                   LinkedTest ? &LinkedTest->GetTargets() : nullptr);
                //ctrs
                BuildCtrTarget(joinedTarget, ctrsTarget);
            }

            dataSetsHolder.PermutationDataSets.resize(permutationCount);

            const bool isTrivialLearnWeights = IsTrivialWeights(DataProvider.GetWeights());
            {
                const auto learnMapping = NCudaLib::TMirrorMapping(ctrsTarget.LearnSlice.Size());

                if (isTrivialLearnWeights == ctrsTarget.IsTrivialWeights())
                {
                    dataSetsHolder.DirectWeights = ctrsTarget.Weights.SliceView(ctrsTarget.LearnSlice);
                } else
                {
                    dataSetsHolder.DirectWeights.Reset(learnMapping);
                    dataSetsHolder.DirectWeights.Write(DataProvider.GetWeights());
                }
                if (isTrivialLearnWeights && ctrsTarget.IsTrivialWeights())
                {
                    dataSetsHolder.DirectTarget = ctrsTarget.WeightedTarget.SliceView(ctrsTarget.LearnSlice);
                } else
                {
                    dataSetsHolder.DirectTarget.Reset(learnMapping);
                    dataSetsHolder.DirectTarget.Write(DataProvider.GetTargets());
                }
            }

            for (ui32 permutationId = 0; permutationId < permutationCount; ++permutationId)
            {
                dataSetsHolder.PermutationDataSets[permutationId].Reset(
                        new TDataSet<CatFeaturesStoragePtrType>(DataProvider,
                                                                permutationId,
                                                                DataProviderPermutationBlockSize));
            }

            for (ui32 permutationId = 0; permutationId < permutationCount; ++permutationId)
            {
                auto& dataSet = *dataSetsHolder.PermutationDataSets[permutationId];
                dataSet.CtrTargets = dataSetsHolder.CtrTargets;

                const auto targetsMapping = NCudaLib::TMirrorMapping(ctrsTarget.LearnSlice.Size());

                dataSet.Indices.Reset(targetsMapping);
                dataSet.Permutation.WriteOrder(dataSet.Indices);

                {
                    if (dataSet.Permutation.IsIdentity())
                    {
                        dataSet.InverseIndices = dataSet.Indices.CopyView();
                        dataSet.Targets = dataSetsHolder.DirectTarget.CopyView();
                    } else
                    {
                        dataSet.InverseIndices.Reset(targetsMapping);
                        dataSet.Permutation.WriteInversePermutation(dataSet.InverseIndices);

                        dataSet.Targets.Reset(dataSetsHolder.DirectTarget.GetMapping());
                        Gather(dataSet.Targets, dataSetsHolder.DirectTarget, dataSet.Indices);
                    }

                    if (isTrivialLearnWeights)
                    {
                        dataSet.Weights = dataSetsHolder.DirectWeights.CopyView();
                    } else
                    {
                        dataSet.Weights.Reset(dataSetsHolder.DirectTarget.GetMapping());
                        Gather(dataSet.Weights, dataSetsHolder.DirectWeights, dataSet.Indices);
                    }

                    if (isTrivialLearnWeights)
                    {
                        dataSet.Weights = dataSetsHolder.DirectWeights.CopyView();
                    } else
                    {
                        dataSet.Weights.Reset(dataSetsHolder.DirectTarget.GetMapping());
                        Gather(dataSet.Weights, dataSetsHolder.DirectWeights, dataSet.Indices);
                    }
                }
            }

            if (LinkedTest != nullptr)
            {
                BuildTestTargetAndIndices(dataSetsHolder, ctrsTarget);
            }

            BuildPermutationIndependentGpuDataSets(dataSetsHolder);

            {
                for (ui32 permutationId = 0; permutationId < permutationCount; ++permutationId)
                {
                    //link common datasets
                    if (permutationId > 0)
                    {
                        dataSetsHolder.PermutationDataSets[permutationId]->CatFeatures = dataSetsHolder.PermutationDataSets[0]->CatFeatures;
                        dataSetsHolder.PermutationDataSets[permutationId]->PermutationIndependentFeatures = dataSetsHolder.PermutationDataSets[0]->PermutationIndependentFeatures;
                    }
                    BuildPermutationDependentGpuFeatures(dataSetsHolder, permutationId);
                }
                MATRIXNET_INFO_LOG << "Build permutation dependent datasets is done" << Endl;
            }

            return dataSetsHolder;
        }

    private:

        template<class TBuilder>
        void AddFeaturesToBuilder(TBuilder& builder,
                                  bool skipFloat,
                                  bool skipCat,
                                  yvector<ui32>& proceededFeatures)
        {
            builder.SetIgnoredFeatures(proceededFeatures);
            if (!skipFloat)
            {
                builder.SetFloatFeatures(FeaturesManager.GetFloatFeatureIds());
            }
            if (!skipCat)
            {
                builder.SetCatFeatures(FeaturesManager.GetCatFeatureIds());
            }
            const auto& features = builder.GetFeatureIds();
            proceededFeatures.insert(proceededFeatures.end(), features.begin(), features.end());

        }

        TGpuFeatures<>*
        GetFeaturesPtr(TDataSet<CatFeaturesStoragePtrType>& dataSet, bool permutationIndependentFeatures)
        {
            if (permutationIndependentFeatures)
            {
                return dataSet.PermutationIndependentFeatures.Get();
            }
            return &dataSet.PermutationDependentFeatures;
        }

        void BuildGpuFeatures(TDataSetsHolder<CatFeaturesStoragePtrType>& dataSetsHolder,
                              bool buildPermutationIndependent,
                              ui32 permutation,
                              bool skipFloat = false,
                              bool skipCat = false)
        {
            if (buildPermutationIndependent)
            {
                CB_ENSURE(permutation == 0);
            }
            auto* gpuFeaturesPtr = GetFeaturesPtr(*dataSetsHolder.PermutationDataSets[permutation],
                                                  buildPermutationIndependent);
            CB_ENSURE(gpuFeaturesPtr);
            auto* linkedTestFeaturesPtr = dataSetsHolder.TestDataSet ? GetFeaturesPtr(*dataSetsHolder.TestDataSet,
                                                                                      buildPermutationIndependent)
                                                                     : nullptr;

            yvector<ui32> proceededFeatures;
            {
                auto commonBinFeaturesBuilder = CreateBuilder<TBinaryFeatureGridPolicy>(dataSetsHolder,
                                                                                        permutation,
                                                                                        buildPermutationIndependent);
                AddFeaturesToBuilder(commonBinFeaturesBuilder, skipFloat, skipCat, proceededFeatures);
                if (commonBinFeaturesBuilder.IsTestGpuDataSetRequired())
                {
                    CB_ENSURE(linkedTestFeaturesPtr, "Provide linked test features");
                }
                commonBinFeaturesBuilder.Build(gpuFeaturesPtr->BinaryFeatures,
                                               commonBinFeaturesBuilder.IsTestGpuDataSetRequired()
                                               ? &linkedTestFeaturesPtr->BinaryFeatures : nullptr);
            }

            {
                auto commonHalfByteFeaturesBuilder = CreateBuilder<THalfByteFeatureGridPolicy>(dataSetsHolder,
                                                                                               permutation,
                                                                                               buildPermutationIndependent);
                AddFeaturesToBuilder(commonHalfByteFeaturesBuilder, skipFloat, skipCat, proceededFeatures);
                if (commonHalfByteFeaturesBuilder.IsTestGpuDataSetRequired())
                {
                    CB_ENSURE(linkedTestFeaturesPtr, "Provide linked test features");
                }
                commonHalfByteFeaturesBuilder.Build(gpuFeaturesPtr->HalfByteFeatures,
                                                    commonHalfByteFeaturesBuilder.IsTestGpuDataSetRequired()
                                                    ? &linkedTestFeaturesPtr->HalfByteFeatures : nullptr);
            }

            {
                auto commonByteFeaturesBuilder = CreateBuilder<TByteFeatureGridPolicy>(dataSetsHolder,
                                                                                       permutation,
                                                                                       buildPermutationIndependent);
                AddFeaturesToBuilder(commonByteFeaturesBuilder, skipFloat, skipCat, proceededFeatures);
                if (commonByteFeaturesBuilder.IsTestGpuDataSetRequired())
                {
                    CB_ENSURE(linkedTestFeaturesPtr, "Provide linked test features");
                }
                commonByteFeaturesBuilder.Build(gpuFeaturesPtr->Features,
                                                commonByteFeaturesBuilder.IsTestGpuDataSetRequired()
                                                ? &linkedTestFeaturesPtr->Features : nullptr);
            }
        }

        void BuildPermutationIndependentGpuDataSets(TDataSetsHolder<CatFeaturesStoragePtrType>& dataSetsHolder)
        {
            auto& firstDataSet = dataSetsHolder.PermutationDataSets[0];
            auto& dataSet = *firstDataSet;

            ResetSharedDataSets(dataSet);

            if (LinkedTest)
            {
                ResetSharedDataSets(*dataSetsHolder.TestDataSet);
            }
            BuildGpuFeatures(dataSetsHolder, true, 0, false, false);
            {
                BuildCompressedCatFeatures(DataProvider, *dataSetsHolder.PermutationDataSets[0]->CatFeatures);

                if (LinkedTest != nullptr)
                {
                    BuildCompressedCatFeatures(*LinkedTest, *dataSetsHolder.TestDataSet->CatFeatures);
                }
            }
            MATRIXNET_INFO_LOG << "Build permutation independent datasets is done" << Endl;
        }

        void BuildPermutationDependentGpuFeatures(TDataSetsHolder<CatFeaturesStoragePtrType>& dataSetsHolder,
                                                  ui32 permutationId)
        {
            BuildGpuFeatures(dataSetsHolder, false, permutationId, true, false);
        }


        void BuildTestTargetAndIndices(TDataSetsHolder<CatFeaturesStoragePtrType>& dataSetsHolder,
                                       const TCtrTargets<NCudaLib::TMirrorMapping>& ctrsTarget)
        {
            dataSetsHolder.TestDataSet.Reset(new TDataSet<CatFeaturesStoragePtrType>(*LinkedTest, 0, 1 /*blockSize*/));
            dataSetsHolder.TestDataSet->CtrTargets = dataSetsHolder.CtrTargets;
            dataSetsHolder.TestDataSet->LinkedHistoryForCtrs = dataSetsHolder.PermutationDataSets[0];

            auto& dataSet = *dataSetsHolder.TestDataSet;

            const auto testMapping = NCudaLib::TMirrorMapping(ctrsTarget.TestSlice.Size());

            dataSet.Indices.Reset(testMapping);
            MakeSequence(dataSet.Indices);
            dataSet.InverseIndices = dataSet.Indices.CopyView();

            dataSet.Targets = TMirrorBuffer<float>::CopyMapping(dataSet.Indices);
            dataSet.Targets.Write(LinkedTest->GetTargets());

            dataSet.Weights = TMirrorBuffer<float>::CopyMapping(dataSet.Indices);
            dataSet.Weights.Write(LinkedTest->GetWeights());
        }

        void BuildCtrTarget(const yvector<float>& joinedTarget,
                            TCtrTargets<NCudaLib::TMirrorMapping>& ctrsTarget)
        {
            ctrsTarget.BinarizedTarget = BuildBinarizedTarget(FeaturesManager,
                                                              joinedTarget);

            ctrsTarget.WeightedTarget.Reset(NCudaLib::TMirrorMapping(joinedTarget.size()));
            ctrsTarget.Weights.Reset(NCudaLib::TMirrorMapping(joinedTarget.size()));

            ctrsTarget.LearnSlice = TSlice(0, DataProvider.GetSampleCount());
            ctrsTarget.TestSlice = TSlice(DataProvider.GetSampleCount(), joinedTarget.size());

            yvector<float> ctrWeights;
            ctrWeights.resize(joinedTarget.size(), 1.0f);

            yvector<float> ctrWeightedTargets(joinedTarget.begin(), joinedTarget.end());

            double totalWeight = 0;
            for (ui32 i = (ui32) ctrsTarget.LearnSlice.Right; i < ctrWeights.size(); ++i)
            {
                ctrWeights[i] = 0;
            }

            for (ui32 i = 0; i < ctrWeightedTargets.size(); ++i)
            {
                ctrWeightedTargets[i] *= ctrWeights[i];
                totalWeight += ctrWeights[i];
            }

            ctrsTarget.TotalWeight = (float) totalWeight;
            ctrsTarget.WeightedTarget.Write(ctrWeightedTargets);
            ctrsTarget.Weights.Write(ctrWeights);

            CB_ENSURE(ctrsTarget.IsTrivialWeights());
        }

        void BuildCompressedCatFeatures(const TDataProvider& dataProvider,
                                        TCompressedCatFeatureDataSet<CatFeaturesStoragePtrType>& dataset)
        {
            TCompressedCatFeatureDataSetBuilder<CatFeaturesStoragePtrType> builder(dataProvider,
                                                                                   FeaturesManager,
                                                                                   dataset);
            for (ui32 catFeature : FeaturesManager.GetCatFeatureIds())
            {
                if (FeaturesManager.UseForTreeCtr(catFeature))
                {
                    builder.Add(catFeature);
                }
            }
            builder.Finish();
        }

        void ResetSharedDataSets(TDataSet<CatFeaturesStoragePtrType>& dataSet)
        {
            dataSet.PermutationIndependentFeatures.Reset(new TGpuFeatures<>());
            dataSet.CatFeatures.Reset(new TCompressedCatFeatureDataSet<CatFeaturesStoragePtrType>());
        }

        template<class TPolicy>
        inline TPermutationDataSetBuilder<TPolicy>
        CreateBuilder(TDataSetsHolder<CatFeaturesStoragePtrType>& dataSetHolder,
                      ui32 permutationId,
                      bool buildCommon)
        {
            using TBuilder = TPermutationDataSetBuilder<TPolicy>;
            TBuilder binaryFeatureBuilder(FeaturesManager,
                                          DataProvider,
                                          dataSetHolder.PermutationDataSets[permutationId]->Permutation,
                                          dataSetHolder.PermutationDataSets[permutationId]->Indices,
                                          *dataSetHolder.CtrTargets);

            if (LinkedTest && permutationId == 0)
            {
                binaryFeatureBuilder.BuildLinkedTest(*LinkedTest,
                                                     dataSetHolder.TestDataSet->Indices);
            }

            binaryFeatureBuilder.ShuffleFeaturesFlag(ShuffleFeatures);

            if (buildCommon)
            {
                binaryFeatureBuilder
                        .UseFloatFeatures(true)
                        .UseOneHot(true)
                        .UseCtrs(TakePermutationIndependent(FeaturesManager.GetEnabledCtrTypes()));
            } else
            {
                yset<ECtrType> ctrs = TakePermutationDependent(FeaturesManager.GetEnabledCtrTypes());

                binaryFeatureBuilder
                        .UseFloatFeatures(false)
                        .UseOneHot(false)
                        .UseCtrs(ctrs);
            }

            return binaryFeatureBuilder;
        }

    private:
        TBinarizedFeaturesManager& FeaturesManager;
        const TDataProvider& DataProvider;
        const TDataProvider* LinkedTest;
        bool ShuffleFeatures = true;
        ui32 DataProviderPermutationBlockSize = 1;
    };
}
