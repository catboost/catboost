#include "compute_by_blocks_helper.h"
#include "structure_searcher_options.h"
#include "split_properties_helper.h"
#include <catboost/private/libs/options/bootstrap_options.h>
#include <util/generic/vector.h>

using namespace NCatboostCuda;

namespace {
    /*
     * so for TCFeature we could easily compute dest for histograms:
     * localFirstBinFeature = FirstFoldIndex - FirstBinaryFeatureInGroup
     * hist[FirstBinaryFeatureInGroup * statCount * leafCount +  leafId * groupSize * statCount + (localFirstBinFeature  + binId) + statId * groupSize
     */
    inline ui64 EstimateMaxTempVecsForGather(const TDocParallelDataSet& dataSet,
                                             double sampleRate) {
        const ui32 devCount = NCudaLib::GetCudaManager().GetDeviceCount();
        ui64 docsPerDevices = ::NHelpers::CeilDivide(dataSet.GetTarget().GetWeights().GetObjectsSlice().Size(), devCount);
        docsPerDevices = Max<ui64>(1, docsPerDevices * sampleRate);
        ui64 freeMemoryBytes = static_cast<ui64>(NCudaLib::GetCudaManager().FreeMemoryMb(0) * 1024 * 1024);
        ui64 singleColumnSize = docsPerDevices * sizeof(ui32);

        return Max<ui64>(::NHelpers::CeilDivide(freeMemoryBytes, static_cast<ui64>(1.5 * singleColumnSize)), 1);
    }

    enum EComputeHistSpecialization {
        BinaryFeatures,
        HalfByteFeatures,
        OneByteFeatures5Bit,
        OneByteFeatures6Bit,
        OneByteFeatures7Bit,
        OneByteFeatures8Bit
    };

    inline EFeaturesGroupingPolicy GetGroupingPolicy(EComputeHistSpecialization specialization) {
        switch (specialization) {
            case EComputeHistSpecialization::BinaryFeatures: {
                return EFeaturesGroupingPolicy::BinaryFeatures;
            }
            case EComputeHistSpecialization::HalfByteFeatures: {
                return EFeaturesGroupingPolicy::HalfByteFeatures;
            }
            case EComputeHistSpecialization::OneByteFeatures5Bit:
            case EComputeHistSpecialization::OneByteFeatures6Bit:
            case EComputeHistSpecialization::OneByteFeatures7Bit:
            case EComputeHistSpecialization::OneByteFeatures8Bit: {
                return EFeaturesGroupingPolicy::OneByteFeatures;
            }
            default: {
                CB_ENSURE(false, "Unknown specialization");
            }
        }
    }

    static inline EComputeHistSpecialization GetHistogramSpecialization(EFeaturesGroupingPolicy policy, int maxFolds) {
        switch (policy) {
            case EFeaturesGroupingPolicy::BinaryFeatures: {
                return EComputeHistSpecialization::BinaryFeatures;
            }
            case EFeaturesGroupingPolicy::HalfByteFeatures: {
                return EComputeHistSpecialization::HalfByteFeatures;
            }
            default: {
                CB_ENSURE(policy == EFeaturesGroupingPolicy::OneByteFeatures, "Unknown specialization");
                if (maxFolds <= 32) {
                    return EComputeHistSpecialization::OneByteFeatures5Bit;
                } else if (maxFolds <= 64) {
                    return EComputeHistSpecialization::OneByteFeatures6Bit;
                } else if (maxFolds <= 128) {
                    return EComputeHistSpecialization::OneByteFeatures7Bit;
                } else {
                    return EComputeHistSpecialization::OneByteFeatures8Bit;
                }
            }
        }
    }

    struct TPolicyFeatures {
        TVector<NCudaLib::TDistributedObject<TCFeature>> Features;
        TVector<ui32> FeaturesIds;
        TMaybe<EComputeHistSpecialization> Policy;
    };

    class TComputeSplitPropsGroupsBuilder {
    public:
        void AddFeatures(EFeaturesGroupingPolicy policy,
                         const TStripeBuffer<TCFeature>& features,
                         TVector<ui32> featureIds) {
            auto featuresCpu = NCudaLib::ReadToDistributedObjectVec(features);
            const ui64 groupSize = GetFeaturesPerInt(policy);
            CB_ENSURE(featureIds.size() == featuresCpu.size());

            const ui32 alignedSize = static_cast<const ui32>(NHelpers::CeilDivide(featureIds.size(), groupSize) * groupSize);
            const ui32 devCount = static_cast<const ui32>(NCudaLib::GetCudaManager().GetDeviceCount());

            for (ui32 f = static_cast<ui32>(featureIds.size()); f < alignedSize; ++f) {
                featureIds.push_back(static_cast<ui32>(-1));
                NCudaLib::TDistributedObject<TCFeature> emptyFeature = NCudaLib::GetCudaManager().CreateDistributedObject<TCFeature>();
                for (ui32 dev = 0; dev < devCount; ++dev) {
                    TCFeature baseFeature = featuresCpu.back().At(dev);
                    baseFeature.Folds = 0;
                    baseFeature.OneHotFeature = false;
                    baseFeature.Mask = 0;
                    baseFeature.Shift = GetShift(policy, f);
                    emptyFeature.Set(dev, baseFeature);
                }
                featuresCpu.push_back(emptyFeature);
            }

            //here everything aligned
            for (size_t f = 0; f < featureIds.size(); f += groupSize) {
                ui32 maxFolds = 0;
                const ui32 end = Min<ui32>(featureIds.size(), f + groupSize);
                for (size_t i = f; i < end; ++i) {
                    maxFolds = Max<ui32>(maxFolds, featuresCpu[i].At(0).Folds);
                }
                EComputeHistSpecialization specialization = GetHistogramSpecialization(policy, maxFolds);
                Features[specialization].FeaturesIds.insert(Features[specialization].FeaturesIds.end(),
                                                            featureIds.begin() + f,
                                                            featureIds.begin() + end);
                Features[specialization].Features.insert(Features[specialization].Features.end(),
                                                         featuresCpu.begin() + f,
                                                         featuresCpu.begin() + end);
            }
        }

        /*
         * Two-pass grouping. On first we add fake features to align
         * On second we build groups to compute, thus we could combine features from different "datasets" in one compute group
         * this pass just splits everything, no reduce-scatter and other stuff builiding
         */
        TVector<TPolicyFeatures> Build(ui32 optimalSingleStreamGroupSize, bool forceOneBlock) {
            TVector<TPolicyFeatures> groups;

            for (auto& computePolicy : Features) {
                const EComputeHistSpecialization policy = computePolicy.first;
                auto& allPolicyFeatures = computePolicy.second;

                allPolicyFeatures.Policy = policy; /* lazy set it, n */

                const ui32 featuresPerInt = GetFeaturesPerInt(GetGroupingPolicy(policy));
                const ui32 featureCount = static_cast<ui32>(allPolicyFeatures.FeaturesIds.size());
                CB_ENSURE(featureCount % featuresPerInt == 0, "Should be already aligned");

                const ui32 columnCount = static_cast<const ui32>(::NHelpers::CeilDivide(allPolicyFeatures.FeaturesIds.size(), featuresPerInt));
                const ui32 groupCount = forceOneBlock ? 1 : ::NHelpers::CeilDivide(columnCount, optimalSingleStreamGroupSize);
                const ui32 featuresPerGroup = ::NHelpers::CeilDivide(columnCount, groupCount) * featuresPerInt;

                for (ui32 firstFeature = 0; firstFeature < featureCount; firstFeature += featuresPerGroup) {
                    groups.push_back(TPolicyFeatures());
                    auto& group = groups.back();
                    const ui32 lastFeature = Min<ui32>(firstFeature + featuresPerGroup,
                                                       featureCount);

                    group.Policy = policy;
                    group.FeaturesIds.insert(group.FeaturesIds.end(),
                                             allPolicyFeatures.FeaturesIds.begin() + firstFeature,
                                             allPolicyFeatures.FeaturesIds.begin() + lastFeature);

                    group.Features.insert(group.Features.end(),
                                          allPolicyFeatures.Features.begin() + firstFeature,
                                          allPolicyFeatures.Features.begin() + lastFeature);
                }
            }
            return groups;
        }

    private:
        TMap<EComputeHistSpecialization, TPolicyFeatures> Features;
    };

    struct TReduceScatterGroupsBuilder {
        TReduceScatterGroupsBuilder() {
        }

        void AddGroup(const TPolicyFeatures& group) {
            const ui32 featureCount = static_cast<const ui32>(group.FeaturesIds.size());
            {
                const ui32 currentFeatureCount = static_cast<const ui32>(Slices.size() ? Slices.back().Right : 0);
                Slices.push_back(TSlice(currentFeatureCount, currentFeatureCount + featureCount));
            }

            const ui32 devCount = static_cast<const ui32>(NCudaLib::GetCudaManager().GetDeviceCount());
            const ui32 featuresPerDevice = NHelpers::CeilDivide(featureCount, devCount);

            {
                auto currentOffsets = NCudaLib::GetCudaManager().CreateDistributedObject<ui32>(0u);
                for (ui32 dev = 0; dev < devCount; ++dev) {
                    currentOffsets.Set(dev, static_cast<ui32>(BinFeaturesBuilder.GetCurrentSize(dev)));
                }
                GroupBinFeatureOffsets.push_back(currentOffsets);
            }

            /* we have A B A B  before reduce
             * After reduce  A B
             * this is sizeof(A) sizeof(B) */
            auto binFeatureCountAfterReduceOnDevices = NCudaLib::GetCudaManager().CreateDistributedObject<ui32>(0u);

            /* we have A B A B  before reduce
             * After reduce  A B
             * this is 0 sizeof(A)
             * so we could for block B obtain offset, where we should write results before reduce
             * */
            auto groupOffsetsBeforeReduce = NCudaLib::GetCudaManager().CreateDistributedObject<ui32>(0u);

            for (ui32 dev = 0; dev < devCount; ++dev) {
                ui32 firstFeature = Min<ui32>(dev * featuresPerDevice, featureCount);
                ui32 lastFeature = Min<ui32>((dev + 1) * featuresPerDevice, featureCount);

                for (ui32 feature = firstFeature; feature < lastFeature; ++feature) {
                    const auto& origFeature = group.Features[feature];
                    const ui32 folds = origFeature.At(dev).Folds;
                    const ui32 featureId = group.FeaturesIds[feature];

                    TBinarizedFeature binarizedFeature;
                    binarizedFeature.Folds = folds;
                    binarizedFeature.OneHotFeature = origFeature.At(dev).OneHotFeature;
                    binarizedFeature.FirstFoldIndex = static_cast<ui32>(BinFeaturesBuilder.GetCurrentSize(dev));
                    BinarizedFeaturesBuilder.Add(dev, binarizedFeature);

                    for (ui32 fold = 0; fold < folds; ++fold) {
                        TCBinFeature binFeature;
                        binFeature.FeatureId = featureId;
                        binFeature.BinId = fold;
                        BinFeaturesBuilder.Add(dev, binFeature);
                        binFeatureCountAfterReduceOnDevices[dev]++;
                    }
                }
            }

            {
                auto afterReduceMapping = NCudaLib::TStripeMapping::CreateFromSizes(binFeatureCountAfterReduceOnDevices);
                AfterReduceMapping.push_back(afterReduceMapping);
                BeforeReduceMapping.push_back(NCudaLib::TStripeMapping::RepeatOnAllDevices(afterReduceMapping.GetObjectsSlice().Size()));
            }

            GroupBinFeatureSizes.push_back(binFeatureCountAfterReduceOnDevices);

            for (ui32 dev = 1; dev < devCount; ++dev) {
                groupOffsetsBeforeReduce[dev] = groupOffsetsBeforeReduce[dev - 1] + binFeatureCountAfterReduceOnDevices[dev - 1];
            }

            TVector<ui32> devIdsAftreReduceForFeatures(featureCount);
            for (ui32 dev = 0; dev < devCount; ++dev) {
                ui32 firstFeature = Min<ui32>(dev * featuresPerDevice, featureCount);
                ui32 lastFeature = Min<ui32>((dev + 1) * featuresPerDevice, featureCount);
                std::fill(devIdsAftreReduceForFeatures.begin() + firstFeature,
                          devIdsAftreReduceForFeatures.begin() + lastFeature,
                          dev);
            }

            auto foldOffsetsInGroup = CreateDistributedObject<ui32>(0);
            ui32 maxFolds = 0;

            for (ui32 f = 0; f < featureCount; ++f) {
                const ui32 devAfterReduce = devIdsAftreReduceForFeatures[f];
                const auto& origFeature = group.Features[f];

                const ui32 foldOffset = foldOffsetsInGroup[devAfterReduce];
                const ui32 folds = static_cast<const ui32>(origFeature.At(0).Folds);
                auto feature = CreateDistributedObject<TFeatureInBlock>();

                for (ui32 dev = 0; dev < devCount; ++dev) {
                    CB_ENSURE(
                        origFeature.At(dev).Folds == folds,
                        "Unexpected number of folds " << origFeature.At(dev).Folds << " at device " << dev
                        << " (should be " << folds << ")");

                    TFeatureInBlock devFeature;
                    devFeature.CompressedIndexOffset = origFeature.At(dev).Offset;
                    devFeature.Folds = origFeature.At(dev).Folds;
                    devFeature.FoldOffsetInGroup = foldOffset;
                    devFeature.GroupOffset = groupOffsetsBeforeReduce.At(devAfterReduce);
                    devFeature.GroupSize = binFeatureCountAfterReduceOnDevices.At(devAfterReduce);
                    feature.Set(dev, devFeature);
                }

                foldOffsetsInGroup[devAfterReduce] += folds;
                maxFolds = Max<ui32>(maxFolds, folds);
                FeaturesBuilder.Add(feature);
            }

            Policies.push_back(GetGroupingPolicy(*group.Policy));
            MaxFoldsInGroup.push_back(maxFolds);
        }

        NCudaLib::TParallelStripeVectorBuilder<TFeatureInBlock> FeaturesBuilder;
        TVector<TSlice> Slices;

        TVector<EFeaturesGroupingPolicy> Policies;
        TVector<ui32> MaxFoldsInGroup;
        //this ones are mapping after reduce
        //we'll reduce-scatter features, then we need to put everything in flat index
        //memory copy could be done 'as is' for each leaf, we just need to select write offset (and size so we could compute memory for single leaf)

        TVector<NCudaLib::TDistributedObject<ui32>> GroupBinFeatureOffsets;
        TVector<NCudaLib::TDistributedObject<ui32>> GroupBinFeatureSizes;

        TVector<NCudaLib::TStripeMapping> BeforeReduceMapping;
        TVector<NCudaLib::TStripeMapping> AfterReduceMapping;

        //this are for argmax. After all was calculated we need to scan + update histgorams, then select best splits
        //this for scans
        NCudaLib::TStripeVectorBuilder<TCBinFeature> BinFeaturesBuilder;
        NCudaLib::TStripeVectorBuilder<TBinarizedFeature> BinarizedFeaturesBuilder;
    };
}

void NCatboostCuda::TComputeSplitPropertiesByBlocksHelper::Rebuild(const TComputeByBlocksConfig& splitPropsConfig) {
    //< 1GB for 2 workings streams
    const ui32 optimalGroupSize = Min<ui32>(32, (ui32)EstimateMaxTempVecsForGather(DataSet, splitPropsConfig.SampleRate) / StreamsCount);
    CATBOOST_DEBUG_LOG << "Estimate group size for compute histograms " << optimalGroupSize << Endl;
    CB_ENSURE(optimalGroupSize >= 4, "Error: not enough memory for learning");

    TVector<TPolicyFeatures> groups;
    {
        TComputeSplitPropsGroupsBuilder groupsBuilder;
        for (EFeaturesGroupingPolicy policy : GetEnumAllValues<NCatboostCuda::EFeaturesGroupingPolicy>()) {
            if (DataSet.HasFeatures()) {
                const auto& features = DataSet.GetFeatures();
                if (features.HasFeaturesForPolicy(policy)) {
                    groupsBuilder.AddFeatures(policy,
                                              features.GetGrid(policy),
                                              features.GetCpuGrid(policy).FeatureIds);
                }
            }
            if (DataSet.HasPermutationDependentFeatures()) {
                const auto& features = DataSet.GetPermutationFeatures();
                if (features.HasFeaturesForPolicy(policy)) {
                    groupsBuilder.AddFeatures(policy,
                                              features.GetGrid(policy),
                                              features.GetCpuGrid(policy).FeatureIds);
                }
            }
        }
        groups = groupsBuilder.Build(optimalGroupSize, splitPropsConfig.ForceOneBlockPerPolicy);
    }

    TReduceScatterGroupsBuilder reduceScatterGroupsBuilder;
    for (auto& group : groups) {
        reduceScatterGroupsBuilder.AddGroup(group);
    }

    reduceScatterGroupsBuilder.FeaturesBuilder.Build<NCudaLib::EPtrType::CudaDevice>(Features);
    BlockSlices.clear();
    BlockSlices = reduceScatterGroupsBuilder.Slices;
    BlockPolicies = reduceScatterGroupsBuilder.Policies;
    MaxFolds = reduceScatterGroupsBuilder.MaxFoldsInGroup;

    BeforeReduceBinFeaturesMappings = reduceScatterGroupsBuilder.BeforeReduceMapping;
    AfterReduceBinFeaturesMappings = reduceScatterGroupsBuilder.AfterReduceMapping;

    WriteOffsets = reduceScatterGroupsBuilder.GroupBinFeatureOffsets;
    WriteSizes = reduceScatterGroupsBuilder.GroupBinFeatureSizes;

    reduceScatterGroupsBuilder.BinarizedFeaturesBuilder.Build(BinarizedFeatures);
    reduceScatterGroupsBuilder.BinFeaturesBuilder.Build(BinFeatures);

    CATBOOST_DEBUG_LOG << "Compute blocks:" << Endl;
    for (ui32 i = 0; i < BlockPolicies.size(); ++i) {
        CATBOOST_DEBUG_LOG << BlockPolicies[i] << " " << BlockSlices[i] << Endl;
    }
}

NCudaLib::TDistributedObject<ui32> TComputeSplitPropertiesByBlocksHelper::BinFeatureCount() const {
    auto result = CreateDistributedObject<ui32>(0);
    for (ui32 dev = 0; dev < result.DeviceCount(); ++dev) {
        result[dev] = BinFeatures.GetMapping().DeviceSlice(dev).Size();
    }
    return result;
}

inline double GetTakenFraction(const NCatboostOptions::TBootstrapConfig& options) {
    if (AreZeroWeightsAfterBootstrap(options.GetBootstrapType())) {
        return options.GetTakenFraction();
    }
    return 1.0;
}

TComputeSplitPropertiesByBlocksHelper& NCatboostCuda::GetComputeByBlocksHelper(const TDocParallelDataSet& dataSet,
                                                                               const TTreeStructureSearcherOptions& options,
                                                                               ui32 statCount) {
    TComputeByBlocksConfig config;
    config.SampleRate = GetTakenFraction(options.BootstrapOptions);
    config.ForceOneBlockPerPolicy = statCount <= 2;
    config.StreamCount = statCount <= 2 ? 1 : 3;

    return *dataSet.Cache(config, [&]() -> THolder<TComputeSplitPropertiesByBlocksHelper> {
        return MakeHolder<TComputeSplitPropertiesByBlocksHelper>(dataSet, config);
    });
}
