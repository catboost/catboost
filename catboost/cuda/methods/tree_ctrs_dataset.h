#pragma once

#include <catboost/cuda/data/feature.h>
#include <catboost/cuda/data/binarizations_manager.h>
#include <catboost/cuda/gpu_data/feature_parallel_dataset.h>

#include <catboost/private/libs/ctr_description/ctr_config.h>

#include <util/generic/map.h>
#include <util/generic/hash.h>
#include <util/generic/set.h>

namespace NCatboostCuda {
    /*
     * TreeCtrs dataSet are cached based on baseTensor, from which they we generated
    * If we don't have enough gpu-ram, then cache in batchs (for one baseTensor generate several dataSets with ctrs)
    * TreeCtrs dataSets always for catFeature i stores all ctrs with this catFeature - perFeatures batch instead of perCtr
    */
    class TTreeCtrDataSet: public TGuidHolder {
    public:
        using TCompressedIndex = TSharedCompressedIndex<TSingleDevLayout>;
        using TFeaturesMapping = typename TSingleDevLayout::TFeaturesMapping;
        using TSamplesMapping = typename TSingleDevLayout::TSamplesMapping;
        using TVec = TCudaBuffer<float, TFeaturesMapping>;
        using TCompressedIndexMapping = typename TSingleDevLayout::TCompressedIndexMapping;

    public:
        template <class TUi32>
        TTreeCtrDataSet(const TBinarizedFeaturesManager& featuresManager,
                        const TFeatureTensor& baseTensor,
                        const TCudaBuffer<TUi32, TSamplesMapping>& baseTensorIndices)
            : FeaturesManager(featuresManager)
            , BaseFeatureTensor(baseTensor)
            , BaseTensorIndices(baseTensorIndices.ConstCopyView())
            , CacheHolder(new TScopedCacheHolder)
        {
        }

        const TVector<TCtr>& GetCtrs() const {
            return Ctrs;
        }

        const TCtr& GetCtr(ui32 featureId) const {
            return Ctrs[featureId];
        }

        TSingleBuffer<float> GetCtrWeights(ui32 maxUniqueValues, float modelSizeReg) const {
            TVector<float> featureWeights;
            for (const auto& ctr: Ctrs) {
                featureWeights.push_back(pow(
                    1 + float(FeaturesManager.GetMaxCtrUniqueValues(ctr)) / maxUniqueValues,
                    -modelSizeReg
                ));
            }

            auto mapping = NCudaLib::TSingleMapping(GetDeviceId(), GetFeatureCount());
            auto featureWeightsBuffer = TSingleBuffer<float>::Create(mapping);
            featureWeightsBuffer.Write(featureWeights);
            return featureWeightsBuffer;
        }

        TScopedCacheHolder& GetCacheHolder() const {
            CB_ENSURE(CacheHolder);
            return *CacheHolder;
        }

        bool HasCompressedIndex() const {
            return CompressedIndex != nullptr && CompressedIndex->GetStorage().GetObjectsSlice().Size();
        }

        const TCompressedIndex::TCompressedDataSet& GetCompressedDataSet() const {
            CB_ENSURE(CompressedIndex != nullptr);
            return CompressedIndex->GetDataSet(0);
        }

        ui32 GetFeatureCount() const {
            return Ctrs.size();
        }

        ui32 GetDeviceId() const {
            return BaseTensorIndices.GetMapping().GetDeviceId();
        }

        ui32 GetCompressedIndexPermutationKey() const {
            return PermutationKey;
        }

        TMap<TCtr, TVector<float>> ReadBorders(const TVector<ui32>& ids) const {
            TVector<float> allBorders;
            CtrBorders.Read(allBorders);
            TMap<TCtr, TVector<float>> result;

            for (auto id : ids) {
                TSlice readSlice = CtrBorderSlices[id];
                result[Ctrs[id]] = ExtractBorders(allBorders.data() + readSlice.Left);
            }
            return result;
        };

        TVector<float> ReadBorders(const ui32 featureId) const {
            TVector<float> borders;
            TSlice readSlice = CtrBorderSlices[featureId];
            CtrBorders.CreateReader().SetReadSlice(readSlice).Read(borders);
            return ExtractBorders(borders.data());
        }

        const TFeatureTensor& GetBaseTensor() const {
            return BaseFeatureTensor;
        }

        const TCudaBuffer<const ui32, TSamplesMapping>& GetBaseTensorIndices() const {
            return BaseTensorIndices;
        }

        void SetPermutationKey(ui32 permutationKey) {
            PermutationKey = permutationKey;
        }

        bool HasCatFeature(ui32 featureId) const {
            return CatFeatures.contains(featureId);
        }

        const TSet<ui32>& GetCatFeatures() const {
            return CatFeatures;
        }

        const THashMap<TFeatureTensor, TVector<NCB::TCtrConfig>>& GetCtrConfigs() const {
            return CtrConfigs;
        }

        //not the best place btw
        ui32 GetMaxFeaturesPerInt() const {
            if (MaxBorderCount <= TCompressedIndexHelper<EFeaturesGroupingPolicy::BinaryFeatures>::MaxFolds()) {
                return TCompressedIndexHelper<EFeaturesGroupingPolicy::BinaryFeatures>::FeaturesPerInt();
            } else if (MaxBorderCount <= TCompressedIndexHelper<EFeaturesGroupingPolicy::HalfByteFeatures>::MaxFolds()) {
                return TCompressedIndexHelper<EFeaturesGroupingPolicy::HalfByteFeatures>::FeaturesPerInt();
            } else {
                CB_ENSURE(MaxBorderCount <= TCompressedIndexHelper<EFeaturesGroupingPolicy::OneByteFeatures>::MaxFolds());
                return TCompressedIndexHelper<EFeaturesGroupingPolicy::OneByteFeatures>::FeaturesPerInt();
            }
        }

    private:
        TVector<float> ExtractBorders(const float* bordersAndSize) const {
            const ui32 borderCount = static_cast<ui32>(bordersAndSize[0]);
            TVector<float> borders(borderCount);
            for (ui32 i = 0; i < borderCount; ++i) {
                borders[i] = bordersAndSize[i + 1];
            }
            return borders;
        }

        const TVector<NCB::TCtrConfig>& GetCtrsConfigsForTensor(const TFeatureTensor& featureTensor) {
            if (CtrConfigs.count(featureTensor) == 0) {
                CtrConfigs[featureTensor] = FeaturesManager.CreateTreeCtrConfigs(ETaskType::GPU);
            }
            return CtrConfigs[featureTensor];
        }

        void AddCatFeature(const ui32 catFeature) {
            {
                TFeatureTensor tensor = BaseFeatureTensor;
                tensor.AddCatFeature(catFeature);
                CB_ENSURE(tensor != BaseFeatureTensor, "Error: expect new tensor");
            }
            CatFeatures.insert(catFeature);
        }

        void BuildFeatureIndex();

        TFeaturesMapping CreateFeaturesMapping() {
            return NCudaLib::TSingleMapping(GetDeviceId(),
                                            static_cast<ui32>(Ctrs.size()));
        }

        ui32 GetDevice(const TFeaturesMapping& featuresMapping, ui32 featureId) {
            for (auto& dev : featuresMapping.NonEmptyDevices()) {
                if (featuresMapping.DeviceSlice(dev).Contains(TSlice(featureId))) {
                    return dev;
                }
            }
            CB_ENSURE(false, "Error: featuresId is out of range");
            return 0;
        }

    private:
        const TBinarizedFeaturesManager& FeaturesManager;

        TFeatureTensor BaseFeatureTensor;
        TCudaBuffer<const ui32, TSamplesMapping> BaseTensorIndices;
        TSet<ui32> CatFeatures;

        THashMap<TCtr, ui32> InverseCtrIndex;
        TVector<TCtr> Ctrs;
        TVector<TSlice> CtrBorderSlices;
        TCudaBuffer<float, TFeaturesMapping> CtrBorders;
        TVector<bool> AreCtrBordersComputed;
        THashMap<TFeatureTensor, TVector<NCB::TCtrConfig>> CtrConfigs; //ctr configs for baseTensor + catFeature

        THolder<TCompressedIndex> CompressedIndex;
        THolder<TScopedCacheHolder> CacheHolder;

        ui32 PermutationKey = 0;
        ui64 MaxBorderCount = 0;

        friend class TTreeCtrDataSetBuilder;

        friend class TTreeCtrDataSetsHelper;
    };
}
