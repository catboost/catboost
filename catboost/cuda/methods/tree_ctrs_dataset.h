#pragma once

#include <catboost/cuda/data/feature.h>
#include <catboost/cuda/data/binarizations_manager.h>
#include <catboost/cuda/gpu_data/feature_parallel_dataset.h>

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
            return CatFeatures.has(featureId);
        }

        const TSet<ui32>& GetCatFeatures() const {
            return CatFeatures;
        }

        const THashMap<TFeatureTensor, TVector<TCtrConfig>>& GetCtrConfigs() const {
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

        const TVector<TCtrConfig>& GetCtrsConfigsForTensor(const TFeatureTensor& featureTensor) {
            if (CtrConfigs.count(featureTensor) == 0) {
                CtrConfigs[featureTensor] = FeaturesManager.CreateTreeCtrConfigs();
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

        void BuildFeatureIndex() {
            CB_ENSURE(InverseCtrIndex.size() == 0, "Error: build could be done only once");

            for (const ui32 feature : CatFeatures) {
                TFeatureTensor tensor = BaseFeatureTensor;
                tensor.AddCatFeature(feature);
                const auto& configs = GetCtrsConfigsForTensor(tensor);
                for (auto& config : configs) {
                    TCtr ctr;
                    ctr.FeatureTensor = tensor;
                    ctr.Configuration = config;
                    const ui32 idx = static_cast<const ui32>(InverseCtrIndex.size());
                    InverseCtrIndex[ctr] = idx;
                    Ctrs.push_back(ctr);
                    const auto borderCount = FeaturesManager.GetCtrBinarization(ctr).BorderCount;
                    MaxBorderCount = Max<ui32>(MaxBorderCount, borderCount);
                    const ui32 bordersSize = 1 + borderCount;
                    const ui32 offset = static_cast<const ui32>(CtrBorderSlices.size() ? CtrBorderSlices.back().Right
                                                                                       : 0);
                    const TSlice bordersSlice = TSlice(offset, offset + bordersSize);
                    CtrBorderSlices.push_back(bordersSlice);
                }
            }

            TFeaturesMapping featuresMapping = CreateFeaturesMapping();

            auto bordersMapping = featuresMapping.Transform([&](TSlice deviceSlice) {
                ui32 size = 0;
                for (ui32 feature = static_cast<ui32>(deviceSlice.Left); feature < deviceSlice.Right; ++feature) {
                    size += CtrBorderSlices[feature].Size();
                }
                return size;
            });
            CtrBorders.Reset(bordersMapping);

            if (CtrBorderSlices.size()) {
                //borders are so small, that it should be almost always faster to write all border vec then by parts
                TVector<float> borders(CtrBorderSlices.back().Right);
                bool needWrite = false;

                for (ui32 i = 0; i < Ctrs.size(); ++i) {
                    const auto& ctr = Ctrs[i];
                    AreCtrBordersComputed.push_back(false);
                    if (FeaturesManager.IsKnown(ctr)) {
                        const auto& ctrBorders = FeaturesManager.GetBorders(FeaturesManager.GetId(ctr));
                        const ui64 offset = CtrBorderSlices[i].Left;
                        borders[offset] = ctrBorders.size();
                        std::copy(ctrBorders.begin(), ctrBorders.end(), borders.begin() + offset + 1);
                        CB_ENSURE(ctrBorders.size() < CtrBorderSlices[i].Size());
                        AreCtrBordersComputed.back() = true;
                        needWrite = true;
                    }
                }
                if (needWrite) {
                    CtrBorders.Write(borders);
                }
            }
        }

        TFeaturesMapping CreateFeaturesMapping() {
            return NCudaLib::TSingleMapping(BaseTensorIndices.GetMapping().GetDeviceId(),
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
        THashMap<TFeatureTensor, TVector<TCtrConfig>> CtrConfigs; //ctr configs for baseTensor + catFeature

        THolder<TCompressedIndex> CompressedIndex;
        THolder<TScopedCacheHolder> CacheHolder;

        ui32 PermutationKey = 0;
        ui64 MaxBorderCount = 0;

        friend class TTreeCtrDataSetBuilder;

        template <NCudaLib::EPtrType>
        friend class TTreeCtrDataSetsHelper;
    };
}
