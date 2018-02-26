#pragma once

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/data/data_provider.h>
#include <catboost/cuda/data/binarizations_manager.h>
#include <catboost/cuda/ctrs/ctr_bins_builder.h>
#include <catboost/cuda/cuda_lib/cuda_buffer_helpers/buffer_resharding.h>
#include <util/generic/noncopyable.h>
namespace NCatboostCuda {
    inline ui64 EstimatePerDeviceMemoryUsageForCatFeaturesDataSet(const TDataProvider& dataProvider,
                                                                  const TBinarizedFeaturesManager& featuresManager) {
        ui32 maxUniqueValue = 0;
        for (auto feature : featuresManager.GetCatFeatureIds()) {
            maxUniqueValue = std::max<ui32>(maxUniqueValue, featuresManager.GetBinCount(feature));
        }
        return static_cast<ui64>(
            CompressedSize<ui64>(static_cast<ui32>(dataProvider.GetSampleCount()), maxUniqueValue) * 1.0 *
            featuresManager.GetCatFeatureIds().size() / NCudaLib::GetCudaManager().GetDeviceCount());
    }

    template <NCudaLib::EPtrType StorageType = NCudaLib::EPtrType::CudaDevice>
    class TCompressedCatFeatureDataSet: public TMoveOnly,
                                         public TGuidHolder {
    public:
        using TCompressedCatFeatureVec = NCudaLib::TCudaBuffer<ui64, NCudaLib::TSingleMapping, StorageType>;

        ui64 GetDocCount() const {
            return DataProvider->GetTargets().size();
        }

        ui32 GetFeatureCount() const {
            return (ui32)CompressedCatIndex.size();
        }

        ui32 GetFeatureCount(ui32 devId) const {
            CB_ENSURE(devId < DeviceFeatures.size(), "Error: " << GetFeatureCount() << " " << DeviceFeatures.size() << "/" << devId);
            return static_cast<ui32>(DeviceFeatures[devId].size());
        }

        const TVector<ui32>& GetDeviceFeatures(ui32 devId) const {
            return DeviceFeatures[devId];
        }

        ui32 UniqueValues(ui32 featureId) const {
            return Features.at(featureId).UniqueValues;
        }

        const TCompressedCatFeatureVec& GetFeature(ui32 featureId) const {
            const ui32 localId = Features.at(featureId).LocalIndex;
            return CompressedCatIndex.at(localId);
        }

    private:
        struct TCatFeature {
            ui32 LocalIndex;
            ui32 UniqueValues;
        };

    private:
        TVector<TCompressedCatFeatureVec> CompressedCatIndex;
        TVector<TVector<ui32>> DeviceFeatures;
        TMap<ui32, TCatFeature> Features;
        const TDataProvider* DataProvider = nullptr;

        template <NCudaLib::EPtrType Type>
        friend class TCompressedCatFeatureDataSetBuilder;
    };

    template <NCudaLib::EPtrType StorageType = NCudaLib::EPtrType::CudaDevice>
    class TCompressedCatFeatureDataSetBuilder {
    public:
        TCompressedCatFeatureDataSetBuilder(const TDataProvider& dataProvider,
                                            TBinarizedFeaturesManager& featuresManager,
                                            TCompressedCatFeatureDataSet<StorageType>& dataSet)
            : DevCount(GetDeviceCount())
            , DataSet(dataSet)
            , DataProvider(dataProvider)
            , FeaturesManager(featuresManager)
        {
            MemoryUsage.resize(DevCount, 0);
            DataSet.DataProvider = &DataProvider;
            DataSet.DeviceFeatures.resize(DevCount);
        }

        TCompressedCatFeatureDataSetBuilder& Add(ui32 featureId) {
            const ui32 dataProviderId = FeaturesManager.GetDataProviderId(featureId);
            const auto& catFeature = dynamic_cast<const ICatFeatureValuesHolder&>(DataProvider.GetFeatureById(
                dataProviderId));
            const ui64 docCount = catFeature.GetSize();

            auto uncompressedCatFeature = catFeature.ExtractValues();
            TSingleBuffer<ui32> tmp = TSingleBuffer<ui32>::Create(
                NCudaLib::TSingleMapping(DeviceId, uncompressedCatFeature.size()));
            tmp.Write(uncompressedCatFeature);
            const auto uniqueValues = FeaturesManager.GetBinCount(featureId);
            const auto compressedSize = CompressedSize<ui64>((ui32)docCount, uniqueValues);
            auto compressedMapping = NCudaLib::TSingleMapping(DeviceId, compressedSize);

            auto& catIndex = DataSet.CompressedCatIndex;
            DataSet.Features[featureId] = {static_cast<ui32>(catIndex.size()), uniqueValues};
            catIndex.push_back(TCudaBuffer<ui64, NCudaLib::TSingleMapping, StorageType>::Create(compressedMapping));
            Compress(tmp, catIndex.back(), uniqueValues);
            MemoryUsage[DeviceId] += compressedMapping.MemorySize();
            DataSet.DeviceFeatures[DeviceId].push_back(featureId);

            DeviceId = (DeviceId + 1) % DevCount;
            return *this;
        }

        void Finish() {
            CB_ENSURE(!BuildDone, "Error: build could be done only once");
            MATRIXNET_INFO_LOG << "Build catFeatures compressed dataset "
                               << "for "
                               << DataSet.GetFeatureCount() << " features and " << DataSet.GetDocCount() << " documents"
                               << Endl;

            for (ui32 dev = 0; dev < DevCount; ++dev) {
                MATRIXNET_INFO_LOG
                    << "Memory usage at #" << dev << ": " << sizeof(ui64) * MemoryUsage[dev] * 1.0 / 1024 / 1024 << "MB"
                    << Endl;
            }
            BuildDone = true;
        }

    private:
        ui32 DevCount;
        TCompressedCatFeatureDataSet<StorageType>& DataSet;
        bool BuildDone = false;
        ui32 DeviceId = 0;
        TVector<ui64> MemoryUsage;

        const TDataProvider& DataProvider;
        const TBinarizedFeaturesManager& FeaturesManager;
    };

    template <class TValue>
    class TLazyStreamValue: public TMoveOnly {
    public:
        explicit TLazyStreamValue(TValue&& value,
                                  ui32 stream = 0)
            : Stream(stream)
            , Value(std::move(value))
        {
        }

        TLazyStreamValue() {
        }

        const TValue& Get() const {
            if (!IsSynced) {
                auto& manager = NCudaLib::GetCudaManager();
                manager.WaitComplete();
                IsSynced = true;
            }
            return Value;
        }

        const TValue& GetInStream(ui32 stream) const {
            if (Stream == stream) {
                return Value;
            } else {
                return Get();
            }
        }

    private:
        ui32 Stream = 0;
        mutable bool IsSynced = false;
        TValue Value;
    };

    template <NCudaLib::EPtrType CatFeatureStorageType = NCudaLib::EPtrType::CudaDevice>
    class TMirrorCatFeatureProvider: public TNonCopyable {
    public:
        TMirrorCatFeatureProvider(const TCompressedCatFeatureDataSet<CatFeatureStorageType>& dataSet,
                                  TScopedCacheHolder& cache)
            : Src(dataSet)
            , ScopedCache(cache)
        {
        }

        const TLazyStreamValue<TMirrorBuffer<ui64>>& BroadcastFeature(ui32 featureId,
                                                                      ui32 builderStream = 0) {
            return ScopedCache.Cache(Src, featureId, [&]() -> TLazyStreamValue<TMirrorBuffer<ui64>> {
                auto& src = Src.GetFeature(featureId);
                auto mapping = NCudaLib::TMirrorMapping(CompressedSize<ui64>((ui32)Src.GetDocCount(),
                                                                             Src.UniqueValues(featureId)));
                TMirrorBuffer<ui64> dst = TMirrorBuffer<ui64>::Create(mapping);
                NCudaLib::Reshard(src, dst, builderStream);
                return TLazyStreamValue<TMirrorBuffer<ui64>>(std::move(dst),
                                                             builderStream);
            });
        }

        const TMirrorBuffer<ui64>& GetFeature(ui32 featureId,
                                              ui32 builderStream = 0) {
            return BroadcastFeature(featureId, builderStream).Get();
        }

    private:
        const TCompressedCatFeatureDataSet<CatFeatureStorageType>& Src;
        TScopedCacheHolder& ScopedCache;
    };
}
