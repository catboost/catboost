#pragma once

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/data/binarizations_manager.h>
#include <catboost/cuda/ctrs/ctr_bins_builder.h>
#include <catboost/cuda/cuda_lib/cuda_buffer_helpers/buffer_resharding.h>

#include <catboost/libs/data/data_provider.h>

#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/generic/noncopyable.h>

namespace NCatboostCuda {
    inline ui64 EstimatePerDeviceMemoryUsageForCatFeaturesDataSet(const NCB::TTrainingDataProvider& dataProvider,
                                                                  const TBinarizedFeaturesManager& featuresManager) {
        ui32 maxUniqueValue = 0;
        for (auto feature : featuresManager.GetCatFeatureIds()) {
            maxUniqueValue = std::max<ui32>(maxUniqueValue, featuresManager.GetBinCount(feature));
        }
        return static_cast<ui64>(
            CompressedSize<ui64>(static_cast<ui32>(dataProvider.GetObjectCount()), maxUniqueValue) * 1.0 *
            featuresManager.GetCatFeatureIds().size() / NCudaLib::GetCudaManager().GetDeviceCount());
    }

    class TCompressedCatFeatureDataSet: public TMoveOnly,
                                         public TGuidHolder {
    public:
        explicit TCompressedCatFeatureDataSet(EGpuCatFeaturesStorage type)
            : StorageType(type)
        {
        }

        template <NCudaLib::EPtrType StorageType>
        using TCompressedCatFeatureVec = NCudaLib::TCudaBuffer<ui64, NCudaLib::TSingleMapping, StorageType>;

        using TCompressedCatFeatureVecCpu = TCompressedCatFeatureVec<NCudaLib::EPtrType::CudaHost>;
        using TCompressedCatFeatureVecGpu = TCompressedCatFeatureVec<NCudaLib::EPtrType::CudaDevice>;

        ui64 GetDocCount() const {
            return DataProvider->GetObjectCount();
        }

        ui32 GetFeatureCount() const {
            return static_cast<ui32>(StorageType == EGpuCatFeaturesStorage::GpuRam ? CompressedCatIndexGpu.size() : CompressedCatIndexCpu.size());
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

        const TCompressedCatFeatureVec<NCudaLib::EPtrType::CudaDevice>& GetFeatureGpu(ui32 featureId) const {
            CB_ENSURE(StorageType == EGpuCatFeaturesStorage::GpuRam);
            const ui32 localId = Features.at(featureId).LocalIndex;
            return CompressedCatIndexGpu.at(localId);
        }

        const TCompressedCatFeatureVec<NCudaLib::EPtrType::CudaHost>& GetFeatureCpu(ui32 featureId) const {
            CB_ENSURE(StorageType == EGpuCatFeaturesStorage::CpuPinnedMemory);
            const ui32 localId = Features.at(featureId).LocalIndex;
            return CompressedCatIndexCpu.at(localId);
        }

        EGpuCatFeaturesStorage GetStorageType() const {
            return StorageType;
        }

    private:
        struct TCatFeature {
            ui32 LocalIndex;
            ui32 UniqueValues;
        };

    private:
        TVector<TCompressedCatFeatureVecCpu> CompressedCatIndexCpu;
        TVector<TCompressedCatFeatureVecGpu> CompressedCatIndexGpu;
        EGpuCatFeaturesStorage StorageType;

        TVector<TVector<ui32>> DeviceFeatures;
        TMap<ui32, TCatFeature> Features;

        const NCB::TTrainingDataProvider* DataProvider = nullptr;

        friend class TCompressedCatFeatureDataSetBuilder;
    };

    class TCompressedCatFeatureDataSetBuilder {
    public:
        TCompressedCatFeatureDataSetBuilder(const NCB::TTrainingDataProvider& dataProvider,
                                            TBinarizedFeaturesManager& featuresManager,
                                            TCompressedCatFeatureDataSet& dataSet,
                                            NPar::ILocalExecutor* localExecutor)
            : DevCount(GetDeviceCount())
            , DataSet(dataSet)
            , DataProvider(dataProvider)
            , FeaturesManager(featuresManager)
            , LocalExecutor(localExecutor)
        {
            MemoryUsage.resize(DevCount, 0);
            DataSet.DataProvider = &DataProvider;
            DataSet.DeviceFeatures.resize(DevCount);
        }

        TCompressedCatFeatureDataSetBuilder& Add(ui32 featureId);

        void Finish();

    private:
        template <NCudaLib::EPtrType PtrType>
        TCompressedCatFeatureDataSetBuilder& AddImpl(ui32 featureId,
                                                     TVector<typename TCompressedCatFeatureDataSet::TCompressedCatFeatureVec<PtrType>>* dst);

    private:
        ui32 DevCount;
        TCompressedCatFeatureDataSet& DataSet;
        bool BuildDone = false;
        ui32 DeviceId = 0;
        TVector<ui64> MemoryUsage;

        const NCB::TTrainingDataProvider& DataProvider;
        const TBinarizedFeaturesManager& FeaturesManager;

        NPar::ILocalExecutor* LocalExecutor;
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

    class TMirrorCatFeatureProvider: public TNonCopyable {
    public:
        TMirrorCatFeatureProvider(const TCompressedCatFeatureDataSet& dataSet,
                                  TScopedCacheHolder& cache)
            : Src(dataSet)
            , ScopedCache(cache)
        {
        }

        const TLazyStreamValue<TMirrorBuffer<ui64>>& BroadcastFeature(ui32 featureId,
                                                                      ui32 builderStream = 0);

        const TMirrorBuffer<ui64>& GetFeature(ui32 featureId,
                                              ui32 builderStream = 0);

    private:
        template <NCudaLib::EPtrType Type>
        const TLazyStreamValue<TMirrorBuffer<ui64>>& BroadcastFeatureImpl(ui32 featureId,
                                                                          const typename TCompressedCatFeatureDataSet::TCompressedCatFeatureVec<Type>& src,
                                                                          ui32 builderStream = 0) {
            return ScopedCache.Cache(Src, featureId, [&]() -> TLazyStreamValue<TMirrorBuffer<ui64>> {
                auto mapping = NCudaLib::TMirrorMapping(CompressedSize<ui64>((ui32)Src.GetDocCount(),
                                                                             Src.UniqueValues(featureId)));
                TMirrorBuffer<ui64> dst = TMirrorBuffer<ui64>::Create(mapping);
                NCudaLib::Reshard(src, dst, builderStream);
                return TLazyStreamValue<TMirrorBuffer<ui64>>(std::move(dst),
                                                             builderStream);
            });
        }

    private:
        const TCompressedCatFeatureDataSet& Src;
        TScopedCacheHolder& ScopedCache;
    };

}
