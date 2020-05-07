#include "cat_features_dataset.h"

namespace NCatboostCuda {
    using EPtrType = NCudaLib::EPtrType;

    TCompressedCatFeatureDataSetBuilder& TCompressedCatFeatureDataSetBuilder::Add(ui32 featureId) {
        if (DataSet.StorageType == EGpuCatFeaturesStorage::GpuRam) {
            return AddImpl<NCudaLib::EPtrType::CudaDevice>(featureId,
                                                           &DataSet.CompressedCatIndexGpu);
        } else {
            CB_ENSURE(DataSet.StorageType == EGpuCatFeaturesStorage::CpuPinnedMemory);
            return AddImpl<NCudaLib::EPtrType::CudaHost>(featureId,
                                                         &DataSet.CompressedCatIndexCpu);
        }
    }

    void TCompressedCatFeatureDataSetBuilder::Finish() {
        CB_ENSURE(!BuildDone, "Error: build could be done only once");
        CATBOOST_INFO_LOG << "Build catFeatures compressed dataset "
                          << "for "
                          << DataSet.GetFeatureCount() << " features and " << DataSet.GetDocCount() << " documents"
                          << Endl;

        for (ui32 dev = 0; dev < DevCount; ++dev) {
            CATBOOST_INFO_LOG
                << "Memory usage at #" << dev << ": " << sizeof(ui64) * MemoryUsage[dev] * 1.0 / 1024 / 1024 << "MB"
                << Endl;
        }
        BuildDone = true;
    }

    template <EPtrType PtrType>
    TCompressedCatFeatureDataSetBuilder& TCompressedCatFeatureDataSetBuilder::AddImpl(ui32 featureId,
                                                                                      TVector<typename TCompressedCatFeatureDataSet::TCompressedCatFeatureVec<PtrType>>* dst) {
        const ui32 dataProviderId = FeaturesManager.GetDataProviderId(featureId);
        const auto& catFeature = **(DataProvider.ObjectsData->GetCatFeature(
            DataProvider.MetaInfo.FeaturesLayout->GetInternalFeatureIdx(dataProviderId)));

        const ui64 docCount = catFeature.GetSize();

        TSingleBuffer<ui32> tmp = TSingleBuffer<ui32>::Create(
            NCudaLib::TSingleMapping(DeviceId, docCount)
        );
        tmp.Write(catFeature.ExtractValues<ui32>(LocalExecutor));
        const auto uniqueValues = FeaturesManager.GetBinCount(featureId);
        const auto compressedSize = CompressedSize<ui64>((ui32)docCount, uniqueValues);
        auto compressedMapping = NCudaLib::TSingleMapping(DeviceId, compressedSize);

        auto& catIndex = *dst;
        DataSet.Features[featureId] = {static_cast<ui32>(catIndex.size()), uniqueValues};
        catIndex.push_back(TCudaBuffer<ui64, NCudaLib::TSingleMapping, PtrType>::Create(compressedMapping));
        Compress(tmp, catIndex.back(), uniqueValues);
        MemoryUsage[DeviceId] += compressedMapping.MemorySize();
        DataSet.DeviceFeatures[DeviceId].push_back(featureId);

        DeviceId = (DeviceId + 1) % DevCount;
        return *this;
    }

    const TLazyStreamValue<TMirrorBuffer<ui64>>&
    TMirrorCatFeatureProvider::BroadcastFeature(ui32 featureId, ui32 builderStream) {
        if (Src.GetStorageType() == EGpuCatFeaturesStorage::CpuPinnedMemory) {
            return BroadcastFeatureImpl<NCudaLib::EPtrType::CudaHost>(featureId,
                                                                      Src.GetFeatureCpu(featureId),
                                                                      builderStream);
        } else {
            return BroadcastFeatureImpl<NCudaLib::EPtrType::CudaDevice>(featureId,
                                                                        Src.GetFeatureGpu(featureId),
                                                                        builderStream);
        }
    }

    const TMirrorBuffer<ui64>& TMirrorCatFeatureProvider::GetFeature(ui32 featureId, ui32 builderStream) {
        return BroadcastFeature(featureId, builderStream).Get();
    }
}
