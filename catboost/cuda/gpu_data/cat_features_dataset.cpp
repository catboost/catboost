#include "cat_features_dataset.h"

namespace NCatboostCuda {
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

    const TLazyStreamValue<TMirrorBuffer<unsigned long>>&
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
