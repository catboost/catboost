#pragma once

#include "tree_ctrs_dataset.h"
#include <catboost/cuda/data/feature.h>
#include <catboost/cuda/data/binarizations_manager.h>
#include <catboost/cuda/gpu_data/feature_parallel_dataset.h>
#include <catboost/cuda/models/oblivious_model.h>
#include <catboost/cuda/gpu_data/oblivious_tree_bin_builder.h>

#include <util/generic/map.h>
#include <util/generic/hash.h>
#include <util/generic/set.h>
#include <catboost/cuda/gpu_data/kernels.h>
#include <catboost/cuda/cuda_lib/device_subtasks_helper.h>
#include <catboost/cuda/gpu_data/batch_binarized_ctr_calcer.h>

namespace NCatboostCuda {
    class TTreeCtrDataSetMemoryUsageEstimator {
    public:
        TTreeCtrDataSetMemoryUsageEstimator(const TBinarizedFeaturesManager& featuresManager,
                                            const double freeMemory,
                                            const ui32 catFeaturesCount,
                                            const ui32 foldCount,
                                            const ui32 maxDepth,
                                            const ui32 docCount,
                                            const NCudaLib::EPtrType ptrType)
            : FreeMemory(freeMemory)
            , MaxDepth(maxDepth)
            , FoldCount(foldCount)
            , DocCount(docCount)
            , CatFeaturesCount(catFeaturesCount)
            , BinFeatureCountPerCatFeature(featuresManager.MaxTreeCtrBinFeaturesCount())
            , CatFeaturesStoragePtrType(ptrType)
            , CtrPerCatFeature(featuresManager.CtrsPerTreeCtrFeatureTensor())
        {
            MaxPackSize = EstimateMaxPackSize();
        }

        ui32 GetStreamCountForCtrCalculation() const {
            if (DocCount > 1e6 && CatFeaturesStoragePtrType == NCudaLib::EPtrType::CudaDevice) {
                return 1;
            }
            if (CatFeaturesStoragePtrType == NCudaLib::EPtrType::CudaHost) {
                return (DocCount > 15e6 ? 1 : 2);
            }
            if (FreeMemory < 512) {
                return 1;
            }
            if (DocCount > 250000) {
                return 4;
            }
            return 8;
        }

        ui32 GetMaxPackSize() const {
            return MaxPackSize;
        }

        bool NotEnoughMemoryForDataSet(const TTreeCtrDataSet& dataSet, ui32 depth) {
            if (dataSet.HasCompressedIndex()) {
                return false;
            }
            const ui32 deviceId = dataSet.GetDeviceId();
            auto freeMemory = NCudaLib::GetCudaManager().FreeMemoryMb(deviceId) - GetReserveMemory(depth);
            return MemoryForDataSet(dataSet) > freeMemory;
        }

        double MemoryForDataSet(const TTreeCtrDataSet& dataSet) {
            ui32 binFeatureCount = 0;

            const int featuresPerInt = static_cast<int>(dataSet.GetMaxFeaturesPerInt());
            double cindexSize = NHelpers::CeilDivide(dataSet.GetCtrs().size(), featuresPerInt) * 4.0 * DocCount / MB;
            double memoryForCtrsCalc = 0;
            if (dataSet.HasCompressedIndex()) {
                binFeatureCount = static_cast<ui32>(dataSet.GetFeatureCount());
            } else {
                binFeatureCount = static_cast<ui32>(BinFeatureCountPerCatFeature * dataSet.GetCatFeatures().size());
                memoryForCtrsCalc += MemoryForCtrInOneStreamCalculation() * GetStreamCountForCtrCalculation();
            }
            return (cindexSize + MemoryForHistogramsMb(binFeatureCount) + memoryForCtrsCalc +
                    MemoryForGridForOneFeatureMb() * dataSet.GetCatFeatures().size()) *
                   ReservationFactor;
        }

        double GetReserveMemory(ui32 currentDepth) const {
            return ReserveMemory + MemoryForTensorMirrorCatFeaturesCache(currentDepth) + MemoryForRestBuffers() +
                   MaxDepth * MemoryForGridForOneFeatureMb() * CatFeaturesCount +
                   MemoryForCtrInOneStreamCalculation() * GetStreamCountForCtrCalculation();
        }

    private:
        double MemoryForCtrInOneStreamCalculation() const {
            return 44 * DocCount * 1.0 / MB;
        }

        double MemoryForTensorMirrorCatFeaturesCache(ui32 currentDepth) const {
            return 8 * (MaxDepth - 1 - currentDepth) * DocCount / MB;
        }

        double MemoryForRestBuffers() const {
            return 8 * DocCount / MB;
        }

        double MemoryForGridForOneFeatureMb() const {
            return (CtrPerCatFeature * (sizeof(TCFeature) + sizeof(float)) +
                    (sizeof(TCBinFeature) + sizeof(float)) * BinFeatureCountPerCatFeature) *
                   1.0 / MB;
        }

        double MemoryForHistogramsMb(ui32 binFeatureCount) const {
            return (1 << MaxDepth) * FoldCount * binFeatureCount * 2.0 * 4.0 / MB;
        }

        double MemoryForHistogramsMb() const {
            return MemoryForHistogramsMb(BinFeatureCountPerCatFeature);
        }

        double MemoryForCompressedIndexMb() {
            return NHelpers::CeilDivide<ui32>(CtrPerCatFeature, 4) * sizeof(ui32) * DocCount / MB;
        }

        double EstimateMaxPackSize() {
            const double freeMemoryForDataSet = FreeMemory - GetReserveMemory(0);

            const double memoryForFeature =
                (MemoryForHistogramsMb() + MemoryForCompressedIndexMb()) * ReservationFactor;
            const double maxFeatures = freeMemoryForDataSet > 0 ? freeMemoryForDataSet / memoryForFeature : 0;

            if (maxFeatures >= CatFeaturesCount) {
                return CatFeaturesCount;
            }
            if (maxFeatures == 0) {
                ythrow TCatBoostException() << "Error: not enough memory for tree-ctrs: " << FreeMemory
                                            << " MB available, need at least "
                                            << (FreeMemory - freeMemoryForDataSet) + memoryForFeature << " MB";
            }
            const ui32 blockCount = static_cast<ui32>(ceil(CatFeaturesCount * 1.0 / maxFeatures));
            return NHelpers::CeilDivide(CatFeaturesCount, blockCount);
        }

        const double FreeMemory;
        const ui32 MaxDepth;
        const ui32 FoldCount;
        const ui32 DocCount;
        const ui32 CatFeaturesCount;
        const ui32 BinFeatureCountPerCatFeature; //for one cat feature
        const NCudaLib::EPtrType CatFeaturesStoragePtrType;
        const ui32 CtrPerCatFeature;
        ui32 MaxPackSize = 0;

        const double MB = 1024 * 1024;
        const double ReserveMemory = 100;
        //some magic consts
        const double ReservationFactor = 1.15;
    };
}
