#pragma once

#include "gpu_structures.h"
#include "grid_policy.h"
#include "feature_layout_common.h"
#include "cuda_features_helper.h"

#include <catboost/cuda/cuda_lib/mapping.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_util/fill.h>
#include <catboost/cuda/data/feature.h>
#include <catboost/cuda/data/binarizations_manager.h>
#include <catboost/libs/data/data_provider.h>

namespace NCatboostCuda {
    template <>
    struct TCudaFeaturesLayoutHelper<TSingleDevLayout> {
        using TFeaturesBlock = TGpuFeaturesBlockDescription<NCudaLib::TSingleMapping, NCudaLib::TSingleMapping>;

        static ui32 GetActiveDevice() {
            TSet<ui32> devices;
            for (auto dev : NCudaLib::GetCudaManager().GetActiveDevices()) {
                devices.insert(dev);
            }
            CB_ENSURE(devices.size() == 1, "Can't create single mapping layout with more than one active device");
            return *devices.begin();
        }

        static NCudaLib::TSingleMapping CreateLayout(ui32 featureCount) {
            return NCudaLib::TSingleMapping(GetActiveDevice(), featureCount);
        }

        template <EFeaturesGroupingPolicy Policy,
                  class TFeaturesBinarizationDescription>
        static THolder<TFeaturesBlock> CreateFeaturesBlock(TVector<ui32> featureIds,
                                                           const TFeaturesBinarizationDescription& info,
                                                           const NCudaLib::TSingleMapping& docsMapping,
                                                           const NCudaLib::TDistributedObject<ui64>& cindexOffsets) {
            THolder<TFeaturesBlock> resultHolder = MakeHolder<TFeaturesBlock>(TCpuGrid(info, featureIds));
            TFeaturesBlock& result = *resultHolder;

            auto layout = CreateLayout(featureIds.size());
            ui32 dev = docsMapping.GetDeviceId();

            result.CudaFeaturesHost.resize(featureIds.size(),
                                           NCudaLib::GetCudaManager().CreateDistributedObject<TCFeature>());

            result.Samples = docsMapping;

            TVector<TCFeature> allFeatures;
            TVector<TCFeature> features;

            TCudaFeaturesHelper helper(result.Grid);

            const ui64 docCount = docsMapping.GetObjectsSlice().Size();
            auto devSlice = layout.DeviceSlice(dev);
            CB_ENSURE(devSlice.Left == 0);
            CB_ENSURE(devSlice.Right == featureIds.size());
            const ui64 cindexDeviceOffset = cindexOffsets.At(dev);
            const ui64 devSize = helper.AddDeviceFeatures<Policy>(devSlice,
                                                                  cindexDeviceOffset,
                                                                  docCount,
                                                                  &allFeatures,
                                                                  &features);
            CB_ENSURE_INTERNAL(
                features.size() == allFeatures.size(),
                "Catboost don't support skipped features in TSingleDevLayout"
            );
            result.CIndexSizes.Set(dev, devSize);
            result.CIndexOffsets.Set(dev, cindexDeviceOffset);

            for (ui32 i = devSlice.Left; i < devSlice.Right; ++i) {
                result.CudaFeaturesHost[i].Set(dev, allFeatures[i]);
            }
            result.FoldsHistogram.Set(dev, result.Grid.ComputeFoldsHistogram(devSlice));

            result.CudaFeaturesDevice.Reset(layout);
            result.CudaFeaturesDevice.Write(features);

            //bin features
            result.BinFeatures = helper.BuildBinaryFeatures(TSlice(0, allFeatures.size()));
            result.BinFeatureCount.Set(dev, result.BinFeatures.size());
            Y_ASSERT(result.BinFeatureCount.At(dev) == result.BinFeatures.size());
            result.HistogramsMapping = NCudaLib::TSingleMapping(dev, result.BinFeatureCount.At(dev));

            result.BinFeaturesForBestSplits.Reset(result.HistogramsMapping);
            result.BinFeaturesForBestSplits.Write(result.BinFeatures);

            return resultHolder;
        }
    };

}
