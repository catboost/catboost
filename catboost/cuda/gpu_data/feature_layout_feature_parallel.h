#pragma once

#include "gpu_structures.h"
#include "grid_policy.h"
#include "feature_layout_common.h"
#include "kernels.h"
#include "cuda_features_helper.h"

#include <catboost/cuda/cuda_lib/mapping.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_util/fill.h>
#include <catboost/cuda/data/feature.h>
#include <catboost/cuda/data/binarizations_manager.h>
#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/data/lazy_columns.h>

namespace NCatboostCuda {
    template <>
    struct TCudaFeaturesLayoutHelper<TFeatureParallelLayout> {
        using TFeaturesBlock = TGpuFeaturesBlockDescription<NCudaLib::TStripeMapping, NCudaLib::TMirrorMapping>;

        static NCudaLib::TStripeMapping CreateLayout(ui32 featureCount) {
            return NCudaLib::TStripeMapping::SplitBetweenDevices(featureCount);
        }

        static NCudaLib::TMirrorMapping CreateDocLayout(ui32 docCount) {
            return NCudaLib::TMirrorMapping(docCount);
        }

        template <EFeaturesGroupingPolicy Policy,
                  class TFeaturesBinarizationDescription>
        static THolder<TFeaturesBlock> CreateFeaturesBlock(TVector<ui32>& featureIds,
                                                           const TFeaturesBinarizationDescription& info,
                                                           const NCudaLib::TMirrorMapping& docsMapping,
                                                           const NCudaLib::TDistributedObject<ui64>& cindexOffsets) {
            auto allFeaturesLayout = CreateLayout(featureIds.size());

            if (Policy == EFeaturesGroupingPolicy::OneByteFeatures) {
                TRandom rand(0);
                Shuffle(featureIds.begin(), featureIds.end(), rand);

                for (ui32 dev : allFeaturesLayout.NonEmptyDevices()) {
                    TSlice devSlice = allFeaturesLayout.DeviceSlice(dev);
                    std::sort(featureIds.begin() + devSlice.Left, featureIds.begin() + devSlice.Right,
                              [&](ui32 left, ui32 right) -> bool {
                                  return info.GetGroupingLevel(left) < info.GetGroupingLevel(right);
                              });
                }
            }

            THolder<TFeaturesBlock> resultHolder = MakeHolder<TFeaturesBlock>(TCpuGrid(info, featureIds));
            TFeaturesBlock& result = *resultHolder;
            TCudaFeaturesHelper helper(result.Grid);

            result.CudaFeaturesHost.resize(featureIds.size(),
                                           NCudaLib::GetCudaManager().CreateDistributedObject<TCFeature>());

            result.Samples = docsMapping;

            TVector<TCFeature> allFeatures;
            TVector<TCFeature> features;

            const ui64 docCount = docsMapping.GetObjectsSlice().Size();

            result.BinFeatureCount = NCudaLib::GetCudaManager().CreateDistributedObject<ui32>(0);
            TVector<size_t> trainFeatureSlicesSizes(NCudaLib::GetCudaManager().GetDeviceCount(), 0);
            size_t prevFeaturesSize = 0;
            for (auto dev : allFeaturesLayout.NonEmptyDevices()) {
                auto devSlice = allFeaturesLayout.DeviceSlice(dev);
                const ui64 cindexDeviceOffset = cindexOffsets.At(dev);
                const ui64 devSize = helper.AddDeviceFeatures<Policy>(devSlice,
                                                                      cindexDeviceOffset,
                                                                      docCount,
                                                                      &allFeatures,
                                                                      &features);
                result.CIndexSizes.Set(dev, devSize);
                result.CIndexOffsets.Set(dev, cindexDeviceOffset);
                trainFeatureSlicesSizes[dev] = features.size() - prevFeaturesSize;
                prevFeaturesSize = features.size();

                for (ui32 i = devSlice.Left; i < devSlice.Right; ++i) {
                    result.CudaFeaturesHost[i].Set(dev, allFeatures[i]);
                }
                result.BinFeatureCount.Set(dev, helper.BuildBinaryFeatures(devSlice).size());
            };
            if (features.size()) {
                auto featuresLayout = NCudaLib::TStripeMapping::CreateFromSizes(trainFeatureSlicesSizes);
                for (auto dev : featuresLayout.NonEmptyDevices()) {
                    auto devSlice = featuresLayout.DeviceSlice(dev);
                    result.FoldsHistogram.Set(dev, result.Grid.ComputeFoldsHistogram(devSlice));
                }
                result.CudaFeaturesDevice.Reset(featuresLayout);
                result.CudaFeaturesDevice.Write(features);
            }

            //bin features
            result.BinFeatures = helper.BuildBinaryFeatures(TSlice(0, allFeatures.size()));
            result.HistogramsMapping = CreateMapping<NCudaLib::TStripeMapping>(result.BinFeatureCount);
            if (result.BinFeatures.size()) {
                result.BinFeaturesForBestSplits.Reset(result.HistogramsMapping);
                result.BinFeaturesForBestSplits.Write(result.BinFeatures);
            }

            return resultHolder;
        }

        static void WriteToCompressedIndex(const NCudaLib::TDistributedObject<TCFeature>& feature,
                                           TConstArrayRef<ui8> bins,
                                           const NCudaLib::TMirrorMapping&,
                                           TStripeBuffer<ui32>* compressedIndex) {
            ui32 writeDev = -1;
            for (ui32 dev = 0; dev < feature.DeviceCount(); ++dev) {
                if (!feature.IsEmpty(dev)) {
                    CB_ENSURE(writeDev == static_cast<ui32>(-1));
                    writeDev = dev;
                }
            }
            CB_ENSURE(writeDev != static_cast<ui32>(-1));

            using TBuffer = TCudaBuffer<ui8, NCudaLib::TSingleMapping, NCudaLib::EPtrType::CudaHost>;
            TBuffer tmp = TBuffer::Create(NCudaLib::TSingleMapping(writeDev,
                                                                   bins.size()));
            tmp.Write(bins);
            WriteCompressedFeature(feature, tmp, *compressedIndex);
        }

        static void WriteToLazyCompressedIndex(const NCudaLib::TDistributedObject<TCFeature>& feature,
                                           const NCB::TLazyQuantizedFloatValuesHolder* lazyQuantizedColumn,
                                           ui32 featureId,
                                           TMaybe<ui16> baseValue,
                                           const NCudaLib::TMirrorMapping& docsMapping,
                                           TStripeBuffer<ui32>* compressedIndex) {
            Y_UNUSED(feature);
            Y_UNUSED(lazyQuantizedColumn);
            Y_UNUSED(featureId);
            Y_UNUSED(baseValue);
            Y_UNUSED(docsMapping);
            Y_UNUSED(compressedIndex);
            CB_ENSURE_INTERNAL(false, "Lazy dataset loading does not support feature parallel layout");
        }
    };

    extern template struct TCudaFeaturesLayoutHelper<TFeatureParallelLayout>;

}
