#pragma once

#include "feature_layout_common.h"

#include "cuda_features_helper.h"
#include "gpu_structures.h"
#include "grid_policy.h"
#include "kernels.h"

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_manager.h>
#include <catboost/cuda/cuda_lib/mapping.h>
#include <catboost/cuda/cuda_lib/slice.h>

#include <catboost/libs/data/lazy_columns.h>
#include <catboost/libs/helpers/cpu_random.h>
#include <catboost/libs/helpers/exception.h>

#include <util/generic/ptr.h>
#include <util/random/shuffle.h>

#include <algorithm>

namespace NCatboostCuda {
    template <>
    struct TCudaFeaturesLayoutHelper<TDocParallelLayout> {
        using TFeaturesBlock = TGpuFeaturesBlockDescription<NCudaLib::TStripeMapping, NCudaLib::TStripeMapping>;

        static NCudaLib::TStripeMapping CreateLayout(const ui32 featureCount) {
            return NCudaLib::TStripeMapping::RepeatOnAllDevices(featureCount);
        }

        static NCudaLib::TStripeMapping CreateDocLayout(ui32 docCount) {
            return NCudaLib::TStripeMapping::SplitBetweenDevices(docCount);
        }

        template <EFeaturesGroupingPolicy Policy,
                  class TFeaturesBinarizationDescription>
        static THolder<TFeaturesBlock> CreateFeaturesBlock(TVector<ui32> featureIds,
                                                           const TFeaturesBinarizationDescription& info,
                                                           const NCudaLib::TStripeMapping& docsMapping,
                                                           const NCudaLib::TDistributedObject<ui64>& cindexOffsets) {
            if (Policy == EFeaturesGroupingPolicy::OneByteFeatures) {
                TRandom rand(0);
                Shuffle(featureIds.begin(), featureIds.end(), rand);

                std::sort(featureIds.begin(), featureIds.end(), [&](ui32 left, ui32 right) -> bool {
                    return info.GetGroupingLevel(left) < info.GetGroupingLevel(right);
                });
            }
            THolder<TFeaturesBlock> resultHolder = MakeHolder<TFeaturesBlock>(TCpuGrid(info, featureIds));
            TFeaturesBlock& result = *resultHolder;
            ui32 featureCount = featureIds.size();

            result.Samples = docsMapping;
            result.CudaFeaturesHost.resize(featureIds.size(),
                                           NCudaLib::GetCudaManager().CreateDistributedObject<TCFeature>());

            TVector<TCFeature> allFeatures;
            TVector<TCFeature> features;
            TCudaFeaturesHelper helper(result.Grid);

            const TSlice featuresSlice = TSlice(0, featureCount);

            for (ui32 dev = 0; dev < GetDeviceCount(); ++dev) {
                const ui64 docCount = docsMapping.DeviceSlice(dev).Size();

                const ui64 devCIndexOffset = cindexOffsets.At(dev);
                const ui64 devSize = helper.AddDeviceFeatures<Policy>(featuresSlice,
                                                                      devCIndexOffset,
                                                                      docCount,
                                                                      &allFeatures,
                                                                      &features);
                result.CIndexSizes.Set(dev, devSize);
                result.CIndexOffsets.Set(dev, devCIndexOffset);

                auto foldsHistogram = result.Grid.ComputeFoldsHistogram();
                for (ui32 i = 0; i < featureCount; ++i) {
                    result.CudaFeaturesHost[i].Set(dev, allFeatures[dev * featureCount + i]);
                    result.FoldsHistogram.Set(dev, foldsHistogram);
                }
            };

            CB_ENSURE(allFeatures.size() == GetDeviceCount() * featureCount);
            if (features.size()) {
                auto layout = CreateLayout(features.size() / GetDeviceCount());
                result.CudaFeaturesDevice.Reset(layout);
                result.CudaFeaturesDevice.Write(features);
            }
            //bin features.
            result.BinFeatures = helper.BuildBinaryFeatures(TSlice(0, featureCount));
            result.BinFeatureCount = NCudaLib::GetCudaManager().CreateDistributedObject<ui32>(result.BinFeatures.size());
            result.HistogramsMapping = NCudaLib::TStripeMapping::RepeatOnAllDevices(result.BinFeatures.size());

            auto mapping = NCudaLib::TStripeMapping::SplitBetweenDevices(featureCount).Transform([&](const TSlice& featureSlice) {
                //to make seed based on device and featuresId work
                return helper.BuildBinaryFeatures(featureSlice).size();
            });
            if (result.BinFeatures.size()) {
                result.BinFeaturesForBestSplits.Reset(mapping);
                result.BinFeaturesForBestSplits.Write(result.BinFeatures);
            }

            return resultHolder;
        }

        static void WriteToCompressedIndex(const NCudaLib::TDistributedObject<TCFeature>& feature,
                                           TConstArrayRef<ui8> bins,
                                           const NCudaLib::TStripeMapping& docsMapping,
                                           TStripeBuffer<ui32>* compressedIndex) {
            TStripeBuffer<ui8> tmp = TStripeBuffer<ui8>::Create(docsMapping);
            tmp.Write(bins);
            WriteCompressedFeature(feature, tmp, *compressedIndex);
        }

        static void WriteToLazyCompressedIndex(const NCudaLib::TDistributedObject<TCFeature>& feature,
                                           const NCB::TLazyQuantizedFloatValuesHolder* lazyQuantizedColumn,
                                           ui32 featureId,
                                           TMaybe<ui16> baseValue,
                                           const NCudaLib::TStripeMapping& docsMapping,
                                           TStripeBuffer<ui32>* compressedIndex) {
            WriteLazyCompressedFeature(
                feature,
                docsMapping,
                lazyQuantizedColumn->GetPathWithScheme(),
                lazyQuantizedColumn->GetId(),
                featureId,
                baseValue,
                *compressedIndex);
        }
    };

    extern template struct TCudaFeaturesLayoutHelper<TDocParallelLayout>;

}
