#pragma once

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_util/sort.h>
#include <catboost/cuda/data/grid_creator.h>

#include <library/grid_creator/binarization.h>

#include <util/generic/vector.h>

namespace NCatboostCuda {
    template <EBorderSelectionType type>
    class TGpuGridBuilder: public TGridBuilderBase<type> {
    public:
        IGridBuilder& AddFeature(const TVector<float>& feature,
                                 ui32 borderCount) override {
            TVector<float> borders;
            if (feature.size() > 1e5) {
                auto gpuFeature = TSingleBuffer<float>::Create(NCudaLib::TSingleMapping(0, feature.size()));
                gpuFeature.Write(feature);
                RadixSort(gpuFeature);

                TVector<float> sortedFeature;
                gpuFeature.Read(sortedFeature);
                borders = TGridBuilderBase<type>::BuildBorders(sortedFeature, borderCount);
            } else {
                TVector<float> tmp(feature.begin(), feature.end());
                Sort(tmp.begin(), tmp.end());
                borders = TGridBuilderBase<type>::BuildBorders(tmp, borderCount);
            }
            Result.push_back(std::move(borders));
            return *this;
        }

        const TVector<TVector<float>>& Borders() override {
            return Result;
        }

    private:
        TVector<TVector<float>> Result;
    };

    using TOnGpuGridBuilderFactory = TGridBuilderFactory<TGpuGridBuilder>;
}
