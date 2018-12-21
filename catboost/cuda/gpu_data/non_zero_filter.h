#pragma once

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_util/fill.h>
#include <catboost/cuda/cuda_util/filter.h>
#include <catboost/cuda/cuda_util/operator.h>
#include <catboost/cuda/cuda_util/reduce.h>
#include <catboost/cuda/cuda_util/sort.h>
#include <catboost/cuda/cuda_util/transform.h>

namespace NCatboostCuda {
    //warning: will not free memory
    template <class TMapping>
    inline void FilterZeroEntries(TCudaBuffer<float, TMapping>* weights,
                                  TCudaBuffer<ui32, TMapping>* nzIndices) {
        TCudaBuffer<ui32, TMapping> status;
        status.Reset(weights->GetMapping());
        NonZeroFilter(*weights, status);

        auto& indices = *nzIndices;
        indices.Reset(status.GetMapping());
        MakeSequence(indices);
        RadixSort(status, indices, true, 0u, 1u);

        TCudaBuffer<ui32, TMapping> nzSizes;
        auto resultMapping = status.GetMapping().Transform([&](const TSlice&) {
            return 1;
        });
        nzSizes.Reset(resultMapping);

        ReduceVector(status, nzSizes, EOperatorType::Sum);

        TVector<ui32> nzSizesMaster;
        nzSizes.Read(nzSizesMaster);
        auto nzMapping = nzSizes.GetMapping().Transform([&](const TSlice& slice) {
            CB_ENSURE(slice.Size() == 1);
            return nzSizesMaster[slice.Left];
        });
        indices.Reset(nzMapping);

        {
            auto tmp = TCudaBuffer<float, TMapping>::CopyMapping(status);
            tmp.Copy(*weights);
            weights->Reset(nzMapping);
            Gather(*weights, tmp, indices);
        }
    }
}
