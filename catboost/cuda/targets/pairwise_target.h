#pragma once

#include <catboost/cuda/cuda_lib/cuda_buffer.h>

namespace NCatboostCuda {
    struct TPairwiseTarget {
        TStripeBuffer<float> PairWeights;
        TStripeBuffer<uint2> Pairs;

        TStripeBuffer<float> PointTarget;
        //optional, pointwise weights or der2
        TStripeBuffer<float> PointWeights;
        TStripeBuffer<ui32> Docs;
    };

}
