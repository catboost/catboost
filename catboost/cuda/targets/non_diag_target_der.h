#pragma once

#include <catboost/cuda/cuda_lib/cuda_buffer.h>

namespace NCatboostCuda {
    struct TNonDiagQuerywiseTargetDers {
        TStripeBuffer<float> PairDer2OrWeights;
        TStripeBuffer<uint2> Pairs;

        TStripeBuffer<float> PointWeightedDer;
        //optional, pointwise weights or der2
        TStripeBuffer<float> PointDer2OrWeights;

        TStripeBuffer<ui32> Docs;
    };

}
