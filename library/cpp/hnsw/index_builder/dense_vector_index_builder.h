#pragma once

#include "index_builder.h"
#include "index_data.h"
#include "dense_vector_distance.h"
#include "dense_vector_storage.h"

#include <util/generic/fwd.h>

#include <stddef.h>

class TBlob;


/**
 * @brief Implementation of NHnsw::BuildIndex for dense vectors.
 *
 * These methods are intended to cover the most common scenario for BuildIndex -
 * building indexes over N-dimensional vectors of POD type i8/i32/float/double.
 *
 * The most common distances (such as L1, L2 and DotProduct) are provided in hnsw/index/dense_vector_distance.h
 *
 * The simpliest vector storage is provided in dense_vector_storage.h
 *
 * @tparam TVectorComponent - type of dense vector component, typically one of i8/i32/float/double
 */
namespace NHnsw {
    struct THnswBuildOptions;

    template <class TVectorComponent,
              class TDistance,
              class TDistanceResult = typename TDistance::TResult,
              class TDistanceLess = typename TDistance::TLess>
    THnswIndexData BuildDenseVectorIndex(const THnswBuildOptions& opts,
                                         const TDenseVectorStorage<TVectorComponent>& itemStorage,
                                         size_t dimension,
                                         const TDistance& distance = {},
                                         const TDistanceLess& distanceLess = {}) {
        TDistanceWithDimension<TVectorComponent, TDistance> distanceWithDimension(distance, dimension);
        return BuildIndex<TDistanceWithDimension<TVectorComponent, TDistance>, TDistanceResult, TDistanceLess>(opts, itemStorage, distanceWithDimension, distanceLess);
    }

    template <class TVectorComponent,
              class TDistance,
              class TDistanceResult = typename TDistance::TResult,
              class TDistanceLess = typename TDistance::TLess>
    THnswIndexData BuildDenseVectorIndex(const THnswBuildOptions& opts,
                                         const TBlob& vectorData,
                                         size_t dimension,
                                         const TDistance& distance = {},
                                         const TDistanceLess& distanceLess = {}) {
        TDenseVectorStorage<TVectorComponent> itemStorage(vectorData, dimension);
        return BuildDenseVectorIndex<TVectorComponent, TDistance, TDistanceResult, TDistanceLess>(opts, itemStorage, dimension, distance, distanceLess);
    }

    template <class TVectorComponent,
              class TDistance,
              class TDistanceResult = typename TDistance::TResult,
              class TDistanceLess = typename TDistance::TLess>
    THnswIndexData BuildDenseVectorIndex(const THnswBuildOptions& opts,
                                         const TString& vectorFilename,
                                         size_t dimension,
                                         const TDistance& distance = {},
                                         const TDistanceLess& distanceLess = {}) {
        TDenseVectorStorage<TVectorComponent> itemStorage(vectorFilename, dimension);
        return BuildDenseVectorIndex<TVectorComponent, TDistance, TDistanceResult, TDistanceLess>(opts, itemStorage, dimension, distance, distanceLess);
    }

}
