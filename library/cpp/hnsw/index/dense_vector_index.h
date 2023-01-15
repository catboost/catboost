#pragma once

#include "index_item_storage_base.h"
#include "dense_vector_distance.h"
#include "dense_vector_item_storage.h"

#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/memory/blob.h>

namespace NHnsw {

    /**
     * @brief Implementation of THnswIndexBase for dense vectors.
     *
     * This class is intended to cover the most common scenario for THnswIndex -
     * searching in a set of N-dimensional vectors of POD type i8/i32/float/double.
     *
     * The most common distances (such as L1, L2 and DotProduct) are provided in dense_vector_distance.h
     *
     * @tparam TVectorComponent - type of dense vector component, typically one of i8/i32/float/double
     */
    template <class TVectorComponent>
    class THnswDenseVectorIndex: public THnswItemStorageIndexBase<THnswDenseVectorIndex<TVectorComponent>>, public TDenseVectorItemStorage<TVectorComponent> {
        using TIndexBase = THnswItemStorageIndexBase<THnswDenseVectorIndex<TVectorComponent>>;
        using TItemStorage = TDenseVectorItemStorage<TVectorComponent>;

    public:
        template <class TDistanceResult>
        using TNeighbor = typename TIndexBase::template TNeighbor<TDistanceResult>;

        template <class TIndexReader = THnswIndexReader>
        THnswDenseVectorIndex(const TString& indexDataFilename,
                              const TString& vectorsFilename,
                              size_t dimension,
                              const TIndexReader& indexReader = TIndexReader())
            : THnswDenseVectorIndex(TBlob::PrechargedFromFile(indexDataFilename),
                                    TBlob::PrechargedFromFile(vectorsFilename),
                                    dimension,
                                    indexReader) {
        }

        template <class TIndexReader = THnswIndexReader>
        THnswDenseVectorIndex(const TBlob& indexDataBlob,
                              const TBlob& vectorData,
                              size_t dimension,
                              const TIndexReader& indexReader = TIndexReader())
            : TIndexBase(indexDataBlob, indexReader)
            , TItemStorage(vectorData, dimension)
        {
        }

        template <class TDistance,
                  class TDistanceResult = typename TDistance::TResult,
                  class TDistanceLess = typename TDistance::TLess>
        TVector<TNeighbor<TDistanceResult>> GetNearestNeighbors(const TVectorComponent* query,
                                                                size_t topSize,
                                                                size_t searchNeighborhoodSize,
                                                                size_t distanceCalcLimit,
                                                                const TDistance& distance = {},
                                                                const TDistanceLess& distanceLess = {}) const {
            auto distanceWithDimension = [this, &distance](const TVectorComponent* a, const TVectorComponent* b) {
                return distance(a, b, this->GetDimension());
            };
            return TIndexBase::template GetNearestNeighbors<decltype(distanceWithDimension), TDistanceResult, TDistanceLess>(
                query, topSize, searchNeighborhoodSize, distanceCalcLimit, distanceWithDimension, distanceLess);
        }

        template <class TDistance,
                  class TDistanceResult = typename TDistance::TResult,
                  class TDistanceLess = typename TDistance::TLess>
        TVector<TNeighbor<TDistanceResult>> GetNearestNeighbors(const TVectorComponent* query,
                                                                size_t topSize,
                                                                size_t searchNeighborhoodSize,
                                                                const TDistance& distance = {},
                                                                const TDistanceLess& distanceLess = {}) const {
            return GetNearestNeighbors(query, topSize, searchNeighborhoodSize, Max<size_t>(), distance, distanceLess);
        }
    };

}
