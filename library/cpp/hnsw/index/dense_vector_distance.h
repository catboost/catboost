#pragma once

#include <library/cpp/dot_product/dot_product.h>
#include <library/cpp/l1_distance/l1_distance.h>
#include <library/cpp/l2_distance/l2_distance.h>
#include <library/cpp/pair_vector_distance/pair_vector_similarity.h>

#ifndef CATBOOST_OPENSOURCE
#include <library/cpp/mean_max_similarity/mean_max_similarity.h>
#endif

#include <util/generic/fwd.h>

namespace NHnsw {
    template <class T>
    struct TL1Distance: public NL1Distance::TL1Distance<T> {
        using TLess = ::TLess<typename NL1Distance::TL1Distance<T>::TResult>;
    };

    template <class T>
    struct TL2SqrDistance: public NL2Distance::TL2SqrDistance<T> {
        using TLess = ::TLess<typename NL2Distance::TL2SqrDistance<T>::TResult>;
    };

    template <class T>
    struct TDotProduct: public NDotProduct::TDotProduct<T> {
        using TLess = ::TGreater<typename NDotProduct::TDotProduct<T>::TResult>;
    };

    template <class T>
    struct TPairVectorSimilarity: public NPairVectorSimilarity::TPairVectorSimilarityMetric<T> {
        using TLess = ::TGreater<typename NPairVectorSimilarity::TPairVectorSimilarityMetric<T>::TResult>;
    };

    #ifndef CATBOOST_OPENSOURCE
    template <class T>
    struct TMeanMaxSimilarity: public NMeanMaxSimilarity::TMeanMaxSimilarityMetric<T> {
        using NMeanMaxSimilarity::TMeanMaxSimilarityMetric<T>::TMeanMaxSimilarityMetric;
        using TLess = ::TGreater<typename NMeanMaxSimilarity::TMeanMaxSimilarityMetric<T>::TResult>;
    };
    #endif
}
