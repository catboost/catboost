#pragma once

#include <library/cpp/dot_product/dot_product.h>
#include <util/generic/fwd.h>
#include <util/generic/ymath.h>
#include <util/system/compiler.h>

namespace NPairVectorSimilarity {
    template<typename T>
    inline double EmbeddingsCos(const T* lhs, const T* rhs, size_t length) {
        double norm = sqrt(DotProduct(lhs, lhs, length) * DotProduct(rhs, rhs, length));

        if (abs(norm) < 1e-7 || IsNan(norm)) {
            return 0;
        }

        return DotProduct(lhs, rhs, length) / norm;
    }

    template<typename T>
    double PairVectorSimilarityMetric(const T* lhs, const T* rhs, size_t length) noexcept {
        double cos1 = EmbeddingsCos(lhs, rhs, length/2);
        double cos2 = EmbeddingsCos(lhs+length/2, rhs+length/2, length/2);

        double norm_cos1 = (cos1 + 1.0) / 2.0;
        double norm_cos2 = (cos2 + 1.0) / 2.0;

        if (abs(norm_cos1 + norm_cos2) < 1e-7) {
            return 0;
        }

        return 2.0 * norm_cos1 * norm_cos2 / (norm_cos1 + norm_cos2);
    }

   template<typename T>
    struct TPairVectorSimilarityMetric {

        using TResult = decltype(PairVectorSimilarityMetric(static_cast<const T*>(nullptr), static_cast<const T*>(nullptr), 0));

        Y_PURE_FUNCTION
        inline TResult operator()(const T* l, const T* r, size_t length) const {
            return PairVectorSimilarityMetric(l, r, length);
        }
    };

}
