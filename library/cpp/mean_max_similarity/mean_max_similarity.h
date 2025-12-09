#pragma once

#include <contrib/libs/eigen/Eigen/Core>
#include <contrib/libs/eigen/Eigen/SparseCore>
#include <library/cpp/dot_product/dot_product.h>
#include <util/generic/ymath.h>
#include <util/system/compiler.h>
#include <util/system/defaults.h>
#include <util/string/builder.h>
#include <util/generic/yexception.h>
#include <limits>
#include <cstddef>

namespace NMeanMaxSimilarity {
    template <typename T>
    T MeanMaxSimilarityMetric(
        const T* lhs, size_t lhsLength,
        const T* rhs, size_t rhsLength,
        size_t tokenDimension
    ) noexcept {
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, Eigen::Unaligned> Q(lhs, lhsLength, tokenDimension);
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, Eigen::Unaligned> D(rhs, rhsLength, tokenDimension);

        return (Q * D.transpose()).rowwise().maxCoeff().mean();
    }

    template <typename T>
    struct TMeanMaxSimilarityMetric {
        size_t DocLength;
        size_t TokenDimension;

        TMeanMaxSimilarityMetric() = default;
        TMeanMaxSimilarityMetric(size_t docLength, size_t tokenDimension)
            : DocLength(docLength)
            , TokenDimension(tokenDimension)
        {
        }

        using TResult = decltype(
            MeanMaxSimilarityMetric(static_cast<const T*>(nullptr), 0,static_cast<const T*>(nullptr),0,0)
        );

        Y_PURE_FUNCTION
        inline TResult operator()(const T* l, const T* r, size_t dimension) const {
            Y_ENSURE(dimension % TokenDimension == 0, "dimension (" << dimension << ") should be divisible by tokenDimension (" << TokenDimension << ")");
            return MeanMaxSimilarityMetric(l, dimension / TokenDimension, r, DocLength, TokenDimension);
        }
    };
} // namespace NMeanMaxSimilarity
