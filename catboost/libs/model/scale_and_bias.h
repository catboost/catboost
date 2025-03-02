#pragma once

#include <catboost/libs/helpers/exception.h>

#include <util/generic/array_ref.h>
#include <util/generic/vector.h>

#include <tuple>


//! For computing final formula as `Scale * sumTrees + Bias`
struct TScaleAndBias {
    double Scale = 1.0;
private:
    TVector<double> Bias = {};

public:
    TScaleAndBias() = default;

    TScaleAndBias(double scale, const TVector<double>& bias)
        : Scale(scale)
        , Bias(bias)
    {
    }

    auto AsTuple() const {
        return std::tie(Scale, Bias);
    }

    bool operator==(const TScaleAndBias& other) const {
        return AsTuple() == other.AsTuple();
    }

    bool operator!=(const TScaleAndBias& other) const {
        return !(*this == other);
    }

    bool IsZeroBias() const {
        for (auto x : Bias) {
            if (x != 0.0) {
                return false;
            }
        }
        return true;
    }

    bool IsIdentity() const {
        return Scale == 1.0 && IsZeroBias();
    }

    const TVector<double>& GetBiasRef() const {
        return Bias;
    }

    double GetOneDimensionalBias(TStringBuf errorMessage = "") const {
        CB_ENSURE_INTERNAL(Bias.size() == 1,
            "Asked one-dimensional bias, has " << Bias.size() << "." << errorMessage);
        return Bias[0];
    }

    double GetOneDimensionalBiasOrZero(TStringBuf errorMessage = "") const {
        if (IsZeroBias()) {
            return 0;
        }
        CB_ENSURE_INTERNAL(Bias.size() == 1,
                           "Asked one-dimensional bias, has " << Bias.size() << "." << errorMessage);
        return Bias[0];
    }
};

/**
 * Apply Scale to all trees and Bias to first tree
 */
void ApplyScaleAndBias(const TScaleAndBias& scaleAndBias, TArrayRef<double> data, size_t treeStart);

#define CB_ENSURE_IDENTITY(scaleAndBias, forWhat) CB_ENSURE(scaleAndBias.IsIdentity(), "Non-identity {Scale,Bias} for " << forWhat << " is not supported")
#define CB_ENSURE_SCALE_IDENTITY(scaleAndBias, forWhat) CB_ENSURE(scaleAndBias.Scale == 1.0, "Non-identity {Scale} for " << forWhat << " is not supported")
