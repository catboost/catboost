#pragma once

#include <util/generic/array_ref.h>

//! For computing final formula as `Scale * sumTrees + Bias`
struct TScaleAndBias {
    double Scale = 1.0;
    double Bias = 0.0;

    auto AsTie() const {
        return std::tie(Scale, Bias);
    }

    bool operator==(const TScaleAndBias& other) const {
        return AsTie() == other.AsTie();
    }

    bool operator!=(const TScaleAndBias& other) const {
        return !(*this == other);
    }

    bool IsIdentity() const {
        return Scale == 1.0 && Bias == 0.0;
    }
};

void ApplyScaleAndBias(const TScaleAndBias& scaleAndBias, TArrayRef<double> data);

#define CB_ENSURE_IDENTITY(scaleAndBias, forWhat) CB_ENSURE(scaleAndBias.IsIdentity(), "Non-identity {Scale,Bias} for " << forWhat << " is not supported")
