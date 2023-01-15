#pragma once

#include <util/generic/algorithm.h>
#include <util/generic/vector.h>

#include <util/ysaveload.h>

#include <utility>

class TLinearModel {
private:
    TVector<double> Coefficients;
    double Intercept;

public:
    Y_SAVELOAD_DEFINE(Coefficients, Intercept);

    TLinearModel(TVector<double>&& coefficients, const double intercept)
        : Coefficients(std::move(coefficients))
        , Intercept(intercept)
    {
    }

    explicit TLinearModel(size_t featuresCount = 0)
        : Coefficients(featuresCount)
        , Intercept(0.)
    {
    }

    const TVector<double>& GetCoefficients() const {
        return Coefficients;
    }

    double GetIntercept() const {
        return Intercept;
    }

    template <typename T>
    double Prediction(const TVector<T>& features) const {
        return InnerProduct(Coefficients, features, Intercept);
    }
};
