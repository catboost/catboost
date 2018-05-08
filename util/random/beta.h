#pragma once

#include <random>
#include <cmath>

template <typename RealType = double>
class TBetaDistribution {
public:
    using TResultType = RealType;

    class TParamType {
    public:
        explicit TParamType(RealType a, RealType b)
            : A(a)
            , B(b)
        {
        }

        RealType GetA() const { return A; }
        RealType GetB() const { return B; }

    private:
        RealType A, B;
    };

    explicit TBetaDistribution(RealType a, RealType b)
        : A(a)
        , B(b)
    {
    }

    explicit TBetaDistribution(const TParamType& param)
        : A(param.GetA())
        , B(param.GetB())
    {
    }

    TParamType param() const {
        return TParamType(GetA(), GetB());
    }

    void param(const TParamType& param) {
        A = TGammaDistType(param.GetA());
        B = TGammaDistType(param.GetB());
    }

    template <typename URNG>
    TResultType operator()(URNG& engine) {
        return generate(engine, A, B);
    }

    template <typename URNG>
    TResultType operator()(URNG& engine, const TParamType& param) {
        TGammaDistType ParamGammaA(param.GetA()), ParamGammaB(param.GetB());
        return generate(engine, ParamGammaA, ParamGammaB);
    }

    RealType GetA() const { return A.alpha(); }
    RealType GetB() const { return B.alpha(); }

private:
    using TGammaDistType = std::gamma_distribution<TResultType>;

    TGammaDistType A, B;

    template <typename URNG>
    TResultType generate(URNG& engine, TGammaDistType& xGamma, TGammaDistType& yGamma) {
        TResultType x = xGamma(engine);
        return x / (x + yGamma(engine));
    }
};

