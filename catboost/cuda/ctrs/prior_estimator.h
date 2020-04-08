#pragma once

#include <catboost/libs/helpers/dynamic_iterator.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/logging/logging.h>
#include <contrib/libs/gamma_function_apache_math_port/gamma_function.h>
#include <util/generic/vector.h>

class TBetaPriorEstimator {
public:
    struct TBetaPrior {
        TBetaPrior() {
        }

        TBetaPrior(double alpha, double beta)
            : Alpha(alpha)
            , Beta(beta)
        {
        }
        double Alpha = 0;
        double Beta = 0;

        inline double LogLikelihood(double clazz) const {
            return LogLikelihood(static_cast<ui32>(clazz), 1);
        }

        inline double LogLikelihood(ui32 k, ui32 n) const {
            double ll = LogGamma(n + 1) - LogGamma(k + 1) - LogGamma(n - k + 1);
            ll += LogGamma(Alpha + Beta) - LogGamma(Alpha) - LogGamma(Beta);
            ll += LogGamma(Beta + n - k) + LogGamma(Alpha + k) - LogGamma(Alpha + Beta + n);
            return ll;
        }

        TBetaPrior Update(double clazz) const {
            return TBetaPrior(Alpha + clazz, Beta + 1.0 - clazz);
        }

        TBetaPrior Update(ui32 k, ui32 n) const {
            return TBetaPrior(Alpha + k, Beta + n - k);
        }
    };

    using TClassesType = ui8;
    static TBetaPrior EstimateBetaPrior(
        const TClassesType* classes,
        NCB::IDynamicBlockIteratorBasePtr&& quantizedValuesIterator,
        ui32 length,
        size_t uniqueValues,
        ui32 iterations = 50,
        double* resultLikelihood = nullptr
    );

private:
    struct TDerivatives {
        double DerAlpha = 0;
        double DerBeta = 0;

        double Der2Alpha = 0;
        double Der2Beta = 0;
        double Der2AlphaBeta = 0;
    };

    static inline double Sqr(double x) {
        return x * x;
    }

    static TBetaPrior OptimizationStep(const TBetaPrior& point,
                                       const TDerivatives& derivatives,
                                       double step = 1.0,
                                       double l2 = 0.01);

    static double Likelihood(const TVector<double>& positiveCounts, const TVector<double>& counts, const TBetaPrior& point);

    static TDerivatives DerAndDer2(const TVector<double>& positiveCounts, const TVector<double>& counts, const TBetaPrior& point);
};
