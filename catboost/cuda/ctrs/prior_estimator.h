#pragma once

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

    template <class TClassesType>
    static TBetaPrior EstimateBetaPrior(const TClassesType* classes,
                                        const ui32* bins, ui32 length,
                                        size_t uniqueValues,
                                        ui32 iterations = 50,
                                        double* resultLikelihood = nullptr) {
        TBetaPrior cursor = {0.5, 0.5};
        TVector<double> positiveCounts(uniqueValues);
        TVector<double> totalCounts(uniqueValues);
        {
            double sum = 0;

            for (ui32 i = 0; i < length; ++i) {
                positiveCounts[bins[i]] += classes[i] ? 1.0 : 0.0;
                totalCounts[bins[i]]++;
                sum += classes[i] ? 1.0 : 0.0;
            }
            {
                cursor.Alpha = sum / length;
                cursor.Beta = 1.0 - cursor.Alpha;
            }
        }

        for (ui32 i = 0; i < iterations; ++i) {
            //MATRIXNET_DEBUG_LOG << "Point (" << cursor.Alpha << ", " << cursor.Beta << "), LogLikelihood " << Likelihood(positiveCounts, totalCounts, cursor) << Endl;
            const auto ders = DerAndDer2(positiveCounts, totalCounts, cursor);
            cursor = OptimizationStep(cursor, ders);
            if (sqrt(Sqr(ders.DerAlpha) + Sqr(ders.DerBeta)) < 1e-9) {
                break;
            }
        }
        if (resultLikelihood) {
            (*resultLikelihood) = Likelihood(positiveCounts, totalCounts, cursor);
        }
        return cursor;
    }

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
                                       double l2 = 0.01) {
        double alpha = point.Alpha;
        double beta = point.Beta;
        CB_ENSURE(alpha > 0 && beta > 0, "Error: illegal point");
        //safety adjust for degenerate solutions
        while (alpha < 1e-8 || beta < 1e-8) {
            alpha *= 10;
            beta *= 10;
        }

        TBetaPrior nextPoint = point;
        const double det = ((derivatives.Der2Alpha - l2) * (derivatives.Der2Beta - l2) - derivatives.Der2AlphaBeta * derivatives.Der2AlphaBeta);

        double directionAlpha = -derivatives.DerAlpha;
        double directionBeta = -derivatives.DerBeta;

        if (det > 0) {
            double mult = 1.0 / det;
            directionAlpha = mult * ((derivatives.Der2Beta - l2) * derivatives.DerAlpha - derivatives.Der2AlphaBeta * derivatives.DerBeta);
            directionBeta = mult * ((derivatives.Der2Alpha - l2) * derivatives.DerBeta - derivatives.Der2AlphaBeta * derivatives.DerAlpha);
        }

        do {
            nextPoint.Alpha = alpha - step * directionAlpha;
            nextPoint.Beta = beta - step * directionBeta;
            step /= 2;
        } while (nextPoint.Alpha < 1e-9 || nextPoint.Beta < 1e-9);
        return nextPoint;
    }

    static double Likelihood(const TVector<double>& positiveCounts, const TVector<double>& counts, const TBetaPrior& point) {
        double ll = 0;

        for (ui32 i = 0; i < positiveCounts.size(); ++i) {
            double first = positiveCounts[i];
            double n = counts[i];
            double second = n - first;

            ll += LogGamma(n + 1) - LogGamma(first + 1) - LogGamma(second + 1) + LogGamma(first + point.Alpha) + LogGamma(second + point.Beta) - LogGamma(n + point.Alpha + point.Beta);
        }
        ll += LogGamma(point.Alpha + point.Beta) * positiveCounts.size();
        ll -= (LogGamma(point.Alpha) + LogGamma(point.Beta)) * positiveCounts.size();
        return ll;
    }

    static TDerivatives DerAndDer2(const TVector<double>& positiveCounts, const TVector<double>& counts, const TBetaPrior& point) {
        TDerivatives derivatives{};
        const int k = static_cast<const int>(positiveCounts.size());
        for (int i = 0; i < k; ++i) {
            const double first = positiveCounts[i];
            const double n = counts[i];
            const double second = n - first;

            derivatives.DerAlpha += Digamma(first + point.Alpha);
            derivatives.Der2Alpha += Trigamma(first + point.Alpha);

            derivatives.DerBeta += Digamma(second + point.Beta);
            derivatives.Der2Beta += Trigamma(second + point.Beta);

            {
                const double tmp = Digamma(n + point.Alpha + point.Beta);
                derivatives.DerAlpha -= tmp;
                derivatives.DerBeta -= tmp;
            }

            {
                const double tmp = Trigamma(n + point.Alpha + point.Beta);
                derivatives.Der2Alpha -= tmp;
                derivatives.Der2Beta -= tmp;
                derivatives.Der2AlphaBeta -= tmp;
            }
        }

        {
            const double tmp = Digamma(point.Alpha + point.Beta);

            derivatives.DerAlpha += (tmp - Digamma(point.Alpha)) * k;
            derivatives.DerBeta += (tmp - Digamma(point.Beta)) * k;
        }

        {
            const double tmp = Trigamma(point.Alpha + point.Beta) * k;
            derivatives.Der2AlphaBeta += tmp;
            derivatives.Der2Alpha += tmp - Trigamma(point.Alpha) * k;
            derivatives.Der2Beta += tmp - Trigamma(point.Beta) * k;
        }
        return derivatives;
    }
};
