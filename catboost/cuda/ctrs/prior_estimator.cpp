#include "prior_estimator.h"

#include <catboost/libs/data/columns.h>

TBetaPriorEstimator::TBetaPrior
TBetaPriorEstimator::EstimateBetaPrior(
    const TBetaPriorEstimator::TClassesType* classes,
    NCB::IDynamicBlockIteratorBasePtr&& quantizedValuesIteratorPtr,
    ui32 length,
    size_t uniqueValues,
    ui32 iterations,
    double* resultLikelihood
) {
    TBetaPrior cursor = {0.5, 0.5};
    TVector<double> positiveCounts(uniqueValues);
    TVector<double> totalCounts(uniqueValues);
    NCB::DispatchIteratorType(
        quantizedValuesIteratorPtr.Get(),
        [&] (auto typedIterator) {
            ui32 i = 0;
            double sum = 0;
            while (auto block = typedIterator->Next(4096)) {
                for (ui32 inBlockId = 0; inBlockId < block.size(); ++i, ++inBlockId) {
                    positiveCounts[block[inBlockId]] += classes[i] ? 1.0 : 0.0;
                    totalCounts[block[inBlockId]]++;
                    sum += classes[i] ? 1.0 : 0.0;
                }
            }
            Y_ASSERT(length == i);
            cursor.Alpha = sum / length;
            cursor.Beta = 1.0 - cursor.Alpha;
        }
    );

    for (ui32 i = 0; i < iterations; ++i) {
        //CATBOOST_DEBUG_LOG << "Point (" << cursor.Alpha << ", " << cursor.Beta << "), LogLikelihood " << Likelihood(positiveCounts, totalCounts, cursor) << Endl;
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

TBetaPriorEstimator::TBetaPrior TBetaPriorEstimator::OptimizationStep(const TBetaPriorEstimator::TBetaPrior& point,
                                                                      const TBetaPriorEstimator::TDerivatives& derivatives,
                                                                      double step, double l2) {
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

double TBetaPriorEstimator::Likelihood(const TVector<double>& positiveCounts, const TVector<double>& counts,
                                       const TBetaPriorEstimator::TBetaPrior& point) {
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

TBetaPriorEstimator::TDerivatives
TBetaPriorEstimator::DerAndDer2(const TVector<double>& positiveCounts, const TVector<double>& counts,
                                const TBetaPriorEstimator::TBetaPrior& point) {
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
