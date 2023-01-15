#include "linear_model.h"
#include "linear_regression.h"

#include <util/generic/ymath.h>

#ifdef _sse2_
#include <emmintrin.h>
#include <xmmintrin.h>
#endif

#include <algorithm>
#include <functional>

namespace {
    inline void AddFeaturesProduct(const double weight, const TVector<double>& features, TVector<double>& linearizedOLSTriangleMatrix);

    TVector<double> Solve(const TVector<double>& olsMatrix, const TVector<double>& olsVector);

    double SumSquaredErrors(const TVector<double>& olsMatrix,
                            const TVector<double>& olsVector,
                            const TVector<double>& solution,
                            const double goalsDeviation);
}

bool TFastLinearRegressionSolver::Add(const TVector<double>& features, const double goal, const double weight) {
    const size_t featuresCount = features.size();

    if (LinearizedOLSMatrix.empty()) {
        LinearizedOLSMatrix.resize((featuresCount + 1) * (featuresCount + 2) / 2);
        OLSVector.resize(featuresCount + 1);
    }

    AddFeaturesProduct(weight, features, LinearizedOLSMatrix);

    const double weightedGoal = goal * weight;
    double* olsVectorElement = OLSVector.data();
    for (const double feature : features) {
        *olsVectorElement += feature * weightedGoal;
        ++olsVectorElement;
    }
    *olsVectorElement += weightedGoal;

    SumSquaredGoals += goal * goal * weight;

    return true;
}

bool TLinearRegressionSolver::Add(const TVector<double>& features, const double goal, const double weight) {
    const size_t featuresCount = features.size();

    if (FeatureMeans.empty()) {
        FeatureMeans.resize(featuresCount);
        LastMeans.resize(featuresCount);
        NewMeans.resize(featuresCount);

        LinearizedOLSMatrix.resize(featuresCount * (featuresCount + 1) / 2);
        OLSVector.resize(featuresCount);
    }

    SumWeights += weight;
    if (!SumWeights.Get()) {
        return false;
    }

    for (size_t featureNumber = 0; featureNumber < featuresCount; ++featureNumber) {
        const double feature = features[featureNumber];
        double& featureMean = FeatureMeans[featureNumber];

        LastMeans[featureNumber] = weight * (feature - featureMean);
        featureMean += weight * (feature - featureMean) / SumWeights.Get();
        NewMeans[featureNumber] = feature - featureMean;
        ;
    }

    double* olsMatrixElement = LinearizedOLSMatrix.data();

    const double* lastMean = LastMeans.data();
    const double* newMean = NewMeans.data();
    const double* lastMeansEnd = lastMean + LastMeans.size();
    const double* newMeansEnd = newMean + NewMeans.size();

#ifdef _sse2_
    for (; lastMean != lastMeansEnd; ++lastMean, ++newMean) {
        __m128d factor = _mm_set_pd(*lastMean, *lastMean);
        const double* secondFeatureMean = newMean;
        for (; secondFeatureMean + 1 < newMeansEnd; secondFeatureMean += 2, olsMatrixElement += 2) {
            __m128d matrixElem = _mm_loadu_pd(olsMatrixElement);
            __m128d secondFeatureMeanElem = _mm_loadu_pd(secondFeatureMean);
            __m128d product = _mm_mul_pd(factor, secondFeatureMeanElem);
            __m128d addition = _mm_add_pd(matrixElem, product);
            _mm_storeu_pd(olsMatrixElement, addition);
        }
        for (; secondFeatureMean < newMeansEnd; ++secondFeatureMean) {
            *olsMatrixElement++ += *lastMean * *secondFeatureMean;
        }
    }
#else
    for (; lastMean != lastMeansEnd; ++lastMean, ++newMean) {
        for (const double* secondFeatureMean = newMean; secondFeatureMean < newMeansEnd; ++secondFeatureMean) {
            *olsMatrixElement++ += *lastMean * *secondFeatureMean;
        }
    }
#endif

    for (size_t firstFeatureNumber = 0; firstFeatureNumber < features.size(); ++firstFeatureNumber) {
        OLSVector[firstFeatureNumber] += weight * (features[firstFeatureNumber] - FeatureMeans[firstFeatureNumber]) * (goal - GoalsMean);
    }

    const double oldGoalsMean = GoalsMean;
    GoalsMean += weight * (goal - GoalsMean) / SumWeights.Get();
    GoalsDeviation += weight * (goal - oldGoalsMean) * (goal - GoalsMean);

    return true;
}

TLinearModel TFastLinearRegressionSolver::Solve() const {
    TVector<double> coefficients = ::Solve(LinearizedOLSMatrix, OLSVector);
    double intercept = 0.;

    if (!coefficients.empty()) {
        intercept = coefficients.back();
        coefficients.pop_back();
    }

    return TLinearModel(std::move(coefficients), intercept);
}

TLinearModel TLinearRegressionSolver::Solve() const {
    TVector<double> coefficients = ::Solve(LinearizedOLSMatrix, OLSVector);
    double intercept = GoalsMean;

    const size_t featuresCount = OLSVector.size();
    for (size_t featureNumber = 0; featureNumber < featuresCount; ++featureNumber) {
        intercept -= FeatureMeans[featureNumber] * coefficients[featureNumber];
    }

    return TLinearModel(std::move(coefficients), intercept);
}

double TFastLinearRegressionSolver::SumSquaredErrors() const {
    const TVector<double> coefficients = ::Solve(LinearizedOLSMatrix, OLSVector);
    return ::SumSquaredErrors(LinearizedOLSMatrix, OLSVector, coefficients, SumSquaredGoals.Get());
}

double TLinearRegressionSolver::SumSquaredErrors() const {
    const TVector<double> coefficients = ::Solve(LinearizedOLSMatrix, OLSVector);
    return ::SumSquaredErrors(LinearizedOLSMatrix, OLSVector, coefficients, GoalsDeviation);
}

bool TSLRSolver::Add(const double feature, const double goal, const double weight) {
    SumWeights += weight;
    if (!SumWeights.Get()) {
        return false;
    }

    const double weightedFeatureDiff = weight * (feature - FeaturesMean);
    const double weightedGoalDiff = weight * (goal - GoalsMean);

    FeaturesMean += weightedFeatureDiff / SumWeights.Get();
    FeaturesDeviation += weightedFeatureDiff * (feature - FeaturesMean);

    GoalsMean += weightedGoalDiff / SumWeights.Get();
    GoalsDeviation += weightedGoalDiff * (goal - GoalsMean);

    Covariation += weightedFeatureDiff * (goal - GoalsMean);

    return true;
}

bool TSLRSolver::Add(const double* featuresBegin,
                     const double* featuresEnd,
                     const double* goalsBegin) {
    for (; featuresBegin != featuresEnd; ++featuresBegin, ++goalsBegin) {
        Add(*featuresBegin, *goalsBegin);
    }
    return true;
}
bool TSLRSolver::Add(const double* featuresBegin,
                     const double* featuresEnd,
                     const double* goalsBegin,
                     const double* weightsBegin) {
    for (; featuresBegin != featuresEnd; ++featuresBegin, ++goalsBegin, ++weightsBegin) {
        Add(*featuresBegin, *goalsBegin, *weightsBegin);
    }
    return true;
}

double TSLRSolver::SumSquaredErrors(const double regularizationParameter) const {
    double factor, offset;
    Solve(factor, offset, regularizationParameter);

    return factor * factor * FeaturesDeviation - 2 * factor * Covariation + GoalsDeviation;
}

namespace {
    // LDL matrix decomposition, see http://en.wikipedia.org/wiki/Cholesky_decomposition#LDL_decomposition_2
    bool LDLDecomposition(const TVector<double>& linearizedOLSMatrix,
                          const double regularizationThreshold,
                          const double regularizationParameter,
                          TVector<double>& decompositionTrace,
                          TVector<TVector<double>>& decompositionMatrix) {
        const size_t featuresCount = decompositionTrace.size();

        size_t olsMatrixElementIdx = 0;
        for (size_t rowNumber = 0; rowNumber < featuresCount; ++rowNumber) {
            double& decompositionTraceElement = decompositionTrace[rowNumber];
            decompositionTraceElement = linearizedOLSMatrix[olsMatrixElementIdx] + regularizationParameter;

            TVector<double>& decompositionRow = decompositionMatrix[rowNumber];
            for (size_t i = 0; i < rowNumber; ++i) {
                decompositionTraceElement -= decompositionRow[i] * decompositionRow[i] * decompositionTrace[i];
            }

            if (fabs(decompositionTraceElement) < regularizationThreshold) {
                return false;
            }

            ++olsMatrixElementIdx;
            decompositionRow[rowNumber] = 1.;
            for (size_t columnNumber = rowNumber + 1; columnNumber < featuresCount; ++columnNumber) {
                TVector<double>& secondDecompositionRow = decompositionMatrix[columnNumber];
                double& decompositionMatrixElement = secondDecompositionRow[rowNumber];

                decompositionMatrixElement = linearizedOLSMatrix[olsMatrixElementIdx];

                for (size_t j = 0; j < rowNumber; ++j) {
                    decompositionMatrixElement -= decompositionRow[j] * secondDecompositionRow[j] * decompositionTrace[j];
                }

                decompositionMatrixElement /= decompositionTraceElement;

                decompositionRow[columnNumber] = decompositionMatrixElement;
                ++olsMatrixElementIdx;
            }
        }

        return true;
    }

    void LDLDecomposition(const TVector<double>& linearizedOLSMatrix,
                          TVector<double>& decompositionTrace,
                          TVector<TVector<double>>& decompositionMatrix) {
        const double regularizationThreshold = 1e-5;
        double regularizationParameter = 0.;

        while (!LDLDecomposition(linearizedOLSMatrix,
                                 regularizationThreshold,
                                 regularizationParameter,
                                 decompositionTrace,
                                 decompositionMatrix)) {
            regularizationParameter = regularizationParameter ? 2 * regularizationParameter : 1e-5;
        }
    }

    TVector<double> SolveLower(const TVector<TVector<double>>& decompositionMatrix,
                               const TVector<double>& decompositionTrace,
                               const TVector<double>& olsVector) {
        const size_t featuresCount = olsVector.size();

        TVector<double> solution(featuresCount);
        for (size_t featureNumber = 0; featureNumber < featuresCount; ++featureNumber) {
            double& solutionElement = solution[featureNumber];
            solutionElement = olsVector[featureNumber];

            const TVector<double>& decompositionRow = decompositionMatrix[featureNumber];
            for (size_t i = 0; i < featureNumber; ++i) {
                solutionElement -= solution[i] * decompositionRow[i];
            }
        }

        for (size_t featureNumber = 0; featureNumber < featuresCount; ++featureNumber) {
            solution[featureNumber] /= decompositionTrace[featureNumber];
        }

        return solution;
    }

    TVector<double> SolveUpper(const TVector<TVector<double>>& decompositionMatrix,
                               const TVector<double>& lowerSolution) {
        const size_t featuresCount = lowerSolution.size();

        TVector<double> solution(featuresCount);
        for (size_t featureNumber = featuresCount; featureNumber > 0; --featureNumber) {
            double& solutionElement = solution[featureNumber - 1];
            solutionElement = lowerSolution[featureNumber - 1];

            const TVector<double>& decompositionRow = decompositionMatrix[featureNumber - 1];
            for (size_t i = featureNumber; i < featuresCount; ++i) {
                solutionElement -= solution[i] * decompositionRow[i];
            }
        }

        return solution;
    }

    TVector<double> Solve(const TVector<double>& olsMatrix, const TVector<double>& olsVector) {
        const size_t featuresCount = olsVector.size();

        TVector<double> decompositionTrace(featuresCount);
        TVector<TVector<double>> decompositionMatrix(featuresCount, TVector<double>(featuresCount));

        LDLDecomposition(olsMatrix, decompositionTrace, decompositionMatrix);

        return SolveUpper(decompositionMatrix, SolveLower(decompositionMatrix, decompositionTrace, olsVector));
    }

    double SumSquaredErrors(const TVector<double>& olsMatrix,
                            const TVector<double>& olsVector,
                            const TVector<double>& solution,
                            const double goalsDeviation) {
        const size_t featuresCount = olsVector.size();

        double sumSquaredErrors = goalsDeviation;
        size_t olsMatrixElementIdx = 0;
        for (size_t i = 0; i < featuresCount; ++i) {
            sumSquaredErrors += olsMatrix[olsMatrixElementIdx] * solution[i] * solution[i];
            ++olsMatrixElementIdx;
            for (size_t j = i + 1; j < featuresCount; ++j) {
                sumSquaredErrors += 2 * olsMatrix[olsMatrixElementIdx] * solution[i] * solution[j];
                ++olsMatrixElementIdx;
            }
            sumSquaredErrors -= 2 * solution[i] * olsVector[i];
        }
        return sumSquaredErrors;
    }

#ifdef _sse2_
    inline void AddFeaturesProduct(const double weight, const TVector<double>& features, TVector<double>& linearizedOLSTriangleMatrix) {
        const double* leftFeature = features.data();
        const double* featuresEnd = features.data() + features.size();
        double* matrixElement = linearizedOLSTriangleMatrix.data();

        size_t unaligned = features.size() & 0x1;

        for (; leftFeature != featuresEnd; ++leftFeature, ++matrixElement) {
            const double weightedFeature = weight * *leftFeature;
            const double* rightFeature = leftFeature;
            __m128d wf = {weightedFeature, weightedFeature};
            for (size_t i = 0; i < unaligned; ++i, ++rightFeature, ++matrixElement) {
                *matrixElement += weightedFeature * *rightFeature;
            }
            unaligned = (unaligned + 1) & 0x1;
            for (; rightFeature != featuresEnd; rightFeature += 2, matrixElement += 2) {
                __m128d rf = _mm_loadu_pd(rightFeature);
                __m128d matrixRow = _mm_loadu_pd(matrixElement);
                __m128d rowAdd = _mm_mul_pd(rf, wf);
                _mm_storeu_pd(matrixElement, _mm_add_pd(rowAdd, matrixRow));
            }
            *matrixElement += weightedFeature;
        }
        linearizedOLSTriangleMatrix.back() += weight;
    }
#else
    inline void AddFeaturesProduct(const double weight, const TVector<double>& features, TVector<double>& linearizedTriangleMatrix) {
        const double* leftFeature = features.data();
        const double* featuresEnd = features.data() + features.size();
        double* matrixElement = linearizedTriangleMatrix.data();
        for (; leftFeature != featuresEnd; ++leftFeature, ++matrixElement) {
            const double weightedFeature = weight * *leftFeature;
            const double* rightFeature = leftFeature;
            for (; rightFeature != featuresEnd; ++rightFeature, ++matrixElement) {
                *matrixElement += weightedFeature * *rightFeature;
            }
            *matrixElement += weightedFeature;
        }
        linearizedTriangleMatrix.back() += weight;
    }
#endif
}

namespace {
    inline double ArgMinPrecise(std::function<double(double)> func, double left, double right) {
        const size_t intervalsCount = 20;
        double points[intervalsCount + 1];
        double values[intervalsCount + 1];
        while (right > left + 1e-5) {
            for (size_t pointNumber = 0; pointNumber <= intervalsCount; ++pointNumber) {
                points[pointNumber] = left + pointNumber * (right - left) / intervalsCount;
                values[pointNumber] = func(points[pointNumber]);
            }
            size_t bestPointNumber = MinElement(values, values + intervalsCount + 1) - values;
            if (bestPointNumber == 0) {
                right = points[bestPointNumber + 1];
                continue;
            }
            if (bestPointNumber == intervalsCount) {
                left = points[bestPointNumber - 1];
                continue;
            }
            right = points[bestPointNumber + 1];
            left = points[bestPointNumber - 1];
        }
        return func(left) < func(right) ? left : right;
    }
}

TFeaturesTransformer TFeaturesTransformerLearner::Solve(const size_t iterationsCount /* = 100 */) {
    TTransformationParameters transformationParameters;

    auto updateParameter = [this, &transformationParameters](double TTransformationParameters::*parameter,
                                                             const double left,
                                                             const double right) {
        auto evalParameter = [this, &transformationParameters, parameter](double parameterValue) {
            transformationParameters.*parameter = parameterValue;
            TFeaturesTransformer transformer(TransformationType, transformationParameters);

            double sse = 0.;
            for (const TPoint& point : Points) {
                const double error = transformer.Transformation(point.Argument) - point.Target;
                sse += error * error;
            }
            return sse;
        };
        transformationParameters.*parameter = ArgMinPrecise(evalParameter, left, right);
    };

    auto updateRegressionParameters = [this, &transformationParameters]() {
        TFeaturesTransformer transformer(TransformationType, transformationParameters);

        TSLRSolver slrSolver;
        for (const TPoint& point : Points) {
            slrSolver.Add(transformer.Transformation(point.Argument), point.Target);
        }

        double factor, intercept;
        slrSolver.Solve(factor, intercept);

        transformationParameters.RegressionFactor *= factor;
        transformationParameters.RegressionIntercept *= factor;
        transformationParameters.RegressionIntercept += intercept;
    };

    for (size_t iterationNumber = 0; iterationNumber < iterationsCount; ++iterationNumber) {
        updateParameter(&TTransformationParameters::FeatureOffset, MinimalArgument, MaximalArgument);
        updateParameter(&TTransformationParameters::FeatureNormalizer, 0., MaximalArgument - MinimalArgument);
        updateRegressionParameters();
    }

    return TFeaturesTransformer(TransformationType, transformationParameters);
}
