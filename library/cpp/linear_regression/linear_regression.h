#pragma once

#include "linear_model.h"
#include "welford.h"

#include <library/cpp/accurate_accumulate/accurate_accumulate.h>

#include <util/generic/vector.h>
#include <util/generic/hash.h>
#include <util/generic/ylimits.h>
#include <util/generic/ymath.h>

class TFastLinearRegressionSolver {
private:
    TKahanAccumulator<double> SumSquaredGoals;

    TVector<double> LinearizedOLSMatrix;
    TVector<double> OLSVector;

public:
    bool Add(const TVector<double>& features, const double goal, const double weight = 1.);
    TLinearModel Solve() const;
    double SumSquaredErrors() const;
};

class TLinearRegressionSolver {
private:
    double GoalsMean = 0.;
    double GoalsDeviation = 0.;

    TVector<double> FeatureMeans;
    TVector<double> LastMeans;
    TVector<double> NewMeans;
    TVector<double> LinearizedOLSMatrix;

    TVector<double> OLSVector;

    TKahanAccumulator<double> SumWeights;

public:
    bool Add(const TVector<double>& features, const double goal, const double weight = 1.);
    TLinearModel Solve() const;
    double SumSquaredErrors() const;
};

template <typename TStoreType>
class TTypedFastSLRSolver {
private:
    TStoreType SumFeatures = TStoreType();
    TStoreType SumSquaredFeatures = TStoreType();

    TStoreType SumGoals = TStoreType();
    TStoreType SumSquaredGoals = TStoreType();

    TStoreType SumProducts = TStoreType();

    TStoreType SumWeights = TStoreType();

public:
    bool Add(const double feature, const double goal, const double weight = 1.) {
        SumFeatures += feature * weight;
        SumSquaredFeatures += feature * feature * weight;

        SumGoals += goal * weight;
        SumSquaredGoals += goal * goal * weight;

        SumProducts += goal * feature * weight;

        SumWeights += weight;

        return true;
    }

    template <typename TFloatType>
    void Solve(TFloatType& factor, TFloatType& intercept, const double regularizationParameter = 0.1) const {
        if (!(double)SumGoals) {
            factor = intercept = TFloatType();
            return;
        }

        double productsDeviation, featuresDeviation;
        SetupSolutionFactors(productsDeviation, featuresDeviation);

        if (!featuresDeviation) {
            factor = TFloatType();
            intercept = (double)SumGoals / (double)SumWeights;
            return;
        }

        factor = productsDeviation / (featuresDeviation + regularizationParameter);
        intercept = (double)SumGoals / (double)SumWeights - factor * (double)SumFeatures / (double)SumWeights;
    }

    double SumSquaredErrors(const double regularizationParameter = 0.1) const {
        if (!(double)SumWeights) {
            return 0.;
        }

        const double sumGoalSquaredDeviations = (double)SumSquaredGoals - (double)SumGoals / (double)SumWeights * (double)SumGoals;

        double productsDeviation, featuresDeviation;
        SetupSolutionFactors(productsDeviation, featuresDeviation);
        if (!featuresDeviation) {
            return sumGoalSquaredDeviations;
        }

        const double factor = productsDeviation / (featuresDeviation + regularizationParameter);

        const double sumSquaredErrors = factor * factor * featuresDeviation - 2 * factor * productsDeviation + sumGoalSquaredDeviations;
        return Max(0., sumSquaredErrors);
    }

private:
    void SetupSolutionFactors(double& productsDeviation, double& featuresDeviation) const {
        if (!(double)SumWeights) {
            productsDeviation = featuresDeviation = 0.;
            return;
        }

        featuresDeviation = (double)SumSquaredFeatures - (double)SumFeatures / (double)SumWeights * (double)SumFeatures;
        if (!featuresDeviation) {
            return;
        }
        productsDeviation = (double)SumProducts - (double)SumFeatures / (double)SumWeights * (double)SumGoals;
    }
};

using TFastSLRSolver = TTypedFastSLRSolver<double>;
using TKahanSLRSolver = TTypedFastSLRSolver<TKahanAccumulator<double>>;

class TSLRSolver {
private:
    double FeaturesMean = 0.;
    double FeaturesDeviation = 0.;

    double GoalsMean = 0.;
    double GoalsDeviation = 0.;

    TKahanAccumulator<double> SumWeights;

    double Covariation = 0.;

public:
    bool Add(const double feature, const double goal, const double weight = 1.);

    bool Add(const double* featuresBegin, const double* featuresEnd, const double* goalsBegin);
    bool Add(const double* featuresBegin, const double* featuresEnd, const double* goalsBegin, const double* weightsBegin);

    bool Add(const TVector<double>& features, const TVector<double>& goals) {
        Y_ASSERT(features.size() == goals.size());
        return Add(features.data(), features.data() + features.size(), goals.data());
    }

    bool Add(const TVector<double>& features, const TVector<double>& goals, const TVector<double>& weights) {
        Y_ASSERT(features.size() == goals.size() && features.size() == weights.size());
        return Add(features.data(), features.data() + features.size(), goals.data(), weights.data());
    }

    template <typename TFloatType>
    void Solve(TFloatType& factor, TFloatType& intercept, const double regularizationParameter = 0.1) const {
        if (!FeaturesDeviation) {
            factor = 0.;
            intercept = GoalsMean;
            return;
        }

        factor = Covariation / (FeaturesDeviation + regularizationParameter);
        intercept = GoalsMean - factor * FeaturesMean;
    }

    double SumSquaredErrors(const double regularizationParameter = 0.1) const;

    double GetSumWeights() const {
        return SumWeights.Get();
    }
};

template <typename TSLRSolverType>
class TTypedBestSLRSolver {
private:
    TVector<TSLRSolverType> SLRSolvers;

public:
    bool Add(const TVector<double>& features, const double goal, const double weight = 1.) {
        if (SLRSolvers.empty()) {
            SLRSolvers.resize(features.size());
        }

        for (size_t featureNumber = 0; featureNumber < features.size(); ++featureNumber) {
            SLRSolvers[featureNumber].Add(features[featureNumber], goal, weight);
        }

        return true;
    }

    TLinearModel Solve(const double regularizationParameter = 0.1) const {
        const TSLRSolverType* bestSolver = nullptr;
        for (const TSLRSolverType& solver : SLRSolvers) {
            if (!bestSolver || solver.SumSquaredErrors(regularizationParameter) < bestSolver->SumSquaredErrors(regularizationParameter)) {
                bestSolver = &solver;
            }
        }

        TVector<double> coefficients(SLRSolvers.size());
        double intercept = 0.0;
        if (bestSolver) {
            bestSolver->Solve(coefficients[bestSolver - SLRSolvers.begin()], intercept, regularizationParameter);
        }

        TLinearModel model(std::move(coefficients), intercept);
        return model;
    }

    double SumSquaredErrors(const double regularizationParameter = 0.1) const {
        if (SLRSolvers.empty()) {
            return 0.;
        }

        double sse = SLRSolvers.begin()->SumSquaredErrors(regularizationParameter);
        for (const TSLRSolver& solver : SLRSolvers) {
            sse = Min(solver.SumSquaredErrors(regularizationParameter), sse);
        }
        return sse;
    }
};

using TFastBestSLRSolver = TTypedBestSLRSolver<TFastSLRSolver>;
using TKahanBestSLRSolver = TTypedBestSLRSolver<TKahanSLRSolver>;
using TBestSLRSolver = TTypedBestSLRSolver<TSLRSolver>;

enum ETransformationType {
    TT_IDENTITY,
    TT_SIGMA,
};

struct TTransformationParameters {
    double RegressionFactor = 1.;
    double RegressionIntercept = 0.;

    double FeatureOffset = 0.;
    double FeatureNormalizer = 1.;

    Y_SAVELOAD_DEFINE(RegressionFactor,
                      RegressionIntercept,
                      FeatureOffset,
                      FeatureNormalizer);
};

class TFeaturesTransformer {
private:
    ETransformationType TransformationType;
    TTransformationParameters TransformationParameters;

public:
    Y_SAVELOAD_DEFINE(TransformationType, TransformationParameters);

    TFeaturesTransformer() = default;

    TFeaturesTransformer(const ETransformationType transformationType,
                         const TTransformationParameters transformationParameters)
        : TransformationType(transformationType)
        , TransformationParameters(transformationParameters)
    {
    }

    double Transformation(const double value) const {
        switch (TransformationType) {
            case ETransformationType::TT_IDENTITY: {
                return value;
            }
            case ETransformationType::TT_SIGMA: {
                const double valueWithoutOffset = value - TransformationParameters.FeatureOffset;
                const double transformedValue = valueWithoutOffset / (fabs(valueWithoutOffset) + TransformationParameters.FeatureNormalizer);
                return TransformationParameters.RegressionIntercept + TransformationParameters.RegressionFactor * transformedValue;
            }
        }
        Y_ASSERT(0);
        return 0.;
    }
};

class TFeaturesTransformerLearner {
private:
    struct TPoint {
        float Argument;
        float Target;
    };

    float MinimalArgument = Max<float>();
    float MaximalArgument = Min<float>();

    ETransformationType TransformationType;
    TVector<TPoint> Points;

public:
    TFeaturesTransformerLearner(const ETransformationType transformationType)
        : TransformationType(transformationType)
    {
    }

    void Add(const float argument, const float target) {
        Points.push_back(TPoint{argument, target});
        MinimalArgument = Min(MinimalArgument, argument);
        MaximalArgument = Max(MaximalArgument, argument);
    }

    TFeaturesTransformer Solve(const size_t iterationsCount = 100);
};

class TFastFeaturesTransformerLearner {
private:
    ETransformationType TransformationType;

    struct TBucket {
        TMeanCalculator ArgumentsMean;
        TMeanCalculator TargetsMean;
    };

    THashMap<double, TBucket> Buckets;
    double Step;

public:
    TFastFeaturesTransformerLearner(const ETransformationType transformationType, const double step = 0.1)
        : TransformationType(transformationType)
        , Step(step)
    {
    }

    void Add(const float argument, const float target) {
        TBucket& bucket = Buckets[round(argument / Step)];
        bucket.ArgumentsMean.Add(argument);
        bucket.TargetsMean.Add(target);
    }

    TFeaturesTransformer Solve(const size_t iterationsCount = 100) {
        TFeaturesTransformerLearner learner(TransformationType);
        for (auto&& argumentWithBucket : Buckets) {
            const TBucket& bucket = argumentWithBucket.second;
            learner.Add(bucket.ArgumentsMean.GetMean(), bucket.TargetsMean.GetMean());
        }
        return learner.Solve(iterationsCount);
    }
};
