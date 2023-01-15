#include <library/cpp/linear_regression/linear_regression.h>
#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/vector.h>
#include <util/generic/ymath.h>
#include <util/random/random.h>

#include <util/system/defaults.h>

namespace {
    void ValueIsCorrect(const double value, const double expectedValue, double possibleRelativeError) {
        UNIT_ASSERT_DOUBLES_EQUAL(value, expectedValue, possibleRelativeError * expectedValue);
    }
}

Y_UNIT_TEST_SUITE(TLinearRegressionTest) {
    Y_UNIT_TEST(MeanAndDeviationTest) {
        TVector<double> arguments;
        TVector<double> weights;

        const size_t argumentsCount = 100;
        for (size_t i = 0; i < argumentsCount; ++i) {
            arguments.push_back(i);
            weights.push_back(i);
        }

        TDeviationCalculator deviationCalculator;
        TMeanCalculator meanCalculator;
        for (size_t i = 0; i < arguments.size(); ++i) {
            meanCalculator.Add(arguments[i], weights[i]);
            deviationCalculator.Add(arguments[i], weights[i]);
        }

        double actualMean = InnerProduct(arguments, weights) / Accumulate(weights, 0.0);
        double actualDeviation = 0.;
        for (size_t i = 0; i < arguments.size(); ++i) {
            double deviation = arguments[i] - actualMean;
            actualDeviation += deviation * deviation * weights[i];
        }

        UNIT_ASSERT(IsValidFloat(meanCalculator.GetMean()));
        UNIT_ASSERT_DOUBLES_EQUAL(meanCalculator.GetMean(), actualMean, 1e-10);

        UNIT_ASSERT(IsValidFloat(deviationCalculator.GetDeviation()));
        UNIT_ASSERT_DOUBLES_EQUAL(meanCalculator.GetMean(), deviationCalculator.GetMean(), 0);

        UNIT_ASSERT(IsValidFloat(meanCalculator.GetSumWeights()));
        UNIT_ASSERT(IsValidFloat(deviationCalculator.GetSumWeights()));
        UNIT_ASSERT_DOUBLES_EQUAL(meanCalculator.GetSumWeights(), deviationCalculator.GetSumWeights(), 0);
        UNIT_ASSERT_DOUBLES_EQUAL(meanCalculator.GetSumWeights(), Accumulate(weights, 0.0), 0);

        ValueIsCorrect(deviationCalculator.GetDeviation(), actualDeviation, 1e-5);

        TMeanCalculator checkRemovingMeanCalculator;
        TDeviationCalculator checkRemovingDeviationCalculator;

        const size_t argumentsToRemoveCount = argumentsCount / 3;
        for (size_t i = 0; i < argumentsCount; ++i) {
            if (i < argumentsToRemoveCount) {
                meanCalculator.Remove(arguments[i], weights[i]);
                deviationCalculator.Remove(arguments[i], weights[i]);
            } else {
                checkRemovingMeanCalculator.Add(arguments[i], weights[i]);
                checkRemovingDeviationCalculator.Add(arguments[i], weights[i]);
            }
        }

        UNIT_ASSERT(IsValidFloat(meanCalculator.GetMean()));
        UNIT_ASSERT(IsValidFloat(checkRemovingMeanCalculator.GetMean()));

        UNIT_ASSERT(IsValidFloat(deviationCalculator.GetDeviation()));
        UNIT_ASSERT(IsValidFloat(checkRemovingDeviationCalculator.GetDeviation()));

        UNIT_ASSERT_DOUBLES_EQUAL(meanCalculator.GetMean(), deviationCalculator.GetMean(), 0);
        UNIT_ASSERT_DOUBLES_EQUAL(meanCalculator.GetMean(), checkRemovingMeanCalculator.GetMean(), 1e-10);

        ValueIsCorrect(deviationCalculator.GetDeviation(), checkRemovingDeviationCalculator.GetDeviation(), 1e-5);
    }

    Y_UNIT_TEST(CovariationTest) {
        TVector<double> firstValues;
        TVector<double> secondValues;
        TVector<double> weights;

        const size_t argumentsCount = 100;
        for (size_t i = 0; i < argumentsCount; ++i) {
            firstValues.push_back(i);
            secondValues.push_back(i * i);
            weights.push_back(i);
        }

        TCovariationCalculator covariationCalculator;
        for (size_t i = 0; i < argumentsCount; ++i) {
            covariationCalculator.Add(firstValues[i], secondValues[i], weights[i]);
        }

        const double firstValuesMean = InnerProduct(firstValues, weights) / Accumulate(weights, 0.0);
        const double secondValuesMean = InnerProduct(secondValues, weights) / Accumulate(weights, 0.0);

        double actualCovariation = 0.;
        for (size_t i = 0; i < argumentsCount; ++i) {
            actualCovariation += (firstValues[i] - firstValuesMean) * (secondValues[i] - secondValuesMean) * weights[i];
        }

        UNIT_ASSERT(IsValidFloat(covariationCalculator.GetCovariation()));
        UNIT_ASSERT(IsValidFloat(covariationCalculator.GetFirstValueMean()));
        UNIT_ASSERT(IsValidFloat(covariationCalculator.GetSecondValueMean()));

        UNIT_ASSERT_DOUBLES_EQUAL(covariationCalculator.GetFirstValueMean(), firstValuesMean, 1e-10);
        UNIT_ASSERT_DOUBLES_EQUAL(covariationCalculator.GetSecondValueMean(), secondValuesMean, 1e-10);

        UNIT_ASSERT(IsValidFloat(covariationCalculator.GetSumWeights()));
        UNIT_ASSERT_DOUBLES_EQUAL(covariationCalculator.GetSumWeights(), Accumulate(weights, 0.0), 0);

        ValueIsCorrect(covariationCalculator.GetCovariation(), actualCovariation, 1e-5);

        TCovariationCalculator checkRemovingCovariationCalculator;

        const size_t argumentsToRemoveCount = argumentsCount / 3;
        for (size_t i = 0; i < argumentsCount; ++i) {
            if (i < argumentsToRemoveCount) {
                covariationCalculator.Remove(firstValues[i], secondValues[i], weights[i]);
            } else {
                checkRemovingCovariationCalculator.Add(firstValues[i], secondValues[i], weights[i]);
            }
        }

        ValueIsCorrect(covariationCalculator.GetCovariation(), checkRemovingCovariationCalculator.GetCovariation(), 1e-5);
    }

    template <typename TSLRSolverType>
    void SLRTest() {
        TVector<double> arguments;
        TVector<double> weights;
        TVector<double> goals;

        const double factor = 2.;
        const double intercept = 105.;
        const double randomError = 0.01;

        const size_t argumentsCount = 10;
        for (size_t i = 0; i < argumentsCount; ++i) {
            arguments.push_back(i);
            weights.push_back(i);
            goals.push_back(arguments.back() * factor + intercept + 2 * (i % 2 - 0.5) * randomError);
        }

        TSLRSolverType slrSolver;
        for (size_t i = 0; i < argumentsCount; ++i) {
            slrSolver.Add(arguments[i], goals[i], weights[i]);
        }

        for (double regularizationThreshold = 0.; regularizationThreshold < 0.05; regularizationThreshold += 0.01) {
            double solutionFactor, solutionIntercept;
            slrSolver.Solve(solutionFactor, solutionIntercept, regularizationThreshold);

            double predictedSumSquaredErrors = slrSolver.SumSquaredErrors(regularizationThreshold);

            UNIT_ASSERT(IsValidFloat(solutionFactor));
            UNIT_ASSERT(IsValidFloat(solutionIntercept));
            UNIT_ASSERT(IsValidFloat(predictedSumSquaredErrors));

            UNIT_ASSERT_DOUBLES_EQUAL(solutionFactor, factor, 1e-2);
            UNIT_ASSERT_DOUBLES_EQUAL(solutionIntercept, intercept, 1e-2);

            double sumSquaredErrors = 0.;
            for (size_t i = 0; i < argumentsCount; ++i) {
                double error = goals[i] - arguments[i] * solutionFactor - solutionIntercept;
                sumSquaredErrors += error * error * weights[i];
            }

            if (!regularizationThreshold) {
                UNIT_ASSERT(predictedSumSquaredErrors < Accumulate(weights, 0.0) * randomError * randomError);
            }
            UNIT_ASSERT_DOUBLES_EQUAL(predictedSumSquaredErrors, sumSquaredErrors, 1e-8);
        }
    }

    Y_UNIT_TEST(FastSLRTest) {
        SLRTest<TFastSLRSolver>();
    }

    Y_UNIT_TEST(KahanSLRTest) {
        SLRTest<TKahanSLRSolver>();
    }

    Y_UNIT_TEST(SLRTest) {
        SLRTest<TSLRSolver>();
    }

    template <typename TLinearRegressionSolverType>
    void LinearRegressionTest() {
        const size_t featuresCount = 10;
        const size_t instancesCount = 10000;
        const double randomError = 0.01;

        TVector<double> coefficients;
        for (size_t featureNumber = 0; featureNumber < featuresCount; ++featureNumber) {
            coefficients.push_back(featureNumber);
        }
        const double intercept = 10;

        TVector<TVector<double>> featuresMatrix;
        TVector<double> goals;
        TVector<double> weights;

        for (size_t instanceNumber = 0; instanceNumber < instancesCount; ++instanceNumber) {
            TVector<double> features;
            for (size_t featureNumber = 0; featureNumber < featuresCount; ++featureNumber) {
                features.push_back(RandomNumber<double>());
            }
            featuresMatrix.push_back(features);

            const double goal = InnerProduct(coefficients, features) + intercept + 2 * (instanceNumber % 2 - 0.5) * randomError;
            goals.push_back(goal);
            weights.push_back(instanceNumber);
        }

        TLinearRegressionSolverType lrSolver;
        for (size_t instanceNumber = 0; instanceNumber < instancesCount; ++instanceNumber) {
            lrSolver.Add(featuresMatrix[instanceNumber], goals[instanceNumber], weights[instanceNumber]);
        }
        const TLinearModel model = lrSolver.Solve();

        for (size_t featureNumber = 0; featureNumber < featuresCount; ++featureNumber) {
            UNIT_ASSERT_DOUBLES_EQUAL(model.GetCoefficients()[featureNumber], coefficients[featureNumber], 1e-2);
        }
        UNIT_ASSERT_DOUBLES_EQUAL(model.GetIntercept(), intercept, 1e-2);

        const double expectedSumSquaredErrors = randomError * randomError * Accumulate(weights, 0.0);
        UNIT_ASSERT_DOUBLES_EQUAL(lrSolver.SumSquaredErrors(), expectedSumSquaredErrors, expectedSumSquaredErrors * 0.01);
    }

    Y_UNIT_TEST(FastLRTest) {
        LinearRegressionTest<TFastLinearRegressionSolver>();
    }

    Y_UNIT_TEST(LRTest) {
        LinearRegressionTest<TLinearRegressionSolver>();
    }

    void TransformationTest(const ETransformationType transformationType, const size_t pointsCount) {
        TVector<float> arguments;
        TVector<float> goals;

        const double regressionFactor = 10.;
        const double regressionIntercept = 100;

        const double featureOffset = -1.5;
        const double featureNormalizer = 15;

        const double left = -100.;
        const double right = +100.;
        const double step = (right - left) / pointsCount;

        for (double argument = left; argument <= right; argument += step) {
            const double goal = regressionIntercept + regressionFactor * (argument - featureOffset) / (fabs(argument - featureOffset) + featureNormalizer);

            arguments.push_back(argument);
            goals.push_back(goal);
        }

        TFastFeaturesTransformerLearner learner(transformationType);
        for (size_t instanceNumber = 0; instanceNumber < arguments.size(); ++instanceNumber) {
            learner.Add(arguments[instanceNumber], goals[instanceNumber]);
        }
        TFeaturesTransformer transformer = learner.Solve();

        double sse = 0.;
        for (size_t instanceNumber = 0; instanceNumber < arguments.size(); ++instanceNumber) {
            const double error = transformer.Transformation(arguments[instanceNumber]) - goals[instanceNumber];
            sse += error * error;
        }
        const double rmse = sqrt(sse / arguments.size());
        UNIT_ASSERT_DOUBLES_EQUAL(rmse, 0., 1e-3);
    }

    Y_UNIT_TEST(SigmaTest100) {
        TransformationTest(ETransformationType::TT_SIGMA, 100);
    }

    Y_UNIT_TEST(SigmaTest1000) {
        TransformationTest(ETransformationType::TT_SIGMA, 1000);
    }

    Y_UNIT_TEST(SigmaTest10000) {
        TransformationTest(ETransformationType::TT_SIGMA, 10000);
    }

    Y_UNIT_TEST(SigmaTest100000) {
        TransformationTest(ETransformationType::TT_SIGMA, 100000);
    }

    Y_UNIT_TEST(SigmaTest1000000) {
        TransformationTest(ETransformationType::TT_SIGMA, 1000000);
    }

    Y_UNIT_TEST(SigmaTest10000000) {
        TransformationTest(ETransformationType::TT_SIGMA, 10000000);
    }

    Y_UNIT_TEST(ResetCalculatorTest) {
        TVector<double> arguments;
        TVector<double> weights;
        const double eps = 1e-10;

        const size_t argumentsCount = 100;
        for (size_t i = 0; i < argumentsCount; ++i) {
            arguments.push_back(i);
            weights.push_back(i);
        }

        TDeviationCalculator deviationCalculator1, deviationCalculator2;
        TMeanCalculator meanCalculator1, meanCalculator2;
        TCovariationCalculator covariationCalculator1, covariationCalculator2;
        for (size_t i = 0; i < arguments.size(); ++i) {
            meanCalculator1.Add(arguments[i], weights[i]);
            meanCalculator2.Add(arguments[i], weights[i]);
            deviationCalculator1.Add(arguments[i], weights[i]);
            deviationCalculator2.Add(arguments[i], weights[i]);
            covariationCalculator1.Add(arguments[i], arguments[arguments.size() - i - 1], weights[i]);
            covariationCalculator2.Add(arguments[i], arguments[arguments.size() - i - 1], weights[i]);
        }

        UNIT_ASSERT_DOUBLES_EQUAL(meanCalculator1.GetMean(), meanCalculator2.GetMean(), eps);
        UNIT_ASSERT_DOUBLES_EQUAL(meanCalculator1.GetSumWeights(), meanCalculator2.GetSumWeights(), eps);

        UNIT_ASSERT_DOUBLES_EQUAL(deviationCalculator1.GetMean(), deviationCalculator2.GetMean(), eps);
        UNIT_ASSERT_DOUBLES_EQUAL(deviationCalculator1.GetDeviation(), deviationCalculator2.GetDeviation(), eps);
        UNIT_ASSERT_DOUBLES_EQUAL(deviationCalculator1.GetStdDev(), deviationCalculator2.GetStdDev(), eps);
        UNIT_ASSERT_DOUBLES_EQUAL(deviationCalculator1.GetSumWeights(), deviationCalculator2.GetSumWeights(), eps);

        UNIT_ASSERT_DOUBLES_EQUAL(covariationCalculator1.GetFirstValueMean(), covariationCalculator2.GetFirstValueMean(), eps);
        UNIT_ASSERT_DOUBLES_EQUAL(covariationCalculator1.GetSecondValueMean(), covariationCalculator2.GetSecondValueMean(), eps);
        UNIT_ASSERT_DOUBLES_EQUAL(covariationCalculator1.GetCovariation(), covariationCalculator2.GetCovariation(), eps);
        UNIT_ASSERT_DOUBLES_EQUAL(covariationCalculator1.GetSumWeights(), covariationCalculator2.GetSumWeights(), eps);

        meanCalculator2.Reset();
        deviationCalculator2.Reset();
        covariationCalculator2.Reset();

        UNIT_ASSERT_DOUBLES_EQUAL(0.0, meanCalculator2.GetMean(), eps);
        UNIT_ASSERT_DOUBLES_EQUAL(0.0, meanCalculator2.GetSumWeights(), eps);

        UNIT_ASSERT_DOUBLES_EQUAL(0.0, deviationCalculator2.GetMean(), eps);
        UNIT_ASSERT_DOUBLES_EQUAL(0.0, deviationCalculator2.GetDeviation(), eps);
        UNIT_ASSERT_DOUBLES_EQUAL(0.0, deviationCalculator2.GetStdDev(), eps);
        UNIT_ASSERT_DOUBLES_EQUAL(0.0, deviationCalculator2.GetSumWeights(), eps);

        UNIT_ASSERT_DOUBLES_EQUAL(0.0, covariationCalculator2.GetFirstValueMean(), eps);
        UNIT_ASSERT_DOUBLES_EQUAL(0.0, covariationCalculator2.GetSecondValueMean(), eps);
        UNIT_ASSERT_DOUBLES_EQUAL(0.0, covariationCalculator2.GetCovariation(), eps);
        UNIT_ASSERT_DOUBLES_EQUAL(0.0, covariationCalculator2.GetSumWeights(), eps);

        for (size_t i = 0; i < arguments.size(); ++i) {
            meanCalculator2.Add(arguments[i], weights[i]);
            deviationCalculator2.Add(arguments[i], weights[i]);
            covariationCalculator2.Add(arguments[i], arguments[arguments.size() - i - 1], weights[i]);
        }

        UNIT_ASSERT_DOUBLES_EQUAL(meanCalculator1.GetMean(), meanCalculator2.GetMean(), 1e-10);
        UNIT_ASSERT_DOUBLES_EQUAL(meanCalculator1.GetSumWeights(), meanCalculator2.GetSumWeights(), 1e-10);

        UNIT_ASSERT_DOUBLES_EQUAL(deviationCalculator1.GetMean(), deviationCalculator2.GetMean(), eps);
        UNIT_ASSERT_DOUBLES_EQUAL(deviationCalculator1.GetDeviation(), deviationCalculator2.GetDeviation(), eps);
        UNIT_ASSERT_DOUBLES_EQUAL(deviationCalculator1.GetStdDev(), deviationCalculator2.GetStdDev(), eps);
        UNIT_ASSERT_DOUBLES_EQUAL(deviationCalculator1.GetSumWeights(), deviationCalculator2.GetSumWeights(), eps);

        UNIT_ASSERT_DOUBLES_EQUAL(covariationCalculator1.GetFirstValueMean(), covariationCalculator2.GetFirstValueMean(), eps);
        UNIT_ASSERT_DOUBLES_EQUAL(covariationCalculator1.GetSecondValueMean(), covariationCalculator2.GetSecondValueMean(), eps);
        UNIT_ASSERT_DOUBLES_EQUAL(covariationCalculator1.GetCovariation(), covariationCalculator2.GetCovariation(), eps);
        UNIT_ASSERT_DOUBLES_EQUAL(covariationCalculator1.GetSumWeights(), covariationCalculator2.GetSumWeights(), eps);
    }
}
