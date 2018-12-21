#pragma once

// TErrorTracker is responsible for overfit detection.
// Here is a usage example:
/*    TErrorTracker errorTracker(learnSettings.OverfittingDetectorType,
*                                errorFunction.MaxIsOptimal(),
*                                learnSettings.OverfittingDetectorThreshold,
*                                learnSettings.OverfittingDetectorIterationsWait,
*                                learnSettings.SaveBestIteration);
*
*    for (int iteration = 0; iteration < 10; ++iteration) {
*        DoLearn();
*        RecalcTestApprox();
*        double testErr = CalcTestErr();
*
*        TVector<double> valsToLog;
*        errorTracker.AddError(testErr, iteration, testApprox, &valsToLog);
*
*        if (errorTracker.GetIsNeedStop()) {
*            break;
*        }
*    }
*    bestIteration = errorTracker.GetBestIteration();
*/

#include "overfitting_detector.h"

#include <catboost/libs/metrics/metric.h>

#include <util/generic/ymath.h>


class TErrorTracker {
public:
    TErrorTracker(EOverfittingDetectorType type,
                  EMetricBestValue bestValueType,
                  double metricBestValue,
                  double threshold,
                  int iterationsWait,
                  bool findBestIteration,
                  bool hasTest)
    : BestPossibleError(metricBestValue)
    , BestValueType(bestValueType) {
        if (bestValueType == EMetricBestValue::Min || bestValueType == EMetricBestValue::Max) {
            OverfittingDetector = CreateOverfittingDetector(type, bestValueType == EMetricBestValue::Max, threshold, iterationsWait, hasTest);
        }
        IsNeedStop = false;

        FindBestIteration = findBestIteration;
        BestIteration = NotSet;
        BestError = bestValueType == EMetricBestValue::Max ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity();
    }

    // Saves error in overfitting detector. Pushes current pvalue to
    // valuesToLog if it's defined
    void AddError(double error, int iteration, TVector<double>* optionalValuesToLog = nullptr) {
        if (FindBestIteration) {
            if (BestValueType == EMetricBestValue::Min && error < BestError ||
                BestValueType == EMetricBestValue::Max && error > BestError ||
                BestValueType == EMetricBestValue::FixedValue && Abs(error - BestPossibleError) < Abs(BestError - BestPossibleError)) {
                BestError = error;
                BestIteration = iteration;
            }
        }

        if (NeedOverfittingDetection(OverfittingDetector.Get())) {
            IsNeedStop = DetectOverfitting(error, OverfittingDetector.Get(), optionalValuesToLog);
        }
    }

    int GetIsNeedStop() const {
        return IsNeedStop;
    }
    int GetBestIteration() const {
        return BestIteration;
    }
    double GetBestError() const {
        return BestError;
    }
    double GetOverfittingDetectorThreshold() const {
        CB_ENSURE(OverfittingDetector, "No overfitting detector found");
        return OverfittingDetector->GetThreshold();
    }
    int GetOverfittingDetectorIterationsWait() const {
        CB_ENSURE(OverfittingDetector, "No overfitting detector found");
        return OverfittingDetector->GetIterationsWait();
    }
    bool IsUsingTracker() const {
        return (NeedOverfittingDetection(OverfittingDetector.Get()) || FindBestIteration);
    }
    bool IsActive() const {
        CB_ENSURE(OverfittingDetector, "No overfitting detector found");
        return OverfittingDetector->IsActive();
    }
    double GetPValue() const {
        CB_ENSURE(OverfittingDetector, "No overfitting detector found");
        return OverfittingDetector->GetCurrentPValue();
    }

private:
    THolder<IOverfittingDetector> OverfittingDetector;
    bool IsNeedStop;

    bool FindBestIteration;

    double BestError;
    int BestIteration;
    double BestPossibleError;
    EMetricBestValue BestValueType;

    enum {
        NotSet = -1
    };
};


TErrorTracker CreateErrorTracker(const NCatboostOptions::TOverfittingDetectorOptions& odOptions,
                                 double metricBestValue,
                                 EMetricBestValue bestValueType,
                                 bool hasTest);
