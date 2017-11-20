#pragma once

// TErrorTracker is responsible for overfit detection in matrixnet.
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

class TErrorTracker {
public:
    TErrorTracker(EOverfittingDetectorType type,
                  bool maxIsOptimal,
                  double threshold,
                  int iterationsWait,
                  bool findBestIteration,
                  bool hasTest) {
        OverfittingDetector = CreateOverfittingDetector(type, maxIsOptimal, threshold, iterationsWait, hasTest);
        IsNeedStop = false;

        FindBestIteration = findBestIteration;
        BestIteration = NotSet;
        BestError = (OverfittingDetector->GetMaxIsOptimal() ? -1e100 : 1e100);
    }

    // Saves error in overfitting detector. Pushes current pvalue to
    // valuesToLog.
    void AddError(double error, int iteration, TVector<double>* valuesToLog) {
        if (FindBestIteration && ((error < BestError) ^ (OverfittingDetector->GetMaxIsOptimal()))) {
            BestError = error;
            BestIteration = iteration;
        }

        if (NeedOverfittingDetection(OverfittingDetector.Get())) {
            IsNeedStop = DetectOverfitting(error, OverfittingDetector.Get(), valuesToLog);
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
        return OverfittingDetector->GetThreshold();
    }
    int GetOverfittingDetectorIterationsWait() const {
        return OverfittingDetector->GetIterationsWait();
    }
    bool IsUsingTracker() const {
        return (NeedOverfittingDetection(OverfittingDetector.Get()) || FindBestIteration);
    }
    bool IsActive() const {
        return OverfittingDetector->IsActive();
    }
    double GetPValue() const {
        return OverfittingDetector->GetCurrentPValue();
    }

private:
    TAutoPtr<IOverfittingDetector> OverfittingDetector;
    bool IsNeedStop;

    bool FindBestIteration;
    double BestError;
    int BestIteration;

    enum {
        NotSet = -1
    };
};
