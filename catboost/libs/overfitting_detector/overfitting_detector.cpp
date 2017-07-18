#include "overfitting_detector.h"

#include <library/statistics/statistics.h>

void TOverfittingDetectorWilcoxon::AddError(double err) {
    if (Threshold <= 0.0)
        return;
    if (!MaxIsOptimal)
        err = -err;
    if (IsEmpty || err > LocalMax) {
        IsEmpty = false;
        DeltasAfterLocalMax.resize(0);
        LocalMax = err;
    } else {
        DeltasAfterLocalMax.push_back(LastError - err);
    }
    LastError = err;
    UpdatePValue();
}

void TOverfittingDetectorWilcoxon::UpdatePValue() {
    if (DeltasAfterLocalMax.ysize() > IterationsWait) {
        CurrentPValue = NStatistics::Wilcoxon(DeltasAfterLocalMax.begin(), DeltasAfterLocalMax.end());
    } else {
        CurrentPValue = 1.0;
    }
}

void TOverfittingDetectorIncToDec::AddError(double err) {
    if (Threshold <= 0.0)
        return;
    if (!MaxIsOptimal)
        err = -err;

    if (IsEmpty || err > LocalMax) {
        if (IsEmpty) {
            IsEmpty = false;
            ExpectedInc = 0;
        }
        LocalMax = err;
        IterationsFromLocalMax = 0;
    } else {
        IterationsFromLocalMax++;
    }

    Errors.push_front(err);
    if (Errors.ysize() > ITERATION_FORGET) {
        Errors.pop_back();
    }

    ExpectedInc *= LAMBDA_FORGET;
    double curMult = 1.0;
    for (int i = 0; i < Errors.ysize(); ++i) {
        ExpectedInc = Max(ExpectedInc, curMult * (err - Errors[i]));
        curMult *= LAMBDA_FORGET;
    }

    LastError = err;
    UpdatePValue();
}

void TOverfittingDetectorIncToDec::UpdatePValue() {
    if (IterationsFromLocalMax > IterationsWait) {
        CurrentPValue = ExpectedInc / Max(LocalMax - LastError, EPS);
        CurrentPValue = exp(-LAMBDA_SCALE / Max(CurrentPValue, EPS));
    } else {
        CurrentPValue = 1.0;
    }
}
