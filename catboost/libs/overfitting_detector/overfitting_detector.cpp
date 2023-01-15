#include "overfitting_detector.h"

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/logging/logging.h>

#include <library/cpp/statistics/statistics.h>

#include <util/system/compiler.h>


class TNoOverfittingDetector : public IOverfittingDetector {
public:
    void AddError(double err) override {
        Y_UNUSED(err);
    }
    bool IsNeedStop() const override {
        return false;
    }
    int GetIterationsWait() const override {
        return 0;
    }
    double GetCurrentPValue() const override {
        return 0.0;
    }
    double GetThreshold() const override {
        return 0.0;
    }
    bool GetMaxIsOptimal() const override {
        return false;
    }
    bool IsActive() const override {
        return false;
    }
};


class TOverfittingDetectorBase : public IOverfittingDetector {
public:
    TOverfittingDetectorBase(bool maxIsOptimal, double threshold, int iterationsWait)
        : IsEmpty(true)
        , Threshold(threshold)
        , MaxIsOptimal(maxIsOptimal)
        , IterationsWait(iterationsWait)
        , CurrentPValue(1.0) {
    }

    int GetIterationsWait() const override {
        return IterationsWait;
    }

    double GetCurrentPValue() const override {
        return CurrentPValue;
    }

    double GetThreshold() const override {
        return Threshold;
    }

    bool GetMaxIsOptimal() const override {
        return MaxIsOptimal;
    }

    bool IsNeedStop() const override {
        return (!IsEmpty) && (CurrentPValue < Threshold);
    }

    bool IsActive() const override {
        return Threshold > 0;
    }

protected:
    bool IsEmpty;

    const double Threshold;
    const bool MaxIsOptimal;
    const int IterationsWait;

    double CurrentPValue;
};

class TOverfittingDetectorWilcoxon : public TOverfittingDetectorBase {
public:
    TOverfittingDetectorWilcoxon(bool maxIsOptimal, double threshold, int iterationsWait, bool hasTest)
        : TOverfittingDetectorBase(maxIsOptimal, hasTest ? threshold : 0, iterationsWait) {
        CB_ENSURE(hasTest || threshold == 0, "No test provided, cannot check overfitting.");
    }


    void AddError(double err) override {
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

private:

    void UpdatePValue() {
        if (DeltasAfterLocalMax.ysize() >= IterationsWait) {
            CurrentPValue = NStatistics::Wilcoxon(DeltasAfterLocalMax.begin(), DeltasAfterLocalMax.end());
        } else {
            CurrentPValue = 1.0;
        }
    }

    TVector<double> DeltasAfterLocalMax;
    double LastError;
    double LocalMax;
};

class TOverfittingDetectorIncToDec : public TOverfittingDetectorBase {
public:
    TOverfittingDetectorIncToDec(bool maxIsOptimal, double threshold, int iterationsWait, bool hasTest)
        : TOverfittingDetectorBase(maxIsOptimal, hasTest ? threshold : 0, iterationsWait) {
    }


    void AddError(double err) override {
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

private:
    const double LAMBDA_FORGET = 0.99;
    const int ITERATION_FORGET = 2000;
    const double LAMBDA_SCALE = 0.5;
    const double EPS = 1e-10;


    void UpdatePValue() {
        if (IterationsFromLocalMax >= IterationsWait) {
            CurrentPValue = ExpectedInc / Max(LocalMax - LastError, EPS);
            CurrentPValue = exp(-LAMBDA_SCALE / Max(CurrentPValue, EPS));
        } else {
            CurrentPValue = 1.0;
        }
    }

    TDeque<double> Errors;

    double LocalMax;
    double ExpectedInc;
    double LastError;

    int IterationsFromLocalMax;
};

THolder<IOverfittingDetector> CreateOverfittingDetector(EOverfittingDetectorType type, bool maxIsOptimal, double threshold, int iterationsWait, bool hasTest) {
    switch (type) {
    case EOverfittingDetectorType::None:
    {
        return MakeHolder<TNoOverfittingDetector>();
    }
    case EOverfittingDetectorType::IncToDec:
    {
        return MakeHolder<TOverfittingDetectorIncToDec>(maxIsOptimal, threshold, iterationsWait, hasTest);
    }
    case EOverfittingDetectorType::Iter:
    {
        return MakeHolder<TOverfittingDetectorIncToDec>(maxIsOptimal, 1.0, iterationsWait, hasTest);
    }
    case EOverfittingDetectorType::Wilcoxon:
    {
        return MakeHolder<TOverfittingDetectorWilcoxon>(maxIsOptimal, threshold, iterationsWait, hasTest);
    }
    default:
    {
        CB_ENSURE(false, "Unknown OD type: " << type);
    }
    }
}

THolder<IOverfittingDetector> CreateOverfittingDetector(const NCatboostOptions::TOverfittingDetectorOptions& options, bool maxIsOptimal, bool hasTest) {
    return CreateOverfittingDetector(options.OverfittingDetectorType, maxIsOptimal, options.AutoStopPValue, options.IterationsWait, hasTest);
}
