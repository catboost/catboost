#pragma once

#include <catboost/private/libs/options/overfitting_detector_options.h>

#include <util/generic/deque.h>
#include <util/generic/vector.h>

class IOverfittingDetector {
public:
    virtual ~IOverfittingDetector() {
    }
    virtual void AddError(double err) = 0;
    virtual bool IsNeedStop() const = 0;
    virtual int GetIterationsWait() const = 0;
    virtual double GetCurrentPValue() const = 0;
    virtual double GetThreshold() const = 0;
    virtual bool GetMaxIsOptimal() const = 0;
    virtual bool IsActive() const = 0;
};

inline bool NeedOverfittingDetection(const IOverfittingDetector* detector) {
    return detector != nullptr && (detector->GetThreshold() > 0.0);
}

inline bool DetectOverfitting(double testError,
                              IOverfittingDetector* detector,
                              TVector<double>* valuesToLog) {
    detector->AddError(testError);
    if (valuesToLog) {
        double pValue = detector->GetCurrentPValue();
        valuesToLog->push_back(pValue);
    }
    return detector->IsNeedStop();
}


THolder<IOverfittingDetector> CreateOverfittingDetector(EOverfittingDetectorType type,
                                                        bool maxIsOptimal,
                                                        double threshold,
                                                        int iterationsWait,
                                                        bool hasTest);

THolder<IOverfittingDetector> CreateOverfittingDetector(const NCatboostOptions::TOverfittingDetectorOptions& options,
                                                        bool maxIsOptimal,
                                                        bool hasTest);
