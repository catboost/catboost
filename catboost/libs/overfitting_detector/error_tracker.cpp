#include "error_tracker.h"
#include "overfitting_detector.h"

TErrorTracker CreateErrorTracker(const NCatboostOptions::TOverfittingDetectorOptions& odOptions,
                                 double metricBestValue,
                                 EMetricBestValue bestValueType,
                                 bool hasTest) {

    return TErrorTracker(odOptions.OverfittingDetectorType,
                         bestValueType,
                         metricBestValue,
                         odOptions.AutoStopPValue,
                         odOptions.IterationsWait,
                         true,
                         hasTest);

}
