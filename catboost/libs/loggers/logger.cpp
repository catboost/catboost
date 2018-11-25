#include "logger.h"

void LogAverages(const TProfileResults& profileResults) {
    CATBOOST_NOTICE_LOG << Endl << "Average times:" << Endl;
    if (profileResults.PassedIterations == 0) {
        CATBOOST_NOTICE_LOG << Endl << "No iterations recorded" << Endl;
        return;
    }

    double time = profileResults.OperationToTimeInAllIterations.at("Iteration time") / profileResults.PassedIterations;
    CATBOOST_NOTICE_LOG << "Iteration time: " << FloatToString(time, PREC_NDIGITS, 3) << " sec" << Endl;

    for (const auto& it : profileResults.OperationToTimeInAllIterations) {
        CATBOOST_NOTICE_LOG << it.first << ": "
                             << FloatToString(it.second / profileResults.PassedIterations, PREC_NDIGITS, 3) << " sec" << Endl;
    }
    CATBOOST_NOTICE_LOG << Endl;
}
