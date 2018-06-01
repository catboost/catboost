#include "logger.h"

void LogAverages(const TProfileResults& profileResults) {
    MATRIXNET_NOTICE_LOG << Endl << "Average times:" << Endl;
    if (profileResults.PassedIterations == 0) {
        MATRIXNET_NOTICE_LOG << Endl << "No iterations recorded" << Endl;
        return;
    }

    double time = 0;
    for (const auto& it : profileResults.OperationToTimeInAllIterations) {
        time += it.second;
    }
    time /= profileResults.PassedIterations;
    MATRIXNET_NOTICE_LOG << "Iteration time: " << FloatToString(time, PREC_NDIGITS, 3) << " sec" << Endl;

    for (const auto& it : profileResults.OperationToTimeInAllIterations) {
        MATRIXNET_NOTICE_LOG << it.first << ": "
                             << FloatToString(it.second / profileResults.PassedIterations, PREC_NDIGITS, 3) << " sec" << Endl;
    }
    MATRIXNET_NOTICE_LOG << Endl;
}
