#include "modes.h"

#include <catboost/libs/dataset_statistics/calculate_statistics.h>


int mode_dataset_statistics(int argc, const char* argv[]) {
    TCalculateStatisticsParams params;
    params.ProcessParams(argc, argv);
    CalculateDatasetStatics(params, params.OutputPath);
    return 0;
}
