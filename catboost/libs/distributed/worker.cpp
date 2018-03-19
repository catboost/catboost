#include "worker.h"
#include "data_types.h"

#include <library/par/par.h>
#include <library/par/par_util.h>

#include <library/par/par_settings.h>

void RunWorker(ui32 numThreads, ui32 nodePort) {
    NPar::TParNetworkSettings::GetRef().RequesterType = NPar::TParNetworkSettings::ERequesterType::NEH; // avoid Netliba
    NPar::RunSlave(numThreads, nodePort);
}

void SetWorkerCustomObjective(const TMaybe<TCustomObjectiveDescriptor>& objective) {
    NCatboostDistributed::TLocalTensorSearchData::GetRef().Objective = objective;
}

void SetWorkerParams(const NCatboostOptions::TCatBoostOptions& params) {
    NCatboostDistributed::TLocalTensorSearchData::GetRef().Params = params;
}
