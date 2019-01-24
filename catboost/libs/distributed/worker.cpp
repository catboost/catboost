#include "worker.h"

#include <library/par/par.h>
#include <library/par/par_util.h>

#include <library/par/par_settings.h>

void RunWorker(ui32 numThreads, ui32 nodePort) {
    // avoid Netliba
    NPar::TParNetworkSettings::GetRef().RequesterType = NPar::TParNetworkSettings::ERequesterType::NEH;
    NPar::RunSlave(numThreads, nodePort);
}
