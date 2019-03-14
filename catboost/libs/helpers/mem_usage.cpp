#include "mem_usage.h"

#include <util/stream/format.h>


void OutputWarningIfCpuRamUsageOverLimit(ui64 cpuRamUsage, ui64 cpuRamLimit) {
    if (cpuRamUsage > cpuRamLimit) {
        CATBOOST_WARNING_LOG << "CatBoost is using more CPU RAM ("
            << HumanReadableSize(cpuRamUsage, SF_BYTES)
            << ") than the limit (" << HumanReadableSize(cpuRamLimit, SF_BYTES)
            << ")\n";
    }
}
