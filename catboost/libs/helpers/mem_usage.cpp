#include "exception.h"
#include "mem_usage.h"

#include <util/stream/format.h>
#include <util/system/info.h>
#include <util/system/yassert.h>


void OutputWarningIfCpuRamUsageOverLimit(ui64 cpuRamUsage, ui64 cpuRamLimit) {
    if (cpuRamUsage > cpuRamLimit) {
        CATBOOST_WARNING_LOG << "CatBoost is using more CPU RAM ("
            << HumanReadableSize(cpuRamUsage, SF_BYTES)
            << ") than the limit (" << HumanReadableSize(cpuRamLimit, SF_BYTES)
            << ")\n";
    }
}

namespace NCB {

    ui64 GetMonopolisticFreeCpuRam() {
        const ui64 totalMemorySize = (ui64)NSystemInfo::TotalMemorySize();
        const ui64 currentProcessRSS = NMemInfo::GetMemInfo().RSS;
        CB_ENSURE(totalMemorySize >= currentProcessRSS, "total memory size < current process RSS");
        return totalMemorySize - currentProcessRSS;
    }

}
