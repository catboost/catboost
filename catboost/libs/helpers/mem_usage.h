#pragma once

#include <catboost/libs/logging/logging.h>

#include <util/system/mem_info.h>
#include <util/system/types.h>


inline void DumpMemUsage(const TString& msg) {
    CATBOOST_DEBUG_LOG << "Mem usage: " << msg << ": " << NMemInfo::GetMemInfo().RSS << Endl;
}

void OutputWarningIfCpuRamUsageOverLimit(ui64 cpuRamUsage, ui64 cpuRamLimit);


namespace NCB {

    /* return Free CPU RAM assuming it is used mostly by current process
     * TODO(akhropov): proper GetFreeCpuRam in arcadia/util
     */
    ui64 GetMonopolisticFreeCpuRam();

}

