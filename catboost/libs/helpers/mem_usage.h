#pragma once

#include <util/system/rusage.h>
#include <catboost/libs/logging/logging.h>

inline void DumpMemUsage(const TString& msg) {
    MATRIXNET_DEBUG_LOG << "Mem usage: " << msg << ": " << TRusage::Get().Rss << Endl;
}
