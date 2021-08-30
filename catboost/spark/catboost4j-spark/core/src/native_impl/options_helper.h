#pragma once

#include <util/generic/fwd.h>
#include <util/generic/yexception.h>
#include <util/system/types.h>

namespace NCatboostOptions {
    class TCatBoostOptions;
}


// 'result' is an output param because SWIG cannot return classes without default copy constructor
void InitCatBoostOptions(
    const TString& plainJsonParamsAsString,
    NCatboostOptions::TCatBoostOptions* result
) throw (yexception);

i32 GetOneHotMaxSize(
    i32 maxCategoricalFeaturesUniqValuesOnLearn,
    bool hasLearnTarget,
    const TString& plainJsonParamsAsString
) throw (yexception);
