#pragma once

#include <catboost/libs/labels/label_converter.h>
#include <catboost/libs/options/catboost_options.h>

#include <util/system/types.h>


namespace NCB {

    ui32 GetApproxDimension(
        const NCatboostOptions::TCatBoostOptions& catBoostOptions,
        const TLabelConverter& labelConverter);

}
