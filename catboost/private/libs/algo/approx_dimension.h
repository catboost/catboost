#pragma once

#include <util/system/types.h>

class TLabelConverter;

namespace NCatboostOptions {
    class TCatBoostOptions;
}


namespace NCB {

    ui32 GetApproxDimension(
        const NCatboostOptions::TCatBoostOptions& catBoostOptions,
        const TLabelConverter& labelConverter,
        ui32 targetDimension
    );

}
