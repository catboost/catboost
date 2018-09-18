#pragma once

#include <util/system/types.h>

#include <cmath>


namespace NCB {

    /* TODO(akhropov): replace with fast implementation,
     *  some ideas: https://stackoverflow.com/questions/3272424/compute-fast-log-base-2-ceiling
     *  util/generic/bitops/CeilLog2 won't work due to 'CeilLog2(1)=1'
     */
    inline ui32 IntLog2(ui32 values) {
        return (ui32)ceil(log2(values));
    }

}

