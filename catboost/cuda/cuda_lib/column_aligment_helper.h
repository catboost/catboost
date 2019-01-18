#pragma once

#include "helpers.h"
#include <util/system/types.h>

namespace NCudaLib {
    namespace NAligment {
        inline constexpr ui64 GetColumnAligment() {
            return 256;
        }

        inline constexpr ui64 AlignedColumnSize(ui64 size) {
            const ui64 aligment = GetColumnAligment();
            return ::NHelpers::CeilDivide(size, aligment) * aligment;
        }

        inline constexpr ui64 ColumnShift(ui64 columnSize, ui64 columnId) {
            return AlignedColumnSize(columnSize) * columnId;
        }

        inline constexpr ui64 GetMemorySize(ui64 columnSize, ui64 columnCount) {
            return AlignedColumnSize(columnSize) * columnCount;
        }

    }
}
