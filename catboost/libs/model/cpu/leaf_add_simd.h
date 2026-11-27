#pragma once

#include <util/system/types.h>

namespace NCB::NModelEvaluation::NDetail {
    template <typename TIndex>
    using TAddLeafFunction = void (*)(size_t, const double*, const TIndex*, double*);

    TAddLeafFunction<ui8> GetAddLeaf8Avx2() noexcept;
    TAddLeafFunction<ui8> GetAddLeaf8Avx512() noexcept;
    TAddLeafFunction<ui32> GetAddLeaf32Avx2() noexcept;
    TAddLeafFunction<ui32> GetAddLeaf32Avx512() noexcept;

    TAddLeafFunction<ui8> GetWideAddLeaf8() noexcept;
    TAddLeafFunction<ui32> GetWideAddLeaf32() noexcept;
} // namespace NCB::NModelEvaluation::NDetail
