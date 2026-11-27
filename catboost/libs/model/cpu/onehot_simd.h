#pragma once

#include <util/system/types.h>

namespace NCB::NModelEvaluation::NDetail {
    using TOneHotFunction = void (*)(const ui32*, size_t, const int*, size_t, ui8*);
    TOneHotFunction GetOneHotAvx2() noexcept;
    TOneHotFunction GetOneHotAvx512() noexcept;
    TOneHotFunction GetWideOneHot() noexcept;
} // namespace NCB::NModelEvaluation::NDetail
