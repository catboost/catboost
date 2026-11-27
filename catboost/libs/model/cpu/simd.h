#pragma once

#include <util/system/types.h>

struct TRepackedBin;

namespace NCB::NModelEvaluation::NDetail {
    // Internal kernels for four shallow trees. docCount must be a multiple of 16;
    // indexes contain unsigned byte leaf IDs. Results need not be SIMD-aligned.
    using TLeafValues4Function = void (*)(
        size_t docCount,
        const double* const* leafValues,
        const ui8* const* indexes,
        double* results);

    // Return nullptr when the compiler, CPU or OS cannot execute the kernel.
    TLeafValues4Function GetLeafValues4Avx2(size_t prefetchDistance = 0) noexcept;
    TLeafValues4Function GetLeafValues4Avx512(size_t prefetchDistance = 0) noexcept;
    TLeafValues4Function GetWideLeafValues4() noexcept;
} // namespace NCB::NModelEvaluation::NDetail

namespace NCB::NModelEvaluation::NDetail {
    template <typename TIndex>
    using TTreeIndexesFunction = void (*)(const ui8*, size_t, TIndex*, const TRepackedBin*, int, bool);
    using TBinarizeFunction = void (*)(const float*, size_t, const float*, size_t, bool, float, ui8*);
    using TScaleBiasFunction = void (*)(double*, size_t, double, double, bool);

    TTreeIndexesFunction<ui8> GetTreeIndexes8Avx2() noexcept;
    TTreeIndexesFunction<ui8> GetTreeIndexes8Avx512() noexcept;
    TTreeIndexesFunction<ui32> GetTreeIndexes32Avx2() noexcept;
    TTreeIndexesFunction<ui32> GetTreeIndexes32Avx512() noexcept;
    TBinarizeFunction GetBinarizeAvx2() noexcept;
    TBinarizeFunction GetBinarizeAvx512() noexcept;
    TScaleBiasFunction GetScaleBiasAvx2() noexcept;
    TScaleBiasFunction GetScaleBiasAvx512() noexcept;

    TTreeIndexesFunction<ui8> GetWideTreeIndexes8() noexcept;
    TTreeIndexesFunction<ui32> GetWideTreeIndexes32() noexcept;
    TBinarizeFunction GetWideBinarize() noexcept;
    TScaleBiasFunction GetWideScaleBias() noexcept;
} // namespace NCB::NModelEvaluation::NDetail
