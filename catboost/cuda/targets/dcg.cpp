#include "dcg.h"

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_kernel_buffer.h>
#include <catboost/cuda/cuda_util/dot_product.h>
#include <catboost/cuda/cuda_util/fill.h>
#include <catboost/cuda/cuda_util/helpers.h>
#include <catboost/cuda/cuda_util/reduce.h>
#include <catboost/cuda/cuda_util/segmented_scan.h>
#include <catboost/cuda/cuda_util/sort.h>
#include <catboost/cuda/cuda_util/transform.h>
#include <catboost/cuda/targets/kernel/dcg.cuh>
#include <catboost/libs/helpers/exception.h>

#include <util/stream/labeled.h>

#include <type_traits>

using NCatboostCuda::NDetail::FuseUi32AndFloatIntoUi64;
using NCatboostCuda::NDetail::FuseUi32AndTwoFloatsIntoUi64;
using NCatboostCuda::NDetail::GatherBySizeAndOffset;
using NCatboostCuda::NDetail::MakeDcgDecays;
using NCatboostCuda::NDetail::MakeDcgExponentialDecays;
using NCatboostCuda::NDetail::MakeElementwiseOffsets;
using NCatboostCuda::NDetail::MakeEndOfGroupMarkers;
using NCatboostCuda::NDetail::RemoveGroupMean;
using NCudaLib::TCudaBuffer;
using NCudaLib::TMirrorMapping;
using NCudaLib::TSingleMapping;
using NCudaLib::TStripeMapping;
using NKernelHost::IMemoryManager;
using NKernelHost::TCudaBufferPtr;
using NKernelHost::TCudaStream;
using NKernelHost::TKernelBase;
using NKernelHost::TStatelessKernel;

// CalculateNdcg

template <typename TMapping>
static float CalculateNdcgImpl(
    const TCudaBuffer<const ui32, TMapping>& sizes,
    const TCudaBuffer<const ui32, TMapping>& offsets,
    const TCudaBuffer<const float, TMapping>& targets,
    const TCudaBuffer<const float, TMapping>& approxes,
    const ENdcgMetricType type,
    ui32 stream)
{
    const auto mapping = targets.GetMapping();
    const auto size = mapping.GetObjectsSlice().Size();
    auto tmpFloats1 = TCudaBuffer<float, TMapping>::Create(mapping);
    auto tmpFloats2 = TCudaBuffer<float, TMapping>::Create(mapping);
    auto elementwiseOffsets = TCudaBuffer<ui32, TMapping>::Create(mapping);
    auto endOfGroupMarkers = TCudaBuffer<ui32, TMapping>::Create(mapping);
    auto decays = TCudaBuffer<float, TMapping>::Create(mapping);
    auto fused = TCudaBuffer<ui64, TMapping>::Create(mapping);
    auto indices = TCudaBuffer<ui32, TMapping>::Create(mapping);
    auto dcg = TCudaBuffer<float, TMapping>::Create(sizes.GetMapping());
    auto idcg = TCudaBuffer<float, TMapping>::Create(sizes.GetMapping());
    TCudaBuffer<float, TMapping> expTargets;
    if (ENdcgMetricType::Exp == type) {
        expTargets = TCudaBuffer<float, TMapping>::Create(mapping);
    }

    // Normalize targets and especially approxes before coverting them to float16 (in
    // FuseUi32AndTwoFloatsIntoUi64).
    //
    // We had three choises here:
    // - do $$(x - min) / (max - min)$$ normalization; in this case we will get all zeroes (or only
    //   lowest bit) in exponent, thus when we convert `float32` to `float16` we will surely loose
    //   some information in lowest bits because we had some information in upper bits of exponent.
    //   And we also may overflow when do $$(max - min)$$ evaluation, though we may use `float64`
    //   for it, but it's too expensive to don on GPU.
    // - do $$Max<float> * (x - min) / (max - min)$$ or similar
    //   $$Max<float16> * (x - mean) / (Max(abs(max), abs(min)) - abs(mean))$$ normalization to
    //   utilize all bits of `float16` but to do it without loosing information we againt have to
    //   use `float64` which is tool expensive on GPU
    // - do $$(x - mean)$$ normalization, it will not loose anything in precision (e.g. won't loose
    //   any information), but won't solve problem with values that can't be represented as
    //   `float16`; so it's our choice
    RemoveGroupMean(approxes, sizes, offsets, tmpFloats1, stream);
    RemoveGroupMean(targets, sizes, offsets, tmpFloats2, stream);

    MakeElementwiseOffsets(sizes, offsets, elementwiseOffsets);
    FillBuffer(endOfGroupMarkers, ui32(0), stream);
    MakeEndOfGroupMarkers(sizes, offsets, endOfGroupMarkers);
    MakeDcgDecays(elementwiseOffsets, decays, stream);

    // Calculate DCG per-query metric values

    // we want to sort them using following predicate (based on `CompareDocs`):
    // bool Cmp(lhsOffset, lhsApprox, lhsTarget, rhsOffset, rhsApprox, rhsTarget) {
    //     if (lhsOffset == rhsOffset) {
    //         if (lhsApprox == rhsApprox) {
    //             return lhsTarget < rhsTarget;
    //         }
    //         return lhsApprox > rhsApprox;
    //     }
    //     return lhsOffset < rhsOffset;
    // }
    FuseUi32AndTwoFloatsIntoUi64(elementwiseOffsets, tmpFloats1, tmpFloats2, fused, true, false, stream);
    MakeSequence(indices, stream);
    RadixSort(fused, indices, false, stream);

    if (ENdcgMetricType::Exp == type) {
        PowVector(targets, 2.f, expTargets, stream);
        AddVector(expTargets, -1.f, stream);
        Gather(tmpFloats1, expTargets, indices, stream);
    } else {
        Gather(tmpFloats1, targets, indices, stream);
    }
    MultiplyVector(tmpFloats1, 1.f / size, stream);
    MultiplyVector(tmpFloats1, decays, stream);
    SegmentedScanVector(tmpFloats1, endOfGroupMarkers, tmpFloats2, true, 1, stream);
    GatherBySizeAndOffset(tmpFloats2, sizes, offsets, dcg, stream);

    // Calculate IDCG per-query metric values

    // here we rely on the fact that `biasedOffsets` are sorted in ascending order
    // and since we only want to sort by `targets `while keeping groups at the same place as they
    // were we will negate `targets` (bitwise) thus within a group they will be sorted in descending
    // order.
    FuseUi32AndFloatIntoUi64(elementwiseOffsets, targets, fused, true, stream);
    MakeSequence(indices, stream);
    RadixSort(fused, indices, false, stream);

    if (ENdcgMetricType::Exp == type) {
        // `expTargets` should already be computed at this point
        Gather(tmpFloats1, expTargets, indices, stream);
    } else {
        Gather(tmpFloats1, targets, indices, stream);
    }
    MultiplyVector(tmpFloats1, 1.f / size, stream);
    MultiplyVector(tmpFloats1, decays, stream);
    SegmentedScanVector(tmpFloats1, endOfGroupMarkers, tmpFloats2, true, 1, stream);
    GatherBySizeAndOffset(tmpFloats2, sizes, offsets, idcg, stream);

    // DCG / IDCG
    DivideVector(dcg, idcg, stream);

    return ReduceToHost(dcg, EOperatorType::Sum, stream);
}

#define Y_CATBOOST_CUDA_F_IMPL(TMapping)                                             \
    template <>                                                                      \
    float NCatboostCuda::CalculateNdcg<TMapping>(                                    \
        const TCudaBuffer<const ui32, TMapping>& sizes,                              \
        const TCudaBuffer<const ui32, TMapping>& offsets,                            \
        const TCudaBuffer<const float, TMapping>& targets,                           \
        const TCudaBuffer<const float, TMapping>& approxes,                          \
        ENdcgMetricType type,                                                        \
        ui32 stream)                                                                 \
    {                                                                                \
        return ::CalculateNdcgImpl(sizes, offsets, targets, approxes, type, stream); \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL,
    TMirrorMapping,
    TSingleMapping,
    TStripeMapping);

#undef Y_CATBOOST_CUDA_F_IMPL

// CalculateIDcg

template <typename TMapping>
static float CalculateIdcgImpl(
    const TCudaBuffer<const ui32, TMapping>& sizes,
    const TCudaBuffer<const ui32, TMapping>& offsets,
    const TCudaBuffer<const float, TMapping>& targets,
    const ENdcgMetricType type,
    const TMaybe<float> exponentialDecay,
    ui32 stream)
{
    const auto mapping = targets.GetMapping();
    const auto size = mapping.GetObjectsSlice().Size();
    auto reorderedTargets = TCudaBuffer<float, TMapping>::Create(mapping);
    auto elementwiseOffsets = TCudaBuffer<ui32, TMapping>::Create(mapping);
    auto decays = TCudaBuffer<float, TMapping>::Create(mapping);
    auto fused = TCudaBuffer<ui64, TMapping>::Create(mapping);
    auto indices = TCudaBuffer<ui32, TMapping>::Create(mapping);
    TCudaBuffer<float, TMapping> expTargets;
    if (ENdcgMetricType::Exp == type) {
        expTargets = TCudaBuffer<float, TMapping>::Create(mapping);
    }

    MakeElementwiseOffsets(sizes, offsets, elementwiseOffsets);

    if (exponentialDecay.Defined()) {
        MakeDcgExponentialDecays(elementwiseOffsets, *exponentialDecay, decays, stream);
    } else {
        MakeDcgDecays(elementwiseOffsets, decays, stream);
    }

    // here we rely on the fact that `biasedOffsets` are sorted in ascending order
    // and since we only want to sort by `targets `while keeping groups at the same place as they
    // were we will negate `targets` (bitwise) thus within a group they will be sorted in descending
    // order.
    FuseUi32AndFloatIntoUi64(elementwiseOffsets, targets, fused, true, stream);
    MakeSequence(indices, stream);
    RadixSort(fused, indices, false, stream);

    if (ENdcgMetricType::Exp == type) {
        PowVector(targets, 2.f, expTargets, stream);
        AddVector(expTargets, -1.f, stream);
        Gather(reorderedTargets, expTargets, indices, stream);
    } else {
        Gather(reorderedTargets, targets, indices, stream);
    }
    MultiplyVector(reorderedTargets, 1.f / size, stream);

    const TCudaBuffer<float, TMapping>* weights = nullptr;
    return DotProduct(decays, reorderedTargets, weights, stream) * size;
}

#define Y_CATBOOST_CUDA_F_IMPL(TMapping)                                                     \
    template <>                                                                              \
    float NCatboostCuda::CalculateIdcg<TMapping>(                                            \
        const TCudaBuffer<const ui32, TMapping>& sizes,                                      \
        const TCudaBuffer<const ui32, TMapping>& offsets,                                    \
        const TCudaBuffer<const float, TMapping>& targets,                                   \
        ENdcgMetricType type,                                                                \
        TMaybe<float> exponentialDecay,                                                      \
        ui32 stream) {                                                                       \
        return ::CalculateIdcgImpl(sizes, offsets, targets, type, exponentialDecay, stream); \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL,
    TMirrorMapping,
    TSingleMapping,
    TStripeMapping);

#undef Y_CATBOOST_CUDA_F_IMPL

// CalculateDcg
template <typename TMapping>
static float CalculateDcgImpl(
    const TCudaBuffer<const ui32, TMapping>& sizes,
    const TCudaBuffer<const ui32, TMapping>& offsets,
    const TCudaBuffer<const float, TMapping>& targets,
    const TCudaBuffer<const float, TMapping>& approxes,
    const ENdcgMetricType type,
    const TMaybe<float> exponentialDecay,
    ui32 stream)
{
    const auto mapping = targets.GetMapping();
    const auto size = mapping.GetObjectsSlice().Size();
    auto tmpFloats1 = TCudaBuffer<float, TMapping>::Create(mapping);
    auto tmpFloats2 = TCudaBuffer<float, TMapping>::Create(mapping);
    auto elementwiseOffsets = TCudaBuffer<ui32, TMapping>::Create(mapping);
    auto fused = TCudaBuffer<ui64, TMapping>::Create(mapping);
    auto indices = TCudaBuffer<ui32, TMapping>::Create(mapping);
    TCudaBuffer<float, TMapping> expTargets;
    if (ENdcgMetricType::Exp == type) {
        expTargets = TCudaBuffer<float, TMapping>::Create(mapping);
    }

    // Normalize targets and especially approxes before coverting them to float16 (in
    // FuseUi32AndTwoFloatsIntoUi64)
    RemoveGroupMean(approxes, sizes, offsets, tmpFloats1, stream);
    RemoveGroupMean(targets, sizes, offsets, tmpFloats2, stream);
    MakeElementwiseOffsets(sizes, offsets, elementwiseOffsets);

    // we want to sort them using following predicate (based on `CompareDocs`):
    // bool Cmp(lhsOffset, lhsApprox, lhsTarget, rhsOffset, rhsApprox, rhsTarget) {
    //     if (lhsOffset == rhsOffset) {
    //         if (lhsApprox == rhsApprox) {
    //             return lhsTarget < rhsTarget;
    //         }
    //         return lhsApprox > rhsApprox;
    //     }
    //     return lhsOffset < rhsOffset;
    // }
    FuseUi32AndTwoFloatsIntoUi64(elementwiseOffsets, tmpFloats1, tmpFloats2, fused, true, false, stream);
    MakeSequence(indices, stream);
    RadixSort(fused, indices, false, stream);

    if (exponentialDecay.Defined()) {
        MakeDcgExponentialDecays(elementwiseOffsets, *exponentialDecay, tmpFloats1, stream);
    } else {
        MakeDcgDecays(elementwiseOffsets, tmpFloats1, stream);
    }

    if (ENdcgMetricType::Exp == type) {
        PowVector(targets, 2.f, expTargets, stream);
        AddVector(expTargets, -1.f, stream);
        Gather(tmpFloats2, expTargets, indices, stream);
    } else {
        Gather(tmpFloats2, targets, indices, stream);
    }
    MultiplyVector(tmpFloats2, 1.f / size, stream);

    const TCudaBuffer<float, TMapping>* weights = nullptr;
    return DotProduct(tmpFloats1, tmpFloats2, weights, stream) * size;
}

#define Y_CATBOOST_CUDA_F_IMPL(TMapping)                                                              \
    template <>                                                                                       \
    float NCatboostCuda::CalculateDcg<TMapping>(                                                      \
        const TCudaBuffer<const ui32, TMapping>& sizes,                                               \
        const TCudaBuffer<const ui32, TMapping>& offsets,                                             \
        const TCudaBuffer<const float, TMapping>& targets,                                            \
        const TCudaBuffer<const float, TMapping>& approxes,                                           \
        ENdcgMetricType type,                                                                         \
        TMaybe<float> exponentialDecay,                                                               \
        ui32 stream) {                                                                                \
        return ::CalculateDcgImpl(sizes, offsets, targets, approxes, type, exponentialDecay, stream); \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL,
    TMirrorMapping,
    TSingleMapping,
    TStripeMapping);

#undef Y_CATBOOST_CUDA_F_IMPL

// MakeDcgDecays

namespace {
    template <typename I, typename T>
    class TDcgDecaysKernel : public TStatelessKernel {
    private:
        TCudaBufferPtr<const I> BiasedOffsets_;
        TCudaBufferPtr<T> Decays_;

    public:
        Y_SAVELOAD_DEFINE(BiasedOffsets_, Decays_);

        TDcgDecaysKernel() = default;
        TDcgDecaysKernel(
            TCudaBufferPtr<const I> biasedOffsets,
            TCudaBufferPtr<T> decays)
            : BiasedOffsets_(biasedOffsets)
            , Decays_(decays)
        {
            Y_ASSERT(BiasedOffsets_.Size() == Decays_.Size());
        }

        void Run(const TCudaStream& stream) const {
            NKernel::MakeDcgDecays(BiasedOffsets_.Get(), Decays_.Get(), BiasedOffsets_.Size(), stream);
        }
    };
}

template <typename I, typename T, typename TMapping>
static void MakeDcgDecaysImpl(
    const TCudaBuffer<I, TMapping>& biasedOffsets,
    TCudaBuffer<T, TMapping>& decays,
    ui32 stream)
{
    using TKernel = TDcgDecaysKernel<I, T>;
    LaunchKernels<TKernel>(biasedOffsets.NonEmptyDevices(), stream, biasedOffsets, decays);
}

#define Y_CATBOOST_CUDA_F_IMPL_PROXY(x) \
    Y_CATBOOST_CUDA_F_IMPL x

#define Y_CATBOOST_CUDA_F_IMPL(I, T, TMapping)                       \
        template <>                                                  \
        void NCatboostCuda::NDetail::MakeDcgDecays<I, T, TMapping>(  \
            const TCudaBuffer<I, TMapping>& biasedOffsets,           \
            TCudaBuffer<T, TMapping>& decays,                        \
            ui32 stream)                                             \
{                                                                    \
    ::MakeDcgDecaysImpl(biasedOffsets, decays, stream);              \
}

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (ui32, float, TMirrorMapping),
    (const ui32, float, TMirrorMapping),
    (ui32, float, TSingleMapping),
    (const ui32, float, TSingleMapping),
    (ui32, float, TStripeMapping),
    (const ui32, float, TStripeMapping));

#undef Y_CATBOOST_CUDA_F_IMPL
#undef Y_CATBOOST_CUDA_F_IMPL_PROXY

// MakeDcgExponentialDecays

namespace {
    template <typename I, typename T>
    class TDcgExponentialDecaysKernel : public TStatelessKernel {
    private:
        TCudaBufferPtr<const I> BiasedOffsets_;
        T Base_ = 0;
        TCudaBufferPtr<T> Decays_;

    public:
        Y_SAVELOAD_DEFINE(BiasedOffsets_, Base_, Decays_);

        TDcgExponentialDecaysKernel() = default;
        TDcgExponentialDecaysKernel(
            TCudaBufferPtr<const I> biasedOffsets,
            T base,
            TCudaBufferPtr<T> decays)
            : BiasedOffsets_(biasedOffsets)
            , Base_(base)
            , Decays_(decays)
        {
            Y_ASSERT(BiasedOffsets_.Size() == Decays_.Size());
        }

        void Run(const TCudaStream& stream) const {
            NKernel::MakeDcgExponentialDecays(BiasedOffsets_.Get(), Decays_.Get(), BiasedOffsets_.Size(), Base_, stream);
        }
    };
}

template <typename I, typename T, typename TMapping>
static void MakeDcgExponentialDecaysImpl(
    const TCudaBuffer<I, TMapping>& biasedOffsets,
    T base,
    TCudaBuffer<T, TMapping>& decays,
    ui32 stream)
{
    using TKernel = TDcgExponentialDecaysKernel<I, T>;
    LaunchKernels<TKernel>(biasedOffsets.NonEmptyDevices(), stream, biasedOffsets, base, decays);
}

#define Y_CATBOOST_CUDA_F_IMPL_PROXY(x) \
    Y_CATBOOST_CUDA_F_IMPL x

#define Y_CATBOOST_CUDA_F_IMPL(I, T, TMapping)                               \
    template <>                                                              \
    void NCatboostCuda::NDetail::MakeDcgExponentialDecays<I, T, TMapping>(   \
        const TCudaBuffer<I, TMapping>& biasedOffsets,                       \
        T base,                                                              \
        TCudaBuffer<T, TMapping>& decays,                                    \
        ui32 stream) {                                                       \
        ::MakeDcgExponentialDecaysImpl(biasedOffsets, base, decays, stream); \
    }


Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (ui32, float, TMirrorMapping),
    (const ui32, float, TMirrorMapping),
    (ui32, float, TSingleMapping),
    (const ui32, float, TSingleMapping),
    (ui32, float, TStripeMapping),
    (const ui32, float, TStripeMapping));

#undef Y_CATBOOST_CUDA_F_IMPL
#undef Y_CATBOOST_CUDA_F_IMPL_PROXY

// FuseUi32AndFloatIntoUi64

namespace {
    class TFuseUi32AndFloatIntoUi64 : public TStatelessKernel {
    private:
        TCudaBufferPtr<const ui32> Ui32s_;
        TCudaBufferPtr<const float> Floats_;
        TCudaBufferPtr<ui64> Fused_;
        bool NegateFloats_ = false;

    public:
        Y_SAVELOAD_DEFINE(Ui32s_, Floats_, Fused_);

        TFuseUi32AndFloatIntoUi64() = default;
        TFuseUi32AndFloatIntoUi64(
            TCudaBufferPtr<const ui32> ui32s,
            TCudaBufferPtr<const float> floats,
            TCudaBufferPtr<ui64> fused,
            bool negateFloats)
            : Ui32s_(ui32s)
            , Floats_(floats)
            , Fused_(fused)
            , NegateFloats_(negateFloats)
        {
            Y_ASSERT(Ui32s_.Size() == Floats_.Size());
            Y_ASSERT(Ui32s_.Size() == Fused_.Size());
        }

        void Run(const TCudaStream& stream) const {
            NKernel::FuseUi32AndFloatIntoUi64(Ui32s_.Get(), Floats_.Get(), Ui32s_.Size(), Fused_.Get(), NegateFloats_, stream);
        }
    };
}

template <typename I, typename T, typename TMapping>
static void FuseUi32AndFloatIntoUi64Impl(
    const TCudaBuffer<I, TMapping>& ui32s,
    const TCudaBuffer<T, TMapping>& floats,
    TCudaBuffer<ui64, TMapping>& fused,
    bool negateFloats,
    ui32 stream)
{
    using TKernel = TFuseUi32AndFloatIntoUi64;
    LaunchKernels<TKernel>(ui32s.NonEmptyDevices(), stream, ui32s, floats, fused, negateFloats);
}

#define Y_CATBOOST_CUDA_F_IMPL_PROXY(x) \
    Y_CATBOOST_CUDA_F_IMPL x

#define Y_CATBOOST_CUDA_F_IMPL(I, T, TMapping)                                  \
        template <>                                                             \
        void NCatboostCuda::NDetail::FuseUi32AndFloatIntoUi64<I, T, TMapping>(  \
            const TCudaBuffer<I, TMapping>& ui32s,                              \
            const TCudaBuffer<T, TMapping>& floats,                             \
            TCudaBuffer<ui64, TMapping>& fused,                                 \
            bool negateFloats,                                                  \
            ui32 stream)                                                        \
{                                                                               \
    ::FuseUi32AndFloatIntoUi64Impl(ui32s, floats, fused, negateFloats, stream); \
}                                                                               \

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (ui32, float, TMirrorMapping),
    (const ui32, float, TMirrorMapping),
    (ui32, const float, TMirrorMapping),
    (const ui32, const float, TMirrorMapping),
    (ui32, float, TSingleMapping),
    (const ui32, float, TSingleMapping),
    (ui32, const float, TSingleMapping),
    (const ui32, const float, TSingleMapping),
    (ui32, float, TStripeMapping),
    (const ui32, float, TStripeMapping),
    (ui32, const float, TStripeMapping),
    (const ui32, const float, TStripeMapping));

#undef Y_CATBOOST_CUDA_F_IMPL
#undef Y_CATBOOST_CUDA_F_IMPL_PROXY

namespace {
    class TFuseUi32AndTwoFloatsIntoUi64 : public TStatelessKernel {
    private:
        TCudaBufferPtr<const ui32> Ui32s_;
        TCudaBufferPtr<const float> Floats1_;
        TCudaBufferPtr<const float> Floats2_;
        TCudaBufferPtr<ui64> Fused_;
        bool NegateFloats1_ = false;
        bool NegateFloats2_ = false;

    public:
        Y_SAVELOAD_DEFINE(Ui32s_, Floats1_, Floats2_, Fused_);

        TFuseUi32AndTwoFloatsIntoUi64() = default;
        TFuseUi32AndTwoFloatsIntoUi64(
            TCudaBufferPtr<const ui32> ui32s,
            TCudaBufferPtr<const float> floats1,
            TCudaBufferPtr<const float> floats2,
            TCudaBufferPtr<ui64> fused,
            bool negateFloats1,
            bool negateFloats2)
            : Ui32s_(ui32s)
            , Floats1_(floats1)
            , Floats2_(floats2)
            , Fused_(fused)
            , NegateFloats1_(negateFloats1)
            , NegateFloats2_(negateFloats2)
        {
            Y_ASSERT(Ui32s_.Size() == Floats1_.Size());
            Y_ASSERT(Ui32s_.Size() == Floats2_.Size());
            Y_ASSERT(Ui32s_.Size() == Fused_.Size());
        }

        void Run(const TCudaStream& stream) const {
            NKernel::FuseUi32AndTwoFloatsIntoUi64(Ui32s_.Get(), Floats1_.Get(), Floats2_.Get(), Ui32s_.Size(), Fused_.Get(), NegateFloats1_, NegateFloats2_, stream);
        }
    };
}

template <typename I, typename T, typename TMapping>
static void FuseUi32AndTwoFloatsIntoUi64Impl(
    const TCudaBuffer<I, TMapping>& ui32s,
    const TCudaBuffer<T, TMapping>& floats1,
    const TCudaBuffer<T, TMapping>& floats2,
    TCudaBuffer<ui64, TMapping>& fused,
    bool negateFloats1,
    bool negateFloats2,
    ui32 stream)
{
    using TKernel = TFuseUi32AndTwoFloatsIntoUi64;
    LaunchKernels<TKernel>(ui32s.NonEmptyDevices(), stream, ui32s, floats1, floats2, fused, negateFloats1, negateFloats2);
}

#define Y_CATBOOST_CUDA_F_IMPL_PROXY(x) \
    Y_CATBOOST_CUDA_F_IMPL x

#define Y_CATBOOST_CUDA_F_IMPL(I, T, TMapping)                                                                \
        template <>                                                                                           \
        void NCatboostCuda::NDetail::FuseUi32AndTwoFloatsIntoUi64<I, T, TMapping>(                            \
            const TCudaBuffer<I, TMapping>& ui32s,                                                            \
            const TCudaBuffer<T, TMapping>& floats1,                                                          \
            const TCudaBuffer<T, TMapping>& floats2,                                                          \
            TCudaBuffer<ui64, TMapping>& fused,                                                               \
            bool negateFloats1,                                                                               \
            bool negateFloats2,                                                                               \
            ui32 stream)                                                                                      \
{                                                                                                             \
    ::FuseUi32AndTwoFloatsIntoUi64Impl(ui32s, floats1, floats2, fused, negateFloats1, negateFloats2, stream); \
}                                                                                                             \

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (ui32, float, TMirrorMapping),
    (const ui32, float, TMirrorMapping),
    (ui32, const float, TMirrorMapping),
    (const ui32, const float, TMirrorMapping),
    (ui32, float, TSingleMapping),
    (const ui32, float, TSingleMapping),
    (ui32, const float, TSingleMapping),
    (const ui32, const float, TSingleMapping),
    (ui32, float, TStripeMapping),
    (const ui32, float, TStripeMapping),
    (ui32, const float, TStripeMapping),
    (const ui32, const float, TStripeMapping));

#undef Y_CATBOOST_CUDA_F_IMPL
#undef Y_CATBOOST_CUDA_F_IMPL_PROXY

// MakeElementwiseOffsets

namespace {
    template <typename T>
    class TMakeElementwiseOffsets : public TStatelessKernel {
    private:
        TCudaBufferPtr<const T> Sizes_;
        TCudaBufferPtr<const T> Offsets_;
        TCudaBufferPtr<T> ElementwiseOffsets_;

    public:
        Y_SAVELOAD_DEFINE(Sizes_, Offsets_, ElementwiseOffsets_);

        TMakeElementwiseOffsets() = default;
        TMakeElementwiseOffsets(
            TCudaBufferPtr<const T> sizes,
            TCudaBufferPtr<const T> offsets,
            TCudaBufferPtr<std::remove_const_t<T>> elementwiseOffsets)
            : Sizes_(sizes)
            , Offsets_(offsets)
            , ElementwiseOffsets_(elementwiseOffsets)
        {
            Y_ASSERT(Sizes_.Size() == Offsets_.Size());
        }

        void Run(const TCudaStream& stream) const {
            NKernel::MakeElementwiseOffsets(Sizes_.Get(), Offsets_.Get(), Sizes_.Size(), ElementwiseOffsets_.Get(), ElementwiseOffsets_.Size(), stream);
        }
    };
}

template <typename T, typename TMapping>
static void MakeElementwiseOffsetsImpl(
    const TCudaBuffer<T, TMapping>& sizes,
    const TCudaBuffer<T, TMapping>& offsets,
    TCudaBuffer<std::remove_const_t<T>, TMapping>& elementwiseOffsets,
    ui32 stream)
{
    using TKernel = TMakeElementwiseOffsets<std::remove_const_t<T>>;
    LaunchKernels<TKernel>(sizes.NonEmptyDevices(), stream, sizes, offsets, elementwiseOffsets);
}

#define Y_CATBOOST_CUDA_F_IMPL_PROXY(x) \
    Y_CATBOOST_CUDA_F_IMPL x

#define Y_CATBOOST_CUDA_F_IMPL(T, TMapping)                                       \
        template <>                                                               \
        void NCatboostCuda::NDetail::MakeElementwiseOffsets<T, TMapping>(         \
        const TCudaBuffer<T, TMapping>& sizes,                                    \
        const TCudaBuffer<T, TMapping>& offsets,                                  \
        TCudaBuffer<std::remove_const_t<T>, TMapping>& elementwiseOffsets,        \
        ui32 stream)                                                              \
    {                                                                             \
        ::MakeElementwiseOffsetsImpl(sizes, offsets, elementwiseOffsets, stream); \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (ui32, TMirrorMapping),
    (const ui32, TMirrorMapping),
    (ui32, TSingleMapping),
    (const ui32, TSingleMapping),
    (ui32, TStripeMapping),
    (const ui32, TStripeMapping));

#undef Y_CATBOOST_CUDA_F_IMPL
#undef Y_CATBOOST_CUDA_F_IMPL_PROXY

// MakeEndOfGroupMarkers

namespace {
    template <typename T>
    class TMakeEndOfGroupMarkers : public TStatelessKernel {
    private:
        TCudaBufferPtr<const T> Sizes_;
        TCudaBufferPtr<const T> Offsets_;
        TCudaBufferPtr<T> EndOfGroupMarkers_;

    public:
        Y_SAVELOAD_DEFINE(Sizes_, Offsets_, EndOfGroupMarkers_);

        TMakeEndOfGroupMarkers() = default;
        TMakeEndOfGroupMarkers(
            TCudaBufferPtr<const T> sizes,
            TCudaBufferPtr<const T> offsets,
            TCudaBufferPtr<std::remove_const_t<T>> endOfGroupMarkers)
            : Sizes_(sizes)
            , Offsets_(offsets)
            , EndOfGroupMarkers_(endOfGroupMarkers)
        {
            Y_ASSERT(Sizes_.Size() == Offsets_.Size());
        }

        void Run(const TCudaStream& stream) const {
            NKernel::MakeEndOfGroupMarkers(Sizes_.Get(), Offsets_.Get(), Sizes_.Size(), EndOfGroupMarkers_.Get(), EndOfGroupMarkers_.Size(), stream);
        }
    };
}

template <typename T, typename TMapping>
static void MakeEndOfGroupMarkersImpl(
    const TCudaBuffer<T, TMapping>& sizes,
    const TCudaBuffer<T, TMapping>& offsets,
    TCudaBuffer<std::remove_const_t<T>, TMapping>& endOfGroupMarkers,
    ui32 stream)
{
    using TKernel = TMakeEndOfGroupMarkers<std::remove_const_t<T>>;
    LaunchKernels<TKernel>(sizes.NonEmptyDevices(), stream, sizes, offsets, endOfGroupMarkers);
}

#define Y_CATBOOST_CUDA_F_IMPL_PROXY(x) \
    Y_CATBOOST_CUDA_F_IMPL x

#define Y_CATBOOST_CUDA_F_IMPL(T, TMapping)                                     \
        template <>                                                             \
        void NCatboostCuda::NDetail::MakeEndOfGroupMarkers<T, TMapping>(        \
        const TCudaBuffer<T, TMapping>& sizes,                                  \
        const TCudaBuffer<T, TMapping>& offsets,                                \
        TCudaBuffer<std::remove_const_t<T>, TMapping>& endOfGroupMarkers,       \
        ui32 stream)                                                            \
    {                                                                           \
        ::MakeEndOfGroupMarkersImpl(sizes, offsets, endOfGroupMarkers, stream); \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (ui32, TMirrorMapping),
    (const ui32, TMirrorMapping),
    (ui32, TSingleMapping),
    (const ui32, TSingleMapping),
    (ui32, TStripeMapping),
    (const ui32, TStripeMapping));

#undef Y_CATBOOST_CUDA_F_IMPL
#undef Y_CATBOOST_CUDA_F_IMPL_PROXY

// GatherBySizeAndOffset

namespace {
    template <typename T, typename I>
    class TGatherBySizeAndOffset : public TStatelessKernel {
    private:
        TCudaBufferPtr<const T> Src_;
        TCudaBufferPtr<const I> Sizes_;
        TCudaBufferPtr<const I> Offsets_;
        TCudaBufferPtr<T> Dst_;

    public:
        Y_SAVELOAD_DEFINE(Src_, Sizes_, Offsets_, Dst_);

        TGatherBySizeAndOffset() = default;
        TGatherBySizeAndOffset(
            TCudaBufferPtr<const T> src,
            TCudaBufferPtr<const I> sizes,
            TCudaBufferPtr<const I> offsets,
            TCudaBufferPtr<T> dst)
            : Src_(src)
            , Sizes_(sizes)
            , Offsets_(offsets)
            , Dst_(dst)
        {
            Y_ASSERT(Sizes_.Size() == Offsets_.Size());
            Y_ASSERT(Sizes_.Size() == Dst_.Size());
        }

        void Run(const TCudaStream& stream) const {
            NKernel::GatherBySizeAndOffset(Src_.Get(), Sizes_.Get(), Offsets_.Get(), Sizes_.Size(), Dst_.Get(), stream);
        }
    };
}

template <typename T, typename I, typename TMapping>
static void GatherBySizeAndOffsetImpl(
    const TCudaBuffer<T, TMapping>& src,
    const TCudaBuffer<I, TMapping>& sizes,
    const TCudaBuffer<I, TMapping>& offsets,
    TCudaBuffer<std::remove_const_t<T>, TMapping>& dst,
    ui32 stream)
{
    using TKernel = TGatherBySizeAndOffset<std::remove_const_t<T>, std::remove_const_t<I>>;
    LaunchKernels<TKernel>(src.NonEmptyDevices(), stream, src, sizes, offsets, dst);
}

#define Y_CATBOOST_CUDA_F_IMPL_PROXY(x) \
    Y_CATBOOST_CUDA_F_IMPL x

#define Y_CATBOOST_CUDA_F_IMPL(T, I, TMapping)                          \
    template <>                                                         \
    void NCatboostCuda::NDetail::GatherBySizeAndOffset<T, I, TMapping>( \
        const TCudaBuffer<T, TMapping>& src,                            \
        const TCudaBuffer<I, TMapping>& sizes,                          \
        const TCudaBuffer<I, TMapping>& offsets,                        \
        TCudaBuffer<std::remove_const_t<T>, TMapping>& dst,             \
        ui32 stream) {                                                  \
        ::GatherBySizeAndOffsetImpl(src, sizes, offsets, dst, stream);  \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (float, ui32, TMirrorMapping),
    (float, const ui32, TMirrorMapping),
    (const float, ui32, TMirrorMapping),
    (const float, const ui32, TMirrorMapping),
    (float, ui32, TSingleMapping),
    (float, const ui32, TSingleMapping),
    (const float, ui32, TSingleMapping),
    (const float, const ui32, TSingleMapping),
    (float, ui32, TStripeMapping),
    (float, const ui32, TStripeMapping),
    (const float, ui32, TStripeMapping),
    (const float, const ui32, TStripeMapping));

#undef Y_CATBOOST_CUDA_F_IMPL
#undef Y_CATBOOST_CUDA_F_IMPL_PROXY

// RemoveGroupMean

namespace {
    template <typename T, typename I>
    class TRemoveGroupMean : public TStatelessKernel {
    private:
        TCudaBufferPtr<const T> Values_;
        TCudaBufferPtr<const I> Sizes_;
        TCudaBufferPtr<const I> Offsets_;
        TCudaBufferPtr<T> Normalized_;

    public:
        Y_SAVELOAD_DEFINE(Values_, Sizes_, Offsets_);

        TRemoveGroupMean() = default;
        TRemoveGroupMean(
            TCudaBufferPtr<const T> values,
            TCudaBufferPtr<const I> sizes,
            TCudaBufferPtr<const I> offsets,
            TCudaBufferPtr<T> normalized)
            : Values_(values)
            , Sizes_(sizes)
            , Offsets_(offsets)
            , Normalized_(normalized)
        {
            Y_ASSERT(Sizes_.Size() == Offsets_.Size());
        }

        void Run(const TCudaStream& stream) const {
            NKernel::RemoveGroupMean(Values_.Get(), Values_.Size(), Sizes_.Get(), Offsets_.Get(), Sizes_.Size(), Normalized_.Get(), stream.GetStream());
        }
    };
}

template <typename T, typename I, typename TMapping>
static void RemoveGroupMeanImpl(
    const TCudaBuffer<T, TMapping>& values,
    const TCudaBuffer<I, TMapping>& sizes,
    const TCudaBuffer<I, TMapping>& offsets,
    TCudaBuffer<std::remove_const_t<T>, TMapping>& normalized,
    ui32 stream)
{
    using TKernel = TRemoveGroupMean<std::remove_const_t<T>, std::remove_const_t<I>>;
    LaunchKernels<TKernel>(values.NonEmptyDevices(), stream, values, sizes, offsets, normalized);
}

#define Y_CATBOOST_CUDA_F_IMPL_PROXY(x) \
    Y_CATBOOST_CUDA_F_IMPL x

#define Y_CATBOOST_CUDA_F_IMPL(T, I, TMapping)                             \
    template <>                                                            \
    void NCatboostCuda::NDetail::RemoveGroupMean<T, I, TMapping>(          \
        const TCudaBuffer<T, TMapping>& values,                            \
        const TCudaBuffer<I, TMapping>& sizes,                             \
        const TCudaBuffer<I, TMapping>& offsets,                           \
        TCudaBuffer<std::remove_const_t<T>, TMapping>& normalized,         \
        ui32 stream) {                                                     \
        ::RemoveGroupMeanImpl(values, sizes, offsets, normalized, stream); \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (float, ui32, TMirrorMapping),
    (float, const ui32, TMirrorMapping),
    (const float, ui32, TMirrorMapping),
    (const float, const ui32, TMirrorMapping),
    (float, ui32, TSingleMapping),
    (float, const ui32, TSingleMapping),
    (const float, ui32, TSingleMapping),
    (const float, const ui32, TSingleMapping),
    (float, ui32, TStripeMapping),
    (float, const ui32, TStripeMapping),
    (const float, ui32, TStripeMapping),
    (const float, const ui32, TStripeMapping));

#undef Y_CATBOOST_CUDA_F_IMPL
#undef Y_CATBOOST_CUDA_F_IMPL_PROXY

namespace NCudaLib {
    REGISTER_KERNEL_TEMPLATE_2(0x110016, TDcgDecaysKernel, ui32, float);

    REGISTER_KERNEL_TEMPLATE_2(0x110017, TDcgExponentialDecaysKernel, ui32, float);

    REGISTER_KERNEL(0x110018, TFuseUi32AndFloatIntoUi64);
    REGISTER_KERNEL(0x110019, TFuseUi32AndTwoFloatsIntoUi64);

    REGISTER_KERNEL_TEMPLATE(0x110022, TMakeElementwiseOffsets, ui32);
    REGISTER_KERNEL_TEMPLATE(0x110023, TMakeEndOfGroupMarkers, ui32);

    REGISTER_KERNEL_TEMPLATE_2(0x110024, TGatherBySizeAndOffset, float, ui32);
}
