#include "dcg.h"

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_kernel_buffer.h>
#include <catboost/cuda/cuda_util/dot_product.h>
#include <catboost/cuda/cuda_util/fill.h>
#include <catboost/cuda/cuda_util/helpers.h>
#include <catboost/cuda/cuda_util/sort.h>
#include <catboost/cuda/cuda_util/transform.h>
#include <catboost/cuda/targets/kernel/dcg.cuh>
#include <catboost/libs/helpers/exception.h>

#include <util/stream/labeled.h>

#include <type_traits>

using NCatboostCuda::NDetail::FuseUi32AndFloatIntoUi64;
using NCatboostCuda::NDetail::FuseUi32AndTwoFloatsIntoUi64;
using NCatboostCuda::NDetail::MakeDcgDecay;
using NCatboostCuda::NDetail::MakeDcgExponentialDecay;
using NCudaLib::TMirrorMapping;
using NCudaLib::TSingleMapping;
using NCudaLib::TStripeMapping;
using NKernelHost::TCudaBufferPtr;
using NKernelHost::TCudaStream;
using NKernelHost::TStatelessKernel;

// CalculateNdcg

template <typename TMapping>
static float CalculateNdcgImpl(
    const NCudaLib::TCudaBuffer<const float, TMapping>& targets,
    const NCudaLib::TCudaBuffer<const float, TMapping>& approxes,
    const NCudaLib::TCudaBuffer<const ui32, TMapping>& biasedOffsets,
    const ENdcgMetricType type,
    const TMaybe<float> exponentialDecay,
    ui32 stream)
{
    CB_ENSURE(false, "Not implemented yet; see MLTOOLS-2431");
    (void)targets;
    (void)approxes;
    (void)biasedOffsets;
    (void)type;
    (void)exponentialDecay;
    (void)stream;
    return 0;
}

#define Y_CATBOOST_CUDA_F_IMPL(TMapping)                                                              \
    template <>                                                                                       \
    float NCatboostCuda::CalculateNdcg<TMapping>(                                                     \
        const NCudaLib::TCudaBuffer<const float, TMapping>& targets,                                  \
        const NCudaLib::TCudaBuffer<const float, TMapping>& approxes,                                 \
        const NCudaLib::TCudaBuffer<const ui32, TMapping>& biasedOffsets,                             \
        ENdcgMetricType type,                                                                         \
        TMaybe<float> exponentialDecay,                                                               \
        ui32 stream)                                                                                  \
    {                                                                                                 \
        return ::CalculateNdcgImpl(targets, approxes, biasedOffsets, type, exponentialDecay, stream); \
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
    const NCudaLib::TCudaBuffer<const float, TMapping>& targets,
    const NCudaLib::TCudaBuffer<const ui32, TMapping>& biasedOffsets,
    const ENdcgMetricType type,
    const TMaybe<float> exponentialDecay,
    ui32 stream)
{
    const auto mapping = targets.GetMapping();
    auto tmp = TCudaBuffer<float, TMapping>::Create(mapping);
    auto decay = TCudaBuffer<float, TMapping>::Create(mapping);
    auto fused = TCudaBuffer<ui64, TMapping>::Create(mapping);
    auto indices = TCudaBuffer<ui32, TMapping>::Create(mapping);
    TCudaBuffer<float, TMapping> expTargets;
    if (ENdcgMetricType::Exp == type) {
        expTargets = TCudaBuffer<float, TMapping>::Create(mapping);
    }

    if (exponentialDecay.Defined()) {
        MakeDcgExponentialDecay(biasedOffsets, *exponentialDecay, tmp, stream);
    } else {
        MakeDcgDecay(biasedOffsets, tmp, stream);
    }

    // here we rely on the fact that `biasedOffsets` are sorted in ascending order
    // and since we only want to sort by `targets `while keeping groups at the same place as they
    // were we will negate `targets` (bitwise) thus within a group they will be sorted in descending
    // order.
    FuseUi32AndFloatIntoUi64(biasedOffsets, targets, fused, true, stream);
    MakeSequence(indices, stream);
    RadixSort(fused, indices, false, stream);
    Gather(decay, tmp, indices, stream);

    if (ENdcgMetricType::Exp == type) {
        PowVector(targets, 2.f, expTargets, stream);
        AddVector(expTargets, -1.f, stream);
    }

    const TCudaBuffer<float, TMapping>* weights = nullptr;
    const auto dotProduct = ENdcgMetricType::Exp == type
        ? DotProduct(decay, expTargets, weights, stream)
        : DotProduct(decay, targets, weights, stream);
    return dotProduct;
}

#define Y_CATBOOST_CUDA_F_IMPL(TMapping)                                                    \
    template <>                                                                             \
    float NCatboostCuda::CalculateIdcg<TMapping>(                                           \
        const NCudaLib::TCudaBuffer<const float, TMapping>& targets,                        \
        const NCudaLib::TCudaBuffer<const ui32, TMapping>& biasedOffsets,                   \
        ENdcgMetricType type,                                                               \
        TMaybe<float> exponentialDecay,                                                     \
        ui32 stream)                                                                        \
    {                                                                                       \
        return ::CalculateIdcgImpl(targets, biasedOffsets, type, exponentialDecay, stream); \
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
    const NCudaLib::TCudaBuffer<const float, TMapping>& targets,
    const NCudaLib::TCudaBuffer<const float, TMapping>& approxes,
    const NCudaLib::TCudaBuffer<const ui32, TMapping>& biasedOffsets,
    const ENdcgMetricType type,
    const TMaybe<float> exponentialDecay,
    ui32 stream)
{
    const auto mapping = targets.GetMapping();
    auto tmp = TCudaBuffer<float, TMapping>::Create(mapping);
    auto decay = TCudaBuffer<float, TMapping>::Create(mapping);
    auto fused = TCudaBuffer<ui64, TMapping>::Create(mapping);
    auto indices = TCudaBuffer<ui32, TMapping>::Create(mapping);
    TCudaBuffer<float, TMapping> expTargets;
    if (ENdcgMetricType::Exp == type) {
        expTargets = TCudaBuffer<float, TMapping>::Create(mapping);
    }

    if (exponentialDecay.Defined()) {
        MakeDcgExponentialDecay(biasedOffsets, *exponentialDecay, tmp, stream);
    } else {
        MakeDcgDecay(biasedOffsets, tmp, stream);
    }

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
    FuseUi32AndTwoFloatsIntoUi64(biasedOffsets, approxes, targets, fused, true, false, stream);
    MakeSequence(indices, stream);
    RadixSort(fused, indices, false, stream);
    Gather(decay, tmp, indices, stream);

    if (ENdcgMetricType::Exp == type) {
        PowVector(targets, 2.f, expTargets, stream);
        AddVector(expTargets, -1.f, stream);
    }

    const TCudaBuffer<float, TMapping>* weights = nullptr;
    const auto dotProduct = ENdcgMetricType::Exp == type
        ? DotProduct(decay, expTargets, weights, stream)
        : DotProduct(decay, targets, weights, stream);
    return dotProduct;
}

#define Y_CATBOOST_CUDA_F_IMPL(TMapping)                                                             \
    template <>                                                                                      \
    float NCatboostCuda::CalculateDcg<TMapping>(                                                     \
        const NCudaLib::TCudaBuffer<const float, TMapping>& targets,                                 \
        const NCudaLib::TCudaBuffer<const float, TMapping>& approxes,                                \
        const NCudaLib::TCudaBuffer<const ui32, TMapping>& biasedOffsets,                            \
        ENdcgMetricType type,                                                                        \
        TMaybe<float> exponentialDecay,                                                              \
        ui32 stream)                                                                                 \
    {                                                                                                \
        return ::CalculateDcgImpl(targets, approxes, biasedOffsets, type, exponentialDecay, stream); \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL,
    TMirrorMapping,
    TSingleMapping,
    TStripeMapping);

#undef Y_CATBOOST_CUDA_F_IMPL

// MakeDcgDecay

namespace {
    template <typename I, typename T>
    class TDcgDecayKernel : public TStatelessKernel {
    private:
        TCudaBufferPtr<const I> BiasedOffsets_;
        TCudaBufferPtr<T> Decay_;

    public:
        Y_SAVELOAD_DEFINE(BiasedOffsets_, Decay_);

        TDcgDecayKernel() = default;
        TDcgDecayKernel(
            TCudaBufferPtr<const I> biasedOffsets,
            TCudaBufferPtr<T> decay)
            : BiasedOffsets_(biasedOffsets)
            , Decay_(decay)
        {
            Y_ASSERT(BiasedOffsets_.Size() == Decay_.Size());
        }

        void Run(const TCudaStream& stream) const {
            NKernel::MakeDcgDecay(BiasedOffsets_.Get(), Decay_.Get(), BiasedOffsets_.Size(), stream);
        }
    };
}

template <typename I, typename T, typename TMapping>
static void MakeDcgDecayImpl(
    const TCudaBuffer<I, TMapping>& biasedOffsets,
    TCudaBuffer<T, TMapping>& decay,
    ui32 stream)
{
    using TKernel = TDcgDecayKernel<I, T>;
    LaunchKernels<TKernel>(biasedOffsets.NonEmptyDevices(), stream, biasedOffsets, decay);
}

#define Y_CATBOOST_CUDA_F_IMPL_PROXY(x) \
    Y_CATBOOST_CUDA_F_IMPL x

#define Y_CATBOOST_CUDA_F_IMPL(I, T, TMapping)                       \
        template <>                                                  \
        void NCatboostCuda::NDetail::MakeDcgDecay<I, T, TMapping>(   \
            const NCudaLib::TCudaBuffer<I, TMapping>& biasedOffsets, \
            NCudaLib::TCudaBuffer<T, TMapping>& decay,               \
            ui32 stream)                                             \
{                                                                    \
    ::MakeDcgDecayImpl(biasedOffsets, decay, stream);                \
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

namespace {
    template <typename I, typename T>
    class TDcgExponentialDecayKernel : public TStatelessKernel {
    private:
        TCudaBufferPtr<const I> BiasedOffsets_;
        T Base_ = 0;
        TCudaBufferPtr<T> Decay_;

    public:
        Y_SAVELOAD_DEFINE(BiasedOffsets_, Base_, Decay_);

        TDcgExponentialDecayKernel() = default;
        TDcgExponentialDecayKernel(
            TCudaBufferPtr<const I> biasedOffsets,
            T base,
            TCudaBufferPtr<T> decay)
            : BiasedOffsets_(biasedOffsets)
            , Base_(base)
            , Decay_(decay)
        {
            Y_ASSERT(BiasedOffsets_.Size() == Decay_.Size());
        }

        void Run(const TCudaStream& stream) const {
            NKernel::MakeDcgExponentialDecay(BiasedOffsets_.Get(), Decay_.Get(), BiasedOffsets_.Size(), Base_, stream);
        }
    };
}

template <typename I, typename T, typename TMapping>
static void MakeDcgExponentialDecayImpl(
    const TCudaBuffer<I, TMapping>& biasedOffsets,
    T base,
    TCudaBuffer<T, TMapping>& decay,
    ui32 stream)
{
    using TKernel = TDcgExponentialDecayKernel<I, T>;
    LaunchKernels<TKernel>(biasedOffsets.NonEmptyDevices(), stream, biasedOffsets, base, decay);
}

#define Y_CATBOOST_CUDA_F_IMPL_PROXY(x) \
    Y_CATBOOST_CUDA_F_IMPL x

#define Y_CATBOOST_CUDA_F_IMPL(I, T, TMapping)                                \
        template <>                                                           \
        void NCatboostCuda::NDetail::MakeDcgExponentialDecay<I, T, TMapping>( \
            const NCudaLib::TCudaBuffer<I, TMapping>& biasedOffsets,          \
            T base,                                                           \
            NCudaLib::TCudaBuffer<T, TMapping>& decay,                        \
            ui32 stream)                                                      \
{                                                                             \
    ::MakeDcgExponentialDecayImpl(biasedOffsets, base, decay, stream);        \
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
    const NCudaLib::TCudaBuffer<I, TMapping>& ui32s,
    const NCudaLib::TCudaBuffer<T, TMapping>& floats,
    NCudaLib::TCudaBuffer<ui64, TMapping>& fused,
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
            const NCudaLib::TCudaBuffer<I, TMapping>& ui32s,                    \
            const NCudaLib::TCudaBuffer<T, TMapping>& floats,                   \
            NCudaLib::TCudaBuffer<ui64, TMapping>& fused,                       \
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
    template <typename T, typename U>
    class TGetBitsKernel : public TStatelessKernel {
    private:
        TCudaBufferPtr<const T> Src_;
        TCudaBufferPtr<U> Dst_;
        ui32 BitsOffset_ = 0;
        ui32 BitsCount_ = 0;

    public:
        Y_SAVELOAD_DEFINE(Src_, Dst_, BitsOffset_, BitsCount_)

        TGetBitsKernel() = default;
        TGetBitsKernel(
            TCudaBufferPtr<const T> src,
            TCudaBufferPtr<U> dst,
            ui32 bitsOffset,
            ui32 bitsCount)
            : Src_(src)
            , Dst_(dst)
            , BitsOffset_(bitsOffset)
            , BitsCount_(bitsCount)
        {
            Y_ASSERT(Src_.Size() == Dst_.Size());
        }

        void Run(const TCudaStream& stream) const {
            NKernel::GetBits(Src_.Get(), Dst_.Get(), Src_.Size(), BitsOffset_, BitsCount_, stream);
        }
    };
}

template <typename T, typename U, typename TMapping>
void GetBitsImpl(
    const NCudaLib::TCudaBuffer<T, TMapping>& src,
    NCudaLib::TCudaBuffer<U, TMapping>& dst,
    ui32 bitsOffset,
    ui32 bitsCount,
    ui32 stream)
{
    CB_ENSURE(bitsCount <= sizeof(T) * 8, LabeledOutput(bitsCount, sizeof(T) * 8));
    CB_ENSURE(bitsCount <= sizeof(U) * 8, LabeledOutput(bitsCount, sizeof(U) * 8));
    CB_ENSURE(bitsOffset <= sizeof(T) * 8, LabeledOutput(bitsOffset, sizeof(T) * 8));

    using TKernel = TGetBitsKernel<const T, U>;
    LaunchKernels<TKernel>(src.NonEmptyDevices(), stream, src, dst, bitsOffset, bitsCount);
}

#define Y_CATBOOST_CUDA_F_IMPL_PROXY(x) \
    Y_CATBOOST_CUDA_F_IMPL x

#define Y_CATBOOST_CUDA_F_IMPL(T, U, TMapping)                \
        template <>                                           \
        void NCatboostCuda::NDetail::GetBits<T, U, TMapping>( \
            const NCudaLib::TCudaBuffer<T, TMapping>& src,    \
            NCudaLib::TCudaBuffer<U, TMapping>& dst,          \
            ui32 bitsOffset,                                  \
            ui32 bitsCount,                                   \
            ui32 stream)                                      \
{                                                             \
    ::GetBitsImpl(src, dst, bitsOffset, bitsCount, stream);   \
}                                                             \

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (ui64, ui32, TMirrorMapping),
    (const ui64, ui32, TMirrorMapping),
    (ui64, ui32, TSingleMapping),
    (const ui64, ui32, TSingleMapping),
    (ui64, ui32, TStripeMapping),
    (const ui64, ui32, TStripeMapping));

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
    const NCudaLib::TCudaBuffer<I, TMapping>& ui32s,
    const NCudaLib::TCudaBuffer<T, TMapping>& floats1,
    const NCudaLib::TCudaBuffer<T, TMapping>& floats2,
    NCudaLib::TCudaBuffer<ui64, TMapping>& fused,
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
            const NCudaLib::TCudaBuffer<I, TMapping>& ui32s,                                                  \
            const NCudaLib::TCudaBuffer<T, TMapping>& floats1,                                                \
            const NCudaLib::TCudaBuffer<T, TMapping>& floats2,                                                \
            NCudaLib::TCudaBuffer<ui64, TMapping>& fused,                                                     \
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

namespace NCudaLib {
    REGISTER_KERNEL_TEMPLATE_2(0x110016, TDcgDecayKernel, ui32, float);

    REGISTER_KERNEL_TEMPLATE_2(0x110017, TDcgExponentialDecayKernel, ui32, float);

    REGISTER_KERNEL(0x110018, TFuseUi32AndFloatIntoUi64);
    REGISTER_KERNEL(0x110019, TFuseUi32AndTwoFloatsIntoUi64);

    REGISTER_KERNEL_TEMPLATE_2(0x110020, TGetBitsKernel, ui64, ui32);
}
