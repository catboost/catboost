#include "transform.h"

#include <catboost/cuda/cuda_util/kernel/transform.cuh>
#include <catboost/cuda/cuda_lib/cuda_kernel_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/libs/helpers/exception.h>

using NCudaLib::EPtrType;
using NCudaLib::TMirrorMapping;
using NCudaLib::TSingleMapping;
using NCudaLib::TStripeMapping;
using NKernelHost::EBinOpType;
using NKernelHost::EFuncType;
using NKernelHost::EMapCopyType;
using NKernelHost::TCudaBufferPtr;
using NKernelHost::TCudaStream;
using NKernelHost::TDeviceBuffer;
using NKernelHost::TStatelessKernel;

// AddVector

namespace {
    template <typename T>
    class TBinOpKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<T> X;
        TCudaBufferPtr<const T> Y;
        T ConstY;
        EBinOpType OperationType;
        const bool SkipZeroesOnDivide = true;

    public:
        TBinOpKernel() = default;

        TBinOpKernel(TCudaBufferPtr<T> x, TCudaBufferPtr<const T> y, EBinOpType operationType)
            : X(x)
            , Y(y)
            , OperationType(operationType)
        {
        }
        //
        TBinOpKernel(TCudaBufferPtr<T> x, T y, EBinOpType operationType)
            : X(x)
            , Y(TCudaBufferPtr<const T>::Nullptr())
            , ConstY(y)
            , OperationType(operationType)
        {
        }

        void Run(const TCudaStream& stream) const {
            const ui64 size = X.Size();
            CB_ENSURE(X.ObjectCount() == X.Size(), "Error, we support only 1-object bin operations currently");

            {
                using namespace NKernel;
                switch (OperationType) {
                    case EBinOpType::AddVec:
                        AddVector<T>(X.Get(), Y.Get(), size, stream.GetStream());
                        break;
                    case EBinOpType::AddConst:
                        AddVector<T>(X.Get(), ConstY, size, stream.GetStream());
                        break;
                    case EBinOpType::SubVec:
                        SubtractVector<T>(X.Get(), Y.Get(), size, stream.GetStream());
                        break;
                    case EBinOpType::MulVec:
                        MultiplyVector<T>(X.Get(), Y.Get(), size, stream.GetStream());
                        break;
                    case EBinOpType::MulConst:
                        MultiplyVector<T>(X.Get(), ConstY, size, stream.GetStream());
                        break;
                    case EBinOpType::DivVec:
                        DivideVector<T>(X.Get(), Y.Get(), size, SkipZeroesOnDivide, stream.GetStream());
                        break;
                }
            }
        }

        Y_SAVELOAD_DEFINE(X, Y, ConstY, OperationType);
    };
}

template <typename T, typename TMapping>
static void AddVectorImpl(
    TCudaBuffer<std::remove_const_t<T>, TMapping>& x,
    const TCudaBuffer<T, TMapping>& y,
    ui32 stream) {
    using TKernel = TBinOpKernel<std::remove_const_t<T>>;
    LaunchKernels<TKernel>(x.NonEmptyDevices(), stream, x, y, EBinOpType::AddVec);
}

#define Y_CATBOOST_CUDA_F_IMPL_PROXY(x) \
    Y_CATBOOST_CUDA_F_IMPL x

#define Y_CATBOOST_CUDA_F_IMPL(T, TMapping)                \
    template <>                                            \
    void AddVector<T, TMapping>(                           \
        TCudaBuffer<std::remove_const_t<T>, TMapping> & x, \
        const TCudaBuffer<T, TMapping>& y,                 \
        ui32 stream) {                                     \
        ::AddVectorImpl(x, y, stream);                     \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (float, TMirrorMapping),
    (int, TMirrorMapping),
    (ui32, TMirrorMapping),
    (double, TMirrorMapping),
    (ui8, TMirrorMapping),
    (uint2, TMirrorMapping),
    (ui16, TMirrorMapping),
    (float, TSingleMapping),
    (int, TSingleMapping),
    (ui32, TSingleMapping),
    (double, TSingleMapping),
    (ui8, TSingleMapping),
    (uint2, TSingleMapping),
    (ui16, TSingleMapping),
    (float, TStripeMapping),
    (int, TStripeMapping),
    (ui32, TStripeMapping),
    (double, TStripeMapping),
    (ui8, TStripeMapping),
    (uint2, TStripeMapping),
    (ui16, TStripeMapping));

#undef Y_CATBOOST_CUDA_F_IMPL
#undef Y_CATBOOST_CUDA_F_IMPL_PROXY

// AddVector

template <typename T, typename TMapping>
static void AddVectorImpl(
    TCudaBuffer<std::remove_const_t<T>, TMapping>& x,
    T value,
    ui32 stream) {
    using TKernel = TBinOpKernel<std::remove_const_t<T>>;
    LaunchKernels<TKernel>(x.NonEmptyDevices(), stream, x, value, EBinOpType::AddConst);
}

#define Y_CATBOOST_CUDA_F_IMPL_PROXY(x) \
    Y_CATBOOST_CUDA_F_IMPL x

#define Y_CATBOOST_CUDA_F_IMPL(T, TMapping)                \
    template <>                                            \
    void AddVector<T, TMapping>(                           \
        TCudaBuffer<std::remove_const_t<T>, TMapping> & x, \
        T value,                                           \
        ui32 stream) {                                     \
        ::AddVectorImpl(x, value, stream);                 \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (float, TMirrorMapping),
    (int, TMirrorMapping),
    (ui32, TMirrorMapping),
    (double, TMirrorMapping),
    (ui8, TMirrorMapping),
    (uint2, TMirrorMapping),
    (ui16, TMirrorMapping),
    (float, TSingleMapping),
    (int, TSingleMapping),
    (ui32, TSingleMapping),
    (double, TSingleMapping),
    (ui8, TSingleMapping),
    (uint2, TSingleMapping),
    (ui16, TSingleMapping),
    (float, TStripeMapping),
    (int, TStripeMapping),
    (ui32, TStripeMapping),
    (double, TStripeMapping),
    (ui8, TStripeMapping),
    (uint2, TStripeMapping),
    (ui16, TStripeMapping));

#undef Y_CATBOOST_CUDA_F_IMPL
#undef Y_CATBOOST_CUDA_F_IMPL_PROXY

// SubstractVector

template <typename T, typename TMapping>
static void SubtractVectorImpl(
    TCudaBuffer<std::remove_const_t<T>, TMapping>& x,
    const TCudaBuffer<T, TMapping>& y,
    ui32 stream) {
    using TKernel = TBinOpKernel<std::remove_const_t<T>>;
    LaunchKernels<TKernel>(x.NonEmptyDevices(), stream, x, y, EBinOpType::SubVec);
}

#define Y_CATBOOST_CUDA_F_IMPL_PROXY(x) \
    Y_CATBOOST_CUDA_F_IMPL x

#define Y_CATBOOST_CUDA_F_IMPL(T, TMapping)                \
    template <>                                            \
    void SubtractVector<T, TMapping>(                      \
        TCudaBuffer<std::remove_const_t<T>, TMapping> & x, \
        const TCudaBuffer<T, TMapping>& y,                 \
        ui32 stream) {                                     \
        ::SubtractVectorImpl(x, y, stream);                \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (float, TMirrorMapping),
    (float const, TMirrorMapping),
    (int, TMirrorMapping),
    (ui32, TMirrorMapping),
    (double, TMirrorMapping),
    (ui8, TMirrorMapping),
    (uint2, TMirrorMapping),
    (ui16, TMirrorMapping),
    (float, TSingleMapping),
    (const float, TSingleMapping),
    (int, TSingleMapping),
    (ui32, TSingleMapping),
    (double, TSingleMapping),
    (ui8, TSingleMapping),
    (uint2, TSingleMapping),
    (ui16, TSingleMapping),
    (float, TStripeMapping),
    (const float, TStripeMapping),
    (int, TStripeMapping),
    (ui32, TStripeMapping),
    (double, TStripeMapping),
    (ui8, TStripeMapping),
    (uint2, TStripeMapping),
    (ui16, TStripeMapping));

#undef Y_CATBOOST_CUDA_F_IMPL
#undef Y_CATBOOST_CUDA_F_IMPL_PROXY

// MultiplyVector

template <typename T, typename TMapping>
static void MultiplyVectorImpl(
    TCudaBuffer<std::remove_const_t<T>, TMapping>& x,
    const TCudaBuffer<T, TMapping>& y,
    ui32 stream) {
    using TKernel = TBinOpKernel<std::remove_const_t<T>>;
    LaunchKernels<TKernel>(x.NonEmptyDevices(), stream, x, y, EBinOpType::MulVec);
}

#define Y_CATBOOST_CUDA_F_IMPL_PROXY(x) \
    Y_CATBOOST_CUDA_F_IMPL x

#define Y_CATBOOST_CUDA_F_IMPL(T, TMapping)                \
    template <>                                            \
    void MultiplyVector<T, TMapping>(                      \
        TCudaBuffer<std::remove_const_t<T>, TMapping> & x, \
        const TCudaBuffer<T, TMapping>& y,                 \
        ui32 stream) {                                     \
        ::MultiplyVectorImpl(x, y, stream);                \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (float, TMirrorMapping),
    (const float, TMirrorMapping),
    (int, TMirrorMapping),
    (ui32, TMirrorMapping),
    (double, TMirrorMapping),
    (ui8, TMirrorMapping),
    (uint2, TMirrorMapping),
    (ui16, TMirrorMapping),
    (float, TSingleMapping),
    (const float, TSingleMapping),
    (int, TSingleMapping),
    (ui32, TSingleMapping),
    (double, TSingleMapping),
    (ui8, TSingleMapping),
    (uint2, TSingleMapping),
    (ui16, TSingleMapping),
    (float, TStripeMapping),
    (const float, TStripeMapping),
    (int, TStripeMapping),
    (ui32, TStripeMapping),
    (double, TStripeMapping),
    (ui8, TStripeMapping),
    (uint2, TStripeMapping),
    (ui16, TStripeMapping));

#undef Y_CATBOOST_CUDA_F_IMPL
#undef Y_CATBOOST_CUDA_F_IMPL_PROXY

// MultiplyVector

template <typename T, typename TMapping>
static void MultiplyVectorImpl(
    TCudaBuffer<std::remove_const_t<T>, TMapping>& x,
    T y,
    ui32 stream) {
    using TKernel = TBinOpKernel<std::remove_const_t<T>>;
    LaunchKernels<TKernel>(x.NonEmptyDevices(), stream, x, y, EBinOpType::MulConst);
}

#define Y_CATBOOST_CUDA_F_IMPL_PROXY(x) \
    Y_CATBOOST_CUDA_F_IMPL x

#define Y_CATBOOST_CUDA_F_IMPL(T, TMapping)                \
    template <>                                            \
    void MultiplyVector<T, TMapping>(                      \
        TCudaBuffer<std::remove_const_t<T>, TMapping> & x, \
        T y,                                               \
        ui32 stream) {                                     \
        ::MultiplyVectorImpl(x, y, stream);                \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (float, TMirrorMapping),
    (int, TMirrorMapping),
    (ui32, TMirrorMapping),
    (double, TMirrorMapping),
    (ui8, TMirrorMapping),
    (uint2, TMirrorMapping),
    (ui16, TMirrorMapping),
    (float, TSingleMapping),
    (int, TSingleMapping),
    (ui32, TSingleMapping),
    (double, TSingleMapping),
    (ui8, TSingleMapping),
    (uint2, TSingleMapping),
    (ui16, TSingleMapping),
    (float, TStripeMapping),
    (int, TStripeMapping),
    (ui32, TStripeMapping),
    (double, TStripeMapping),
    (ui8, TStripeMapping),
    (uint2, TStripeMapping),
    (ui16, TStripeMapping));

#undef Y_CATBOOST_CUDA_F_IMPL
#undef Y_CATBOOST_CUDA_F_IMPL_PROXY

// DivideVector

template <typename T, typename TMapping>
static void DivideVectorImpl(
    TCudaBuffer<std::remove_const_t<T>, TMapping>& x,
    const TCudaBuffer<T, TMapping>& y,
    ui32 stream) {
    using TKernel = TBinOpKernel<std::remove_const_t<T>>;
    LaunchKernels<TKernel>(x.NonEmptyDevices(), stream, x, y, EBinOpType::DivVec);
}

#define Y_CATBOOST_CUDA_F_IMPL_PROXY(x) \
    Y_CATBOOST_CUDA_F_IMPL x

#define Y_CATBOOST_CUDA_F_IMPL(T, TMapping)                \
    template <>                                            \
    void DivideVector<T, TMapping>(                        \
        TCudaBuffer<std::remove_const_t<T>, TMapping> & x, \
        const TCudaBuffer<T, TMapping>& y,                 \
        ui32 stream) {                                     \
        ::DivideVectorImpl(x, y, stream);                  \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (float, TMirrorMapping),
    (int, TMirrorMapping),
    (ui32, TMirrorMapping),
    (double, TMirrorMapping),
    (ui8, TMirrorMapping),
    (uint2, TMirrorMapping),
    (ui16, TMirrorMapping),
    (float, TSingleMapping),
    (int, TSingleMapping),
    (ui32, TSingleMapping),
    (double, TSingleMapping),
    (ui8, TSingleMapping),
    (uint2, TSingleMapping),
    (ui16, TSingleMapping),
    (float, TStripeMapping),
    (int, TStripeMapping),
    (ui32, TStripeMapping),
    (double, TStripeMapping),
    (ui8, TStripeMapping),
    (uint2, TStripeMapping),
    (ui16, TStripeMapping));

#undef Y_CATBOOST_CUDA_F_IMPL
#undef Y_CATBOOST_CUDA_F_IMPL_PROXY

// ExpVector

namespace {
    template <typename T>
    class TApplyFuncKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<T> X;
        EFuncType Type;

    public:
        TApplyFuncKernel() = default;

        TApplyFuncKernel(TCudaBufferPtr<T> x, EFuncType type)
            : X(x)
            , Type(type)
        {
        }

        void Run(const TCudaStream& stream) const {
            using namespace NKernel;
            switch (Type) {
                case EFuncType::Exp: {
                    ExpVector<T>(X.Get(), X.Size(), stream.GetStream());
                    break;
                }
                case EFuncType::Identity: {
                    CB_ENSURE(false, "Unimplemented");
                }
            }
        }

        Y_SAVELOAD_DEFINE(X, Type);
    };
}

template <typename T, typename TMapping>
static void ExpVectorImpl(
    TCudaBuffer<T, TMapping>& x,
    ui32 stream) {
    using TKernel = TApplyFuncKernel<T>;
    LaunchKernels<TKernel>(x.NonEmptyDevices(), stream, x, EFuncType::Exp);
}

#define Y_CATBOOST_CUDA_F_IMPL_PROXY(x) \
    Y_CATBOOST_CUDA_F_IMPL x

#define Y_CATBOOST_CUDA_F_IMPL(T, TMapping) \
    template <>                             \
    void ExpVector<T, TMapping>(            \
        TCudaBuffer<T, TMapping> & x,       \
        ui32 stream) {                      \
        ::ExpVectorImpl(x, stream);         \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (float, TMirrorMapping),
    (float, TSingleMapping),
    (float, TStripeMapping));

#undef Y_CATBOOST_CUDA_F_IMPL
#undef Y_CATBOOST_CUDA_F_IMPL_PROXY

// Gather

namespace {
    template <typename T, typename Index = ui32>
    class TMapCopyKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<T> Dest;
        TCudaBufferPtr<const T> Source;
        TCudaBufferPtr<const Index> Map;
        EMapCopyType Type;
        Index Mask;

    public:
        TMapCopyKernel() = default;

        TMapCopyKernel(TCudaBufferPtr<T> dest,
                       TCudaBufferPtr<const T> source,
                       TCudaBufferPtr<const Index> map,
                       EMapCopyType type,
                       Index mask = static_cast<Index>(-1ULL))
            : Dest(dest)
            , Source(source)
            , Map(map)
            , Type(type)
            , Mask(mask)
        {
        }

        Y_SAVELOAD_DEFINE(Source, Dest, Map, Type, Mask);

        void Run(const TCudaStream& stream) const {
            Index allMask = static_cast<Index>(-1ULL);
            CB_ENSURE(Map.Size() < std::numeric_limits<Index>::max());
            using namespace NKernel;
            const int columnCount = static_cast<const int>(Dest.GetColumnCount());
            CB_ENSURE(columnCount == (int)Source.GetColumnCount());

            switch (Type) {
                case EMapCopyType::Gather: {
                    if (Mask != allMask) {
                        CB_ENSURE(columnCount == 1);
                        GatherWithMask<T, Index>(Dest.Get(), Source.Get(), Map.Get(), Map.Size(), Mask, stream.GetStream());
                    } else {
                        Gather<T, Index>(Dest.Get(), Source.Get(), Map.Get(), Map.Size(), columnCount, Dest.AlignedColumnSize(), Source.AlignedColumnSize(), stream.GetStream());
                    }
                    break;
                }
                case EMapCopyType::Scatter: {
                    if (Mask != allMask) {
                        CB_ENSURE(columnCount == 1);
                        ScatterWithMask<T, Index>(Dest.Get(), Source.Get(), Map.Get(), Map.Size(), Mask, stream.GetStream());
                    } else {
                        Scatter<T, Index>(Dest.Get(), Source.Get(), Map.Get(), Map.Size(), columnCount, Dest.AlignedColumnSize(), Source.AlignedColumnSize(), stream.GetStream());
                    }
                    break;
                }
            }
        }
    };
}

template <typename T, typename TMapping, typename U>
static void GatherImpl(
    TCudaBuffer<std::remove_const_t<T>, TMapping>& dst,
    const TCudaBuffer<T, TMapping>& src,
    const TCudaBuffer<U, TMapping>& map,
    ui32 stream) {
    using TKernel = TMapCopyKernel<std::remove_const_t<T>, std::remove_const_t<U> >;
    LaunchKernels<TKernel>(dst.NonEmptyDevices(), stream, dst, src, map, EMapCopyType::Gather);
}

#define Y_CATBOOST_CUDA_F_IMPL_PROXY(x) \
    Y_CATBOOST_CUDA_F_IMPL x

#define Y_CATBOOST_CUDA_F_IMPL(T, TMapping, U)               \
    template <>                                              \
    void Gather<T, TMapping, U>(                             \
        TCudaBuffer<std::remove_const_t<T>, TMapping> & dst, \
        const TCudaBuffer<T, TMapping>& src,                 \
        const TCudaBuffer<U, TMapping>& map,                 \
        ui32 stream) {                                       \
        CB_ENSURE(src.GetObjectsSlice().Size() <= static_cast<U>(-1),      \
            "Source is too large for " << sizeof(U) * 8 << "-bit gather"); \
        ::GatherImpl(dst, src, map, stream);                 \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (float, TMirrorMapping, ui32),
    (float, TMirrorMapping, const ui32),
    (const float, TMirrorMapping, ui32),
    (const float, TMirrorMapping, const ui32),
    (int, TMirrorMapping, ui32),
    (ui8, TMirrorMapping, ui32),
    (ui32, TMirrorMapping, ui32),
    (ui32, TMirrorMapping, const ui32),
    (const ui32, TMirrorMapping, const ui32),
    (const uint2, TMirrorMapping, ui32),
    (bool, TMirrorMapping, ui32))

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (float, TSingleMapping, ui32),
    (float, TSingleMapping, const ui32),
    (const float, TSingleMapping, ui32),
    (const float, TSingleMapping, const ui32),
    (int, TSingleMapping, ui32),
    (ui8, TSingleMapping, ui32),
    (ui32, TSingleMapping, ui32),
    (ui32, TSingleMapping, const ui32),
    (const ui32, TSingleMapping, const ui32),
    (const uint2, TSingleMapping, ui32),
    (bool, TSingleMapping, ui32));

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (float, TStripeMapping, ui32),
    (float, TStripeMapping, ui64),
    (float, TStripeMapping, const ui32),
    (const float, TStripeMapping, ui32),
    (const float, TStripeMapping, const ui32),
    (int, TStripeMapping, ui32),
    (ui8, TStripeMapping, ui32),
    (ui32, TStripeMapping, ui32),
    (ui32, TStripeMapping, const ui32),
    (const ui32, TStripeMapping, const ui32),
    (const uint2, TStripeMapping, ui32),
    (const uint2, TStripeMapping, ui64),
    (uint2, TStripeMapping, ui64),
    (bool, TStripeMapping, ui32));

#undef Y_CATBOOST_CUDA_F_IMPL
#undef Y_CATBOOST_CUDA_F_IMPL_PROXY

// Gather

template <typename T, typename U>
static void GatherImpl(
    TCudaBuffer<std::remove_const_t<T>, TStripeMapping>& dst,
    const TCudaBuffer<T, TMirrorMapping>& src,
    const TCudaBuffer<U, TStripeMapping>& map,
    ui32 stream) {
    using TKernel = TMapCopyKernel<std::remove_const_t<T>, std::remove_const_t<U> >;
    LaunchKernels<TKernel>(dst.NonEmptyDevices(), stream, dst, src, map, EMapCopyType::Gather);
}

#define Y_CATBOOST_CUDA_F_IMPL_PROXY(x) \
    Y_CATBOOST_CUDA_F_IMPL x

#define Y_CATBOOST_CUDA_F_IMPL(T, U)                               \
    template <>                                                    \
    void Gather<T, U>(                                             \
        TCudaBuffer<std::remove_const_t<T>, TStripeMapping> & dst, \
        const TCudaBuffer<T, TMirrorMapping>& src,                 \
        const TCudaBuffer<U, TStripeMapping>& map,                 \
        ui32 stream) {                                             \
        CB_ENSURE(src.GetObjectsSlice().Size() <= static_cast<U>(-1),      \
            "Source is too large for " << sizeof(U) * 8 << "-bit gather"); \
        ::GatherImpl(dst, src, map, stream);                       \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (float, ui32),
    (float, ui64),
    (int, ui32),
    (ui8, ui32),
    (ui32, ui32),
    (ui32, const ui32),
    (uint2, ui32),
    (uint2, ui64),
    (bool, ui32));

#undef Y_CATBOOST_CUDA_F_IMPL
#undef Y_CATBOOST_CUDA_F_IMPL_PROXY

// GatherWithMask

template <typename T, typename TMapping, typename U>
static void GatherWithMaskImpl(
    TCudaBuffer<std::remove_const_t<T>, TMapping>& dst,
    const TCudaBuffer<T, TMapping>& src,
    const TCudaBuffer<U, TMapping>& map,
    ui32 mask,
    ui32 stream) {
    using TKernel = TMapCopyKernel<std::remove_const_t<T>, std::remove_const_t<U> >;
    LaunchKernels<TKernel>(dst.NonEmptyDevices(), stream, dst, src, map, EMapCopyType::Gather, mask);
}

#define Y_CATBOOST_CUDA_F_IMPL_PROXY(x) \
    Y_CATBOOST_CUDA_F_IMPL x

#define Y_CATBOOST_CUDA_F_IMPL(T, TMapping, U)               \
    template <>                                              \
    void GatherWithMask<T, TMapping, U>(                     \
        TCudaBuffer<std::remove_const_t<T>, TMapping> & dst, \
        const TCudaBuffer<T, TMapping>& src,                 \
        const TCudaBuffer<U, TMapping>& map,                 \
        ui32 mask,                                           \
        ui32 stream) {                                       \
        CB_ENSURE(src.GetObjectsSlice().Size() <= static_cast<U>(-1),      \
            "Source is too large for " << sizeof(U) * 8 << "-bit gather"); \
        ::GatherWithMaskImpl(dst, src, map, mask, stream);   \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (float, TMirrorMapping, ui32),
    (float, TMirrorMapping, const ui32),
    (const float, TMirrorMapping, const ui32),
    (int, TMirrorMapping, ui32),
    (ui8, TMirrorMapping, ui32),
    (const ui8, TMirrorMapping, const ui32),
    (ui32, TMirrorMapping, ui32),
    (uint2, TMirrorMapping, ui32),
    (bool, TMirrorMapping, ui32));

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (float, TSingleMapping, ui32),
    (float, TSingleMapping, const ui32),
    (const float, TSingleMapping, const ui32),
    (int, TSingleMapping, ui32),
    (ui8, TSingleMapping, ui32),
    (const ui8, TSingleMapping, const ui32),
    (ui32, TSingleMapping, ui32),
    (uint2, TSingleMapping, ui32),
    (bool, TSingleMapping, ui32));

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (float, TStripeMapping, ui32),
    (const float, TStripeMapping, const ui32),
    (int, TStripeMapping, ui32),
    (ui8, TStripeMapping, ui32),
    (const ui8, TStripeMapping, const ui32),
    (ui32, TStripeMapping, ui32),
    (uint2, TStripeMapping, ui32),
    (bool, TStripeMapping, ui32));

#undef Y_CATBOOST_CUDA_F_IMPL
#undef Y_CATBOOST_CUDA_F_IMPL_PROXY

// Scatter
//
template <typename T, typename TMapping, typename U>
static void ScatterImpl(
    TCudaBuffer<std::remove_const_t<T>, TMapping>& dst,
    const TCudaBuffer<T, TMapping>& src,
    const TCudaBuffer<U, TMapping>& map,
    ui32 stream) {
    using TKernel = TMapCopyKernel<std::remove_const_t<T>, std::remove_const_t<U> >;
    LaunchKernels<TKernel>(dst.NonEmptyDevices(), stream, dst, src, map, EMapCopyType::Scatter);
}

#define Y_CATBOOST_CUDA_F_IMPL_PROXY(x) \
    Y_CATBOOST_CUDA_F_IMPL x

#define Y_CATBOOST_CUDA_F_IMPL(T, TMapping, U)               \
    template <>                                              \
    void Scatter<T, TMapping, U>(                            \
        TCudaBuffer<std::remove_const_t<T>, TMapping> & dst, \
        const TCudaBuffer<T, TMapping>& src,                 \
        const TCudaBuffer<U, TMapping>& map,                 \
        ui32 stream) {                                       \
        CB_ENSURE(src.GetObjectsSlice().Size() <= static_cast<U>(-1),      \
            "Source is too large for " << sizeof(U) * 8 << "-bit scatter"); \
        ::ScatterImpl(dst, src, map, stream);                \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (float, TMirrorMapping, ui32),
    (int, TMirrorMapping, ui32),
    (ui8, TMirrorMapping, ui32),
    (ui32, TMirrorMapping, ui32),
    (uint2, TMirrorMapping, ui32),
    (bool, TMirrorMapping, ui32),
    (float, TSingleMapping, ui32),
    (int, TSingleMapping, ui32),
    (ui8, TSingleMapping, ui32),
    (ui32, TSingleMapping, ui32),
    (uint2, TSingleMapping, ui32),
    (bool, TSingleMapping, ui32),
    (float, TStripeMapping, ui32),
    (int, TStripeMapping, ui32),
    (ui8, TStripeMapping, ui32),
    (ui32, TStripeMapping, ui32),
    (uint2, TStripeMapping, ui32),
    (bool, TStripeMapping, ui32));

#undef Y_CATBOOST_CUDA_F_IMPL
#undef Y_CATBOOST_CUDA_F_IMPL_PROXY

// ScatterWithMask

template <typename T, typename TMapping, typename U>
static void ScatterWithMaskImpl(
    TCudaBuffer<std::remove_const_t<T>, TMapping>& dst,
    const TCudaBuffer<T, TMapping>& src,
    const TCudaBuffer<U, TMapping>& map,
    ui32 mask,
    ui32 stream) {
    using TKernel = TMapCopyKernel<std::remove_const_t<T>, std::remove_const_t<U> >;
    LaunchKernels<TKernel>(dst.NonEmptyDevices(), stream, dst, src, map, EMapCopyType::Scatter, mask);
}

#define Y_CATBOOST_CUDA_F_IMPL_PROXY(x) \
    Y_CATBOOST_CUDA_F_IMPL x

#define Y_CATBOOST_CUDA_F_IMPL(T, TMapping, U)               \
    template <>                                              \
    void ScatterWithMask<T, TMapping, U>(                    \
        TCudaBuffer<std::remove_const_t<T>, TMapping> & dst, \
        const TCudaBuffer<T, TMapping>& src,                 \
        const TCudaBuffer<U, TMapping>& map,                 \
        ui32 mask,                                           \
        ui32 stream) {                                       \
        CB_ENSURE(src.GetObjectsSlice().Size() <= static_cast<U>(-1),      \
            "Source is too large for " << sizeof(U) * 8 << "-bit scatter"); \
        ::ScatterWithMaskImpl(dst, src, map, mask, stream);  \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (float, TMirrorMapping, ui32),
    (float, TMirrorMapping, const ui32),
    (int, TMirrorMapping, ui32),
    (ui8, TMirrorMapping, ui32),
    (ui32, TMirrorMapping, ui32),
    (ui32, TMirrorMapping, const ui32),
    (uint2, TMirrorMapping, ui32),
    (bool, TMirrorMapping, ui32),
    (float, TSingleMapping, ui32),
    (float, TSingleMapping, const ui32),
    (int, TSingleMapping, ui32),
    (ui8, TSingleMapping, ui32),
    (ui32, TSingleMapping, ui32),
    (ui32, TSingleMapping, const ui32),
    (uint2, TSingleMapping, ui32),
    (bool, TSingleMapping, ui32),
    (float, TStripeMapping, ui32),
    (float, TStripeMapping, const ui32),
    (int, TStripeMapping, ui32),
    (ui8, TStripeMapping, ui32),
    (ui32, TStripeMapping, ui32),
    (ui32, TStripeMapping, const ui32),
    (uint2, TStripeMapping, ui32),
    (bool, TStripeMapping, ui32));

#undef Y_CATBOOST_CUDA_F_IMPL
#undef Y_CATBOOST_CUDA_F_IMPL_PROXY

// Reverse

namespace {
    template <typename T>
    class TReverseKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<T> Data;

    public:
        TReverseKernel() = default;

        TReverseKernel(TCudaBufferPtr<T> data)
            : Data(data)
        {
        }

        Y_SAVELOAD_DEFINE(Data);

        void Run(const TCudaStream& stream) const {
            CB_ENSURE(Data.Size() == Data.ObjectCount());
            NKernel::Reverse<T>(Data.Get(), Data.Size(), stream.GetStream());
        }
    };
}

template <typename T, typename TMapping>
static void ReverseImpl(
    TCudaBuffer<T, TMapping>& data,
    ui32 stream) {
    using TKernel = TReverseKernel<T>;
    LaunchKernels<TKernel>(data.NonEmptyDevices(), stream, data);
}

#define Y_CATBOOST_CUDA_F_IMPL_PROXY(x) \
    Y_CATBOOST_CUDA_F_IMPL x

#define Y_CATBOOST_CUDA_F_IMPL(T, TMapping) \
    template <>                             \
    void Reverse<T, TMapping>(              \
        TCudaBuffer<T, TMapping> & data,    \
        ui32 stream) {                      \
        ::ReverseImpl(data, stream);        \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (float, TMirrorMapping));

#undef Y_CATBOOST_CUDA_F_IMPL
#undef Y_CATBOOST_CUDA_F_IMPL_PROXY

// PowVector

namespace {
    template <typename T>
    class TPowKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<T> X;
        T Base = 0;

    public:
        TPowKernel() = default;

        TPowKernel(TCudaBufferPtr<T> x, T base)
            : X(x)
            , Base(base)
        {
        }

        void Run(const TCudaStream& stream) const {
            NKernel::PowVector(X.Get(), X.Size(), Base, stream);
        }

        Y_SAVELOAD_DEFINE(X, Base);
    };
}

template <typename T, typename TMapping>
void PowVectorImpl(TCudaBuffer<T, TMapping>& x, float base, ui32 stream) {
    using TKernel = TPowKernel<T>;
    LaunchKernels<TKernel>(x.NonEmptyDevices(), stream, x, base);
}

#define Y_CATBOOST_CUDA_F_IMPL_PROXY(x) \
    Y_CATBOOST_CUDA_F_IMPL x

#define Y_CATBOOST_CUDA_F_IMPL(T, TMapping)                                              \
    template <>                                                                          \
    void PowVector<T, TMapping>(TCudaBuffer<T, TMapping> & x, float base, ui32 stream) { \
        PowVectorImpl(x, base, stream);                                                  \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (float, TMirrorMapping),
    (float, TSingleMapping),
    (float, TStripeMapping));

#undef Y_CATBOOST_CUDA_F_IMPL
#undef Y_CATBOOST_CUDA_F_IMPL_PROXY

// PowVector

namespace {
    template <typename T>
    class TPowWithOutputKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const T> X;
        TCudaBufferPtr<T> Y;
        T Base = 0;

    public:
        TPowWithOutputKernel() = default;

        TPowWithOutputKernel(TCudaBufferPtr<const T> x, T base, TCudaBufferPtr<T> y)
            : X(x)
            , Y(y)
            , Base(base)
        {
            Y_ASSERT(X.Size() == Y.Size());
        }

        void Run(const TCudaStream& stream) const {
            NKernel::PowVector(X.Get(), X.Size(), Base, Y.Get(), stream);
        }

        Y_SAVELOAD_DEFINE(X, Y, Base);
    };
}

template <typename T, typename U, typename TMapping>
void PowVectorImpl(
    const TCudaBuffer<T, TMapping>& x,
    std::remove_const_t<T> base,
    TCudaBuffer<U, TMapping>& y,
    ui32 stream) {
    using TKernel = TPowWithOutputKernel<std::remove_const_t<T>>;
    LaunchKernels<TKernel>(x.NonEmptyDevices(), stream, x, base, y);
}

#define Y_CATBOOST_CUDA_F_IMPL_PROXY(x) \
    Y_CATBOOST_CUDA_F_IMPL x

#define Y_CATBOOST_CUDA_F_IMPL(T, U, TMapping) \
    template <>                                \
    void PowVector<T, U, TMapping>(            \
        const TCudaBuffer<T, TMapping>& x,     \
        std::remove_const_t<T> base,           \
        TCudaBuffer<U, TMapping>& y,           \
        ui32 stream) {                         \
        PowVectorImpl(x, base, y, stream);     \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (float, float, TMirrorMapping),
    (const float, float, TMirrorMapping),
    (float, float, TSingleMapping),
    (const float, float, TSingleMapping),
    (float, float, TStripeMapping),
    (const float, float, TStripeMapping));

#undef Y_CATBOOST_CUDA_F_IMPL
#undef Y_CATBOOST_CUDA_F_IMPL_PROXY

namespace NCudaLib {
    REGISTER_KERNEL_TEMPLATE(0x110001, TBinOpKernel, float);
    REGISTER_KERNEL_TEMPLATE(0x110002, TBinOpKernel, int);
    REGISTER_KERNEL_TEMPLATE(0x110003, TBinOpKernel, ui32);
    REGISTER_KERNEL_TEMPLATE(0x110004, TBinOpKernel, double);
    REGISTER_KERNEL_TEMPLATE(0x110005, TBinOpKernel, ui8);
    REGISTER_KERNEL_TEMPLATE(0x110006, TBinOpKernel, uint2);
    REGISTER_KERNEL_TEMPLATE(0x110007, TBinOpKernel, ui16);

    REGISTER_KERNEL_TEMPLATE(0x110008, TApplyFuncKernel, float);

    REGISTER_KERNEL_TEMPLATE_2(0x110009, TMapCopyKernel, float, ui32);
    REGISTER_KERNEL_TEMPLATE_2(0x110010, TMapCopyKernel, int, ui32);
    REGISTER_KERNEL_TEMPLATE_2(0x110011, TMapCopyKernel, ui8, ui32);
    REGISTER_KERNEL_TEMPLATE_2(0x110012, TMapCopyKernel, ui32, ui32);
    REGISTER_KERNEL_TEMPLATE_2(0x110013, TMapCopyKernel, uint2, ui32);
    REGISTER_KERNEL_TEMPLATE_2(0x110014, TMapCopyKernel, bool, ui32);
    REGISTER_KERNEL_TEMPLATE_2(0x110035, TMapCopyKernel, float, ui64);
    REGISTER_KERNEL_TEMPLATE_2(0x110036, TMapCopyKernel, uint2, ui64);

    REGISTER_KERNEL_TEMPLATE(0x110015, TPowKernel, float);
    REGISTER_KERNEL_TEMPLATE(0x110021, TPowWithOutputKernel, float);

    REGISTER_KERNEL_TEMPLATE(0x110026, TReverseKernel, float);
    REGISTER_KERNEL_TEMPLATE(0x110027, TReverseKernel, int);
    REGISTER_KERNEL_TEMPLATE(0x110028, TReverseKernel, ui32);
    REGISTER_KERNEL_TEMPLATE(0x11002A, TReverseKernel, ui8);
    REGISTER_KERNEL_TEMPLATE(0x11002C, TReverseKernel, ui16);
}
