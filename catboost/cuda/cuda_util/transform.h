#pragma once

#include <catboost/cuda/cuda_util/kernel/transform.cuh>
#include <catboost/cuda/cuda_lib/cuda_kernel_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/libs/helpers/exception.h>

namespace NKernelHost {
    enum class EBinOpType {
        AddVec,
        AddConst,
        SubVec,
        MulVec,
        MulConst,
        DivVec
    };

    enum class EFuncType {
        Exp,
        Identity
    };

    enum class EMapCopyType {
        Gather,
        Scatter
    };

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
                case EFuncType::Exp:
                    ExpVector<T>(X.Get(), X.Size(), stream.GetStream());
                    break;
                case EFuncType::Identity: {
                    CB_ENSURE(false, "Unimplemented");
                }
            }
        }

        Y_SAVELOAD_DEFINE(X, Type);
    };

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
            switch (Type) {
                case EMapCopyType::Gather: {
                    if (Mask != allMask) {
                        GatherWithMask<T, Index>(Dest.Get(), Source.Get(), Map.Get(), Map.Size(), Mask, stream.GetStream());
                    } else {
                        Gather<T, Index>(Dest.Get(), Source.Get(), Map.Get(), Map.Size(), stream.GetStream());
                    }
                    break;
                }
                case EMapCopyType::Scatter: {
                    if (Mask != allMask) {
                        ScatterWithMask<T, Index>(Dest.Get(), Source.Get(), Map.Get(), Map.Size(), Mask, stream.GetStream());
                    } else {
                        Scatter<T, Index>(Dest.Get(), Source.Get(), Map.Get(), Map.Size(), stream.GetStream());
                    }
                    break;
                }
            }
        }
    };
    //
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

template <typename T, class TMapping>
inline void AddVector(TCudaBuffer<std::remove_const_t<T>, TMapping>& x, const TCudaBuffer<T, TMapping>& y, ui32 stream = 0) {
    using TKernel = NKernelHost::TBinOpKernel<std::remove_const_t<T>>;
    LaunchKernels<TKernel>(x.NonEmptyDevices(), stream, x, y, NKernelHost::EBinOpType::AddVec);
}

template <typename T, class TMapping>
inline void AddVector(TCudaBuffer<std::remove_const_t<T>, TMapping>& x, T value, ui32 stream = 0) {
    using TKernel = NKernelHost::TBinOpKernel<std::remove_const_t<T>>;
    LaunchKernels<TKernel>(x.NonEmptyDevices(), stream, x, value, NKernelHost::EBinOpType::AddConst);
}

template <typename T, class TMapping>
inline void SubtractVector(TCudaBuffer<std::remove_const_t<T>, TMapping>& x, const TCudaBuffer<T, TMapping>& y, ui32 stream = 0) {
    using TKernel = NKernelHost::TBinOpKernel<std::remove_const_t<T>>;
    LaunchKernels<TKernel>(x.NonEmptyDevices(), stream, x, y, NKernelHost::EBinOpType::SubVec);
}

template <typename T, class TMapping>
inline void MultiplyVector(TCudaBuffer<std::remove_const_t<T>, TMapping>& x, const TCudaBuffer<T, TMapping>& y, ui32 stream = 0) {
    using TKernel = NKernelHost::TBinOpKernel<std::remove_const_t<T>>;
    LaunchKernels<TKernel>(x.NonEmptyDevices(), stream, x, y, NKernelHost::EBinOpType::MulVec);
}

template <typename T, class TMapping>
inline void MultiplyVector(TCudaBuffer<std::remove_const_t<T>, TMapping>& x, T y, ui32 stream = 0) {
    using TKernel = NKernelHost::TBinOpKernel<std::remove_const_t<T>>;
    LaunchKernels<TKernel>(x.NonEmptyDevices(), stream, x, y, NKernelHost::EBinOpType::MulConst);
}

template <typename T, class TMapping>
inline void DivideVector(TCudaBuffer<std::remove_const_t<T>, TMapping>& x, const TCudaBuffer<T, TMapping>& y, ui32 stream = 0) {
    using TKernel = NKernelHost::TBinOpKernel<std::remove_const_t<T>>;
    LaunchKernels<TKernel>(x.NonEmptyDevices(), stream, x, y, NKernelHost::EBinOpType::DivVec);
}

template <typename T, class TMapping>
inline void ExpVector(TCudaBuffer<T, TMapping>& x, ui32 stream = 0) {
    using TKernel = NKernelHost::TApplyFuncKernel<T>;
    LaunchKernels<TKernel>(x.NonEmptyDevices(), stream, x, NKernelHost::EFuncType::Exp);
}

template <typename T, class TMapping, class U = const ui32>
inline void Gather(TCudaBuffer<std::remove_const_t<T>, TMapping>& dst, const TCudaBuffer<T, TMapping>& src, const TCudaBuffer<U, TMapping>& map,
                   ui32 stream = 0) {
    using TKernel = NKernelHost::TMapCopyKernel<std::remove_const_t<T>, ui32>;
    LaunchKernels<TKernel>(dst.NonEmptyDevices(), stream, dst, src, map, NKernelHost::EMapCopyType::Gather);
}

template <typename T, class U = const ui32>
inline void Gather(TCudaBuffer<std::remove_const_t<T>, NCudaLib::TStripeMapping>& dst,
                   const TCudaBuffer<T, NCudaLib::TMirrorMapping>& src,
                   const TCudaBuffer<U, NCudaLib::TStripeMapping>& map,
                   ui32 stream = 0) {
    using TKernel = NKernelHost::TMapCopyKernel<std::remove_const_t<T>, ui32>;
    LaunchKernels<TKernel>(dst.NonEmptyDevices(), stream, dst, src, map, NKernelHost::EMapCopyType::Gather);
}

template <typename T, class TMapping, class U = const ui32>
inline void GatherWithMask(TCudaBuffer<std::remove_const_t<T>, TMapping>& dst, const TCudaBuffer<T, TMapping>& src, const TCudaBuffer<U, TMapping>& map,
                           ui32 mask, ui32 stream = 0) {
    using TKernel = NKernelHost::TMapCopyKernel<std::remove_const_t<T>, ui32>;
    LaunchKernels<TKernel>(dst.NonEmptyDevices(), stream, dst, src, map, NKernelHost::EMapCopyType::Gather, mask);
}

template <typename T, class TMapping, class U = const ui32>
inline void Scatter(TCudaBuffer<std::remove_const_t<T>, TMapping>& dst, const TCudaBuffer<T, TMapping>& src, const TCudaBuffer<U, TMapping>& map, ui32 stream = 0) {
    using TKernel = NKernelHost::TMapCopyKernel<std::remove_const_t<T>, ui32>;
    LaunchKernels<TKernel>(dst.NonEmptyDevices(), stream, dst, src, map, NKernelHost::EMapCopyType::Scatter);
}

template <typename T, class TMapping, class U = const ui32>
inline void ScatterWithMask(TCudaBuffer<std::remove_const_t<T>, TMapping>& dst, const TCudaBuffer<T, TMapping>& src, const TCudaBuffer<U, TMapping>& map, ui32 mask, ui32 stream = 0) {
    using TKernel = NKernelHost::TMapCopyKernel<std::remove_const_t<T>, ui32>;
    LaunchKernels<TKernel>(dst.NonEmptyDevices(), stream, dst, src, map, NKernelHost::EMapCopyType::Scatter, mask);
}

//
template <typename T, class TMapping>
inline void Reverse(TCudaBuffer<T, TMapping>& data, ui32 stream = 0) {
    using TKernel = NKernelHost::TReverseKernel<T>;
    LaunchKernels<TKernel>(data.NonEmptyDevices(), stream, data);
}
