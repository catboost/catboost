#include "sort.h"
#include "reorder_bins.h"

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_kernel_buffer.h>
#include <catboost/cuda/cuda_lib/kernel.h>
#include <catboost/cuda/cuda_util/kernel/sort.cuh>
#include <catboost/cuda/cuda_util/kernel/transform.cuh>
#include <catboost/libs/helpers/exception.h>

#include <util/generic/cast.h>
#include <util/stream/labeled.h>

using NCudaLib::TMirrorMapping;
using NCudaLib::TSingleMapping;
using NCudaLib::TStripeMapping;
using NKernelHost::IMemoryManager;
using NKernelHost::TCudaBufferPtr;
using NKernelHost::TCudaStream;
using NKernelHost::TKernelBase;
using NKernelHost::uchar;

namespace {
    template <typename T>
    struct TValueConversion {
        using TValue = T;
    };

    template <>
    struct TValueConversion<char> {
        using TValue = unsigned char;
    };

    template <>
    struct TValueConversion<bool> {
        using TValue = unsigned char;
    };

    template <>
    struct TValueConversion<short> {
        using TValue = unsigned short;
    };

    template <>
    struct TValueConversion<int> {
        using TValue = ui32;
    };

    template <>
    struct TValueConversion<float> {
        using TValue = ui32;
    };

    template <typename K, typename V>
    class TRadixSortKernel: public TKernelBase<NKernel::TRadixSortContext> {
    private:
        TCudaBufferPtr<K> Keys;
        TCudaBufferPtr<V> Values;
        using TValueStorage = typename TValueConversion<V>::TValue;
        bool CompareGreater;
        ui32 FirstBit;
        ui32 LastBit;
        TCudaBufferPtr<K> TmpKeys;
        TCudaBufferPtr<V> TmpValues;

    public:
        using TKernelContext = NKernel::TRadixSortContext;

        TRadixSortKernel() = default;

        TRadixSortKernel(TCudaBufferPtr<K> keys,
                         TCudaBufferPtr<V> values,
                         bool compareGreater,
                         ui32 firstBit = 0,
                         ui32 lastBit = sizeof(K) * 8)
            : Keys(keys)
            , Values(values)
            , CompareGreater(compareGreater)
            , FirstBit(firstBit)
            , LastBit(lastBit)
        {
        }

        TRadixSortKernel(TCudaBufferPtr<K> keys,
                         bool compareGreater,
                         ui32 firstBit = 0,
                         ui32 lastBit = sizeof(K) * 8)
            : Keys(keys)
            , Values(TCudaBufferPtr<V>::Nullptr())
            , CompareGreater(compareGreater)
            , FirstBit(firstBit)
            , LastBit(lastBit)
        {
        }

        TRadixSortKernel(TCudaBufferPtr<K> keys,
                         TCudaBufferPtr<V> values,
                         bool compareGreater,
                         ui32 firstBit,
                         ui32 lastBit,
                         TCudaBufferPtr<K> tmpKeys,
                         TCudaBufferPtr<V> tmpValues)
            : Keys(keys)
            , Values(values)
            , CompareGreater(compareGreater)
            , FirstBit(firstBit)
            , LastBit(lastBit)
            , TmpKeys(tmpKeys)
            , TmpValues(tmpValues)
        {
        }

        template <bool NeedOnlyTempStorage = false>
        static inline void AllocateMemory(IMemoryManager& manager, ui32 size, NKernel::TRadixSortContext& context) {
            if (!NeedOnlyTempStorage) {
                context.TempKeys = manager.Allocate<char>(size * sizeof(K));
                if (context.ValueSize) {
                    context.TempValues = manager.Allocate<char>(size * (ui64)context.ValueSize);
                }
            }
            context.TempStorage = manager.Allocate<char>(context.TempStorageSize);
        }

        inline void MakeTempKeysAndValuesPtrs(NKernel::TRadixSortContext& context) const {
            CB_ENSURE(context.UseExternalBufferForTempKeysAndValues);
            CB_ENSURE(TmpKeys.Size() == Keys.Size(), LabeledOutput(TmpKeys.Size(), Keys.Size()));
            CB_ENSURE(TmpValues.Size() == Values.Size(), LabeledOutput(TmpValues.Size(), Values.Size()));
            context.TempKeys = TmpKeys.GetData().GetRawHandleBasedPtr();
            context.TempValues = TmpValues.GetData().GetRawHandleBasedPtr();
        }

        THolder<TKernelContext> PrepareContext(IMemoryManager& manager) const {
            CB_ENSURE(Keys.Size() == Keys.ObjectCount(), LabeledOutput(Keys.Size(), Keys.ObjectCount()));
            CB_ENSURE(Keys.Size() < (static_cast<ui64>(1) << 32), LabeledOutput(Keys.Size()));

            const ui32 size = Keys.Size();
            const ui32 valueSize = Values.Size() ? sizeof(V) : 0;
            if (valueSize) {
                CB_ENSURE(Values.Size() == Keys.Size());
            }
            auto context = MakeHolder<TKernelContext>(FirstBit, LastBit, valueSize, CompareGreater);
            context->UseExternalBufferForTempKeysAndValues = TmpKeys.Size() > 0;

            if (size) {
                //fill temp storage size by cub
                CUDA_SAFE_CALL(NKernel::RadixSort((K*)nullptr, (TValueStorage*)nullptr, size, *context, 0));
                if (context->UseExternalBufferForTempKeysAndValues) {
                    AllocateMemory<true>(manager, size, *context);
                } else {
                    AllocateMemory<false>(manager, size, *context);
                }
            }
            return context;
        }

        void Run(const TCudaStream& stream, TKernelContext& context) const {
            const ui32 size = SafeIntegerCast<ui32>(Keys.Size());

            if (size == 0) {
                return;
            }
            if (context.UseExternalBufferForTempKeysAndValues) {
                MakeTempKeysAndValuesPtrs(context);
            }
            //we need safecall for cub-based routines
            CUDA_SAFE_CALL(NKernel::RadixSort(Keys.Get(), context.ValueSize ? (TValueStorage*)(Values.Get()) : (TValueStorage*)nullptr, size, context, stream.GetStream()));
        }

        Y_SAVELOAD_DEFINE(Keys, Values, CompareGreater, FirstBit, LastBit, TmpKeys, TmpValues);
    };
}

// RadixSort

template <typename K, typename TMapping>
static void RadixSortImpl(TCudaBuffer<K, TMapping>& keys, bool compareGreater, ui32 stream) {
    using TKernel = TRadixSortKernel<K, char>;
    LaunchKernels<TKernel>(keys.NonEmptyDevices(), stream, keys, compareGreater);
}

#define Y_CATBOOST_CUDA_F_IMPL_PROXY(x) \
    Y_CATBOOST_CUDA_F_IMPL x

#define Y_CATBOOST_CUDA_F_IMPL(K, TMapping)                                                          \
    template <>                                                                                      \
    void RadixSort<K, TMapping>(TCudaBuffer<K, TMapping> & keys, bool compareGreater, ui32 stream) { \
        ::RadixSortImpl(keys, compareGreater, stream);                                               \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (float, TMirrorMapping),
    (ui32, TMirrorMapping),
    (ui64, TMirrorMapping),
    (bool, TMirrorMapping),
    (float, TSingleMapping),
    (ui32, TSingleMapping),
    (ui64, TSingleMapping),
    (bool, TSingleMapping),
    (float, TStripeMapping),
    (ui32, TStripeMapping),
    (ui64, TStripeMapping),
    (bool, TStripeMapping));

#undef Y_CATBOOST_CUDA_F_IMPL
#undef Y_CATBOOST_CUDA_F_IMPL_PROXY

// RadixSort

template <typename K, typename V, typename TMapping>
static void RadixSortImpl(
    TCudaBuffer<K, TMapping>& keys,
    TCudaBuffer<V, TMapping>& values,
    bool compareGreater,
    ui32 stream) {
    using TKernel = TRadixSortKernel<K, V>;
    LaunchKernels<TKernel>(keys.NonEmptyDevices(), stream, keys, values, compareGreater);
}

#define Y_CATBOOST_CUDA_F_IMPL_PROXY(x) \
    Y_CATBOOST_CUDA_F_IMPL x

#define Y_CATBOOST_CUDA_F_IMPL(K, V, TMapping)                 \
    template <>                                                \
    void RadixSort<K, V, TMapping>(                            \
        TCudaBuffer<K, TMapping> & keys,                       \
        TCudaBuffer<V, TMapping> & values,                     \
        bool compareGreater,                                   \
        ui32 stream) {                                         \
        ::RadixSortImpl(keys, values, compareGreater, stream); \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (float, uchar, TMirrorMapping),
    (float, char, TMirrorMapping),
    (float, ui16, TMirrorMapping),
    (float, i16, TMirrorMapping),
    (float, ui32, TMirrorMapping),
    (float, i32, TMirrorMapping),
    (float, float, TMirrorMapping),
    (ui32, uchar, TMirrorMapping),
    (ui32, char, TMirrorMapping),
    (ui32, ui16, TMirrorMapping),
    (ui32, i16, TMirrorMapping),
    (ui32, ui32, TMirrorMapping),
    (ui32, i32, TMirrorMapping),
    (ui32, float, TMirrorMapping),
    (ui64, i32, TMirrorMapping),
    (float, uint2, TMirrorMapping),
    (ui64, ui32, TMirrorMapping),
    (bool, ui32, TMirrorMapping));

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (float, uchar, TSingleMapping),
    (float, char, TSingleMapping),
    (float, ui16, TSingleMapping),
    (float, i16, TSingleMapping),
    (float, ui32, TSingleMapping),
    (float, i32, TSingleMapping),
    (float, float, TSingleMapping),
    (ui32, uchar, TSingleMapping),
    (ui32, char, TSingleMapping),
    (ui32, ui16, TSingleMapping),
    (ui32, i16, TSingleMapping),
    (ui32, ui32, TSingleMapping),
    (ui32, i32, TSingleMapping),
    (ui32, float, TSingleMapping),
    (ui64, i32, TSingleMapping),
    (float, uint2, TSingleMapping),
    (ui64, ui32, TSingleMapping),
    (bool, ui32, TSingleMapping));

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (float, uchar, TStripeMapping),
    (float, char, TStripeMapping),
    (float, ui16, TStripeMapping),
    (float, i16, TStripeMapping),
    (float, ui32, TStripeMapping),
    (float, i32, TStripeMapping),
    (float, float, TStripeMapping),
    (ui32, uchar, TStripeMapping),
    (ui32, char, TStripeMapping),
    (ui32, ui16, TStripeMapping),
    (ui32, i16, TStripeMapping),
    (ui32, ui32, TStripeMapping),
    (ui32, i32, TStripeMapping),
    (ui32, float, TStripeMapping),
    (ui64, i32, TStripeMapping),
    (float, uint2, TStripeMapping),
    (ui64, ui32, TStripeMapping),
    (bool, ui32, TStripeMapping));

#undef Y_CATBOOST_CUDA_F_IMPL
#undef Y_CATBOOST_CUDA_F_IMPL_PROXY

// RadixSort

template <typename K, typename V, typename TMapping>
static void RadixSortImpl(
    TCudaBuffer<K, TMapping>& keys, TCudaBuffer<V, TMapping>& values,
    TCudaBuffer<K, TMapping>& tmpKeys, TCudaBuffer<V, TMapping>& tmpValues,
    ui32 offset,
    ui32 bits,
    ui64 stream) {
    if (bits == 0) {
        return;
    }
    using TKernel = TRadixSortKernel<K, V>;
    CB_ENSURE((offset + bits) <= (sizeof(K) * 8), LabeledOutput(offset + bits, sizeof(K) + 8));
    LaunchKernels<TKernel>(keys.NonEmptyDevices(), stream, keys, values, false, offset, offset + bits, tmpKeys, tmpValues);
}

#define Y_CATBOOST_CUDA_F_IMPL_PROXY(x) \
    Y_CATBOOST_CUDA_F_IMPL x

#define Y_CATBOOST_CUDA_F_IMPL(K, V, TMapping)                                    \
    template <>                                                                   \
    void RadixSort<K, V, TMapping>(                                               \
        TCudaBuffer<K, TMapping> & keys, TCudaBuffer<V, TMapping> & values,       \
        TCudaBuffer<K, TMapping> & tmpKeys, TCudaBuffer<V, TMapping> & tmpValues, \
        ui32 offset,                                                              \
        ui32 bits,                                                                \
        ui64 stream) {                                                            \
        ::RadixSortImpl(keys, values, tmpKeys, tmpValues, offset, bits, stream);  \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (float, uchar, TMirrorMapping),
    (float, char, TMirrorMapping),
    (float, ui16, TMirrorMapping),
    (float, i16, TMirrorMapping),
    (float, ui32, TMirrorMapping),
    (float, i32, TMirrorMapping),
    (float, float, TMirrorMapping),
    (ui32, uchar, TMirrorMapping),
    (ui32, char, TMirrorMapping),
    (ui32, ui16, TMirrorMapping),
    (ui32, i16, TMirrorMapping),
    (ui32, ui32, TMirrorMapping),
    (ui32, i32, TMirrorMapping),
    (ui32, float, TMirrorMapping),
    (ui64, i32, TMirrorMapping),
    (float, uint2, TMirrorMapping),
    (ui64, ui32, TMirrorMapping),
    (bool, ui32, TMirrorMapping));

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (float, uchar, TSingleMapping),
    (float, char, TSingleMapping),
    (float, ui16, TSingleMapping),
    (float, i16, TSingleMapping),
    (float, ui32, TSingleMapping),
    (float, i32, TSingleMapping),
    (float, float, TSingleMapping),
    (ui32, uchar, TSingleMapping),
    (ui32, char, TSingleMapping),
    (ui32, ui16, TSingleMapping),
    (ui32, i16, TSingleMapping),
    (ui32, ui32, TSingleMapping),
    (ui32, i32, TSingleMapping),
    (ui32, float, TSingleMapping),
    (ui64, i32, TSingleMapping),
    (float, uint2, TSingleMapping),
    (ui64, ui32, TSingleMapping),
    (bool, ui32, TSingleMapping));

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (float, uchar, TStripeMapping),
    (float, char, TStripeMapping),
    (float, ui16, TStripeMapping),
    (float, i16, TStripeMapping),
    (float, ui32, TStripeMapping),
    (float, i32, TStripeMapping),
    (float, float, TStripeMapping),
    (ui32, uchar, TStripeMapping),
    (ui32, char, TStripeMapping),
    (ui32, ui16, TStripeMapping),
    (ui32, i16, TStripeMapping),
    (ui32, ui32, TStripeMapping),
    (ui32, i32, TStripeMapping),
    (ui32, float, TStripeMapping),
    (ui64, i32, TStripeMapping),
    (float, uint2, TStripeMapping),
    (ui64, ui32, TStripeMapping),
    (bool, ui32, TStripeMapping));

#undef Y_CATBOOST_CUDA_F_IMPL
#undef Y_CATBOOST_CUDA_F_IMPL_PROXY

// RadixSort

template <typename K, typename V, typename TMapping>
static void RadixSortImpl(
    TCudaBuffer<K, TMapping>& keys,
    TCudaBuffer<V, TMapping>& values,
    bool compareGreater,
    ui32 offset,
    ui32 bits,
    ui32 stream) {
    if (offset == bits) {
        return;
    }
    using TKernel = TRadixSortKernel<K, V>;
    LaunchKernels<TKernel>(keys.NonEmptyDevices(), stream, keys, values, compareGreater, offset, bits);
}

#define Y_CATBOOST_CUDA_F_IMPL_PROXY(x) \
    Y_CATBOOST_CUDA_F_IMPL x

#define Y_CATBOOST_CUDA_F_IMPL(K, V, TMapping)                               \
    template <>                                                              \
    void RadixSort<K, V, TMapping>(                                          \
        TCudaBuffer<K, TMapping> & keys,                                     \
        TCudaBuffer<V, TMapping> & values,                                   \
        bool compareGreater,                                                 \
        ui32 offset,                                                         \
        ui32 bits,                                                           \
        ui32 stream) {                                                       \
        ::RadixSortImpl(keys, values, compareGreater, offset, bits, stream); \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (float, uchar, TMirrorMapping),
    (float, char, TMirrorMapping),
    (float, ui16, TMirrorMapping),
    (float, i16, TMirrorMapping),
    (float, ui32, TMirrorMapping),
    (float, i32, TMirrorMapping),
    (float, float, TMirrorMapping),
    (ui32, uchar, TMirrorMapping),
    (ui32, char, TMirrorMapping),
    (ui32, ui16, TMirrorMapping),
    (ui32, i16, TMirrorMapping),
    (ui32, ui32, TMirrorMapping),
    (ui32, i32, TMirrorMapping),
    (ui32, float, TMirrorMapping),
    (ui64, i32, TMirrorMapping),
    (float, uint2, TMirrorMapping),
    (ui64, ui32, TMirrorMapping),
    (bool, ui32, TMirrorMapping));

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (float, uchar, TSingleMapping),
    (float, char, TSingleMapping),
    (float, ui16, TSingleMapping),
    (float, i16, TSingleMapping),
    (float, ui32, TSingleMapping),
    (float, i32, TSingleMapping),
    (float, float, TSingleMapping),
    (ui32, uchar, TSingleMapping),
    (ui32, char, TSingleMapping),
    (ui32, ui16, TSingleMapping),
    (ui32, i16, TSingleMapping),
    (ui32, ui32, TSingleMapping),
    (ui32, i32, TSingleMapping),
    (ui32, float, TSingleMapping),
    (ui64, i32, TSingleMapping),
    (float, uint2, TSingleMapping),
    (ui64, ui32, TSingleMapping),
    (bool, ui32, TSingleMapping));

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (float, uchar, TStripeMapping),
    (float, char, TStripeMapping),
    (float, ui16, TStripeMapping),
    (float, i16, TStripeMapping),
    (float, ui32, TStripeMapping),
    (float, i32, TStripeMapping),
    (float, float, TStripeMapping),
    (ui32, uchar, TStripeMapping),
    (ui32, char, TStripeMapping),
    (ui32, ui16, TStripeMapping),
    (ui32, i16, TStripeMapping),
    (ui32, ui32, TStripeMapping),
    (ui32, i32, TStripeMapping),
    (ui32, float, TStripeMapping),
    (ui32, ui64, TStripeMapping),
    (ui64, i32, TStripeMapping),
    (float, uint2, TStripeMapping),
    (ui64, ui32, TStripeMapping),
    (ui64, ui64, TStripeMapping),
    (bool, ui32, TStripeMapping));

#undef Y_CATBOOST_CUDA_F_IMPL
#undef Y_CATBOOST_CUDA_F_IMPL_PROXY

// ReorderBins

template <typename TMapping, typename TIndex>
static void ReorderBinsImpl(
    TCudaBuffer<ui32, TMapping>& bins,
    TCudaBuffer<TIndex, TMapping>& indices,
    ui32 offset,
    ui32 bits,
    ui64 stream) {
    if (bits == 0) {
        return;
    }
    using TKernel = TRadixSortKernel<ui32, TIndex>;
    CB_ENSURE((offset + bits) <= (sizeof(ui32) * 8), LabeledOutput(offset + bits, sizeof(ui32) * 8));
    LaunchKernels<TKernel>(bins.NonEmptyDevices(), stream, bins, indices, false, offset, offset + bits);
}

#define Y_CATBOOST_CUDA_F_IMPL_PROXY(x) \
    Y_CATBOOST_CUDA_F_IMPL x

#define Y_CATBOOST_CUDA_F_IMPL(TMapping, TIndex)                \
    template <>                                                 \
    void ReorderBins<TMapping, TIndex>(                         \
        TCudaBuffer<ui32, TMapping> & bins,                     \
        TCudaBuffer<TIndex, TMapping> & indices,                \
        ui32 offset,                                            \
        ui32 bits,                                              \
        ui64 stream) {                                          \
        ::ReorderBinsImpl(bins, indices, offset, bits, stream); \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (TMirrorMapping, ui32),
    (TSingleMapping, ui32),
    (TStripeMapping, ui32),
    (TMirrorMapping, ui64),
    (TSingleMapping, ui64),
    (TStripeMapping, ui64));

#undef Y_CATBOOST_CUDA_F_IMPL
#undef Y_CATBOOST_CUDA_F_IMPL_PROXY

// ReorderBins

template <typename TMapping>
static void ReorderBinsImpl(
    TCudaBuffer<ui32, TMapping>& bins,
    TCudaBuffer<ui32, TMapping>& indices,
    ui32 offset,
    ui32 bits,
    TCudaBuffer<ui32, TMapping>& tmpBins,
    TCudaBuffer<ui32, TMapping>& tmpIndices,
    ui64 stream) {
    if (bits == 0) {
        return;
    }
    using TKernel = TRadixSortKernel<ui32, ui32>;
    CB_ENSURE((offset + bits) <= (sizeof(ui32) * 8), LabeledOutput(offset + bits, sizeof(ui32) * 8));
    LaunchKernels<TKernel>(bins.NonEmptyDevices(), stream, bins, indices, false, offset, offset + bits, tmpBins, tmpIndices);
}

#define Y_CATBOOST_CUDA_F_IMPL(TMapping)                                             \
    template <>                                                                      \
    void ReorderBins<TMapping>(                                                      \
        TCudaBuffer<ui32, TMapping> & bins,                                          \
        TCudaBuffer<ui32, TMapping> & indices,                                       \
        ui32 offset,                                                                 \
        ui32 bits,                                                                   \
        TCudaBuffer<ui32, TMapping> & tmpBins,                                       \
        TCudaBuffer<ui32, TMapping> & tmpIndices,                                    \
        ui64 stream) {                                                               \
        ::ReorderBinsImpl(bins, indices, offset, bits, tmpBins, tmpIndices, stream); \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL,
    TMirrorMapping,
    TSingleMapping,
    TStripeMapping);

#undef Y_CATBOOST_CUDA_F_IMPL

// register kernels

namespace NCudaLib {
    //TODO(noxoomo): remap on master side
    REGISTER_KERNEL_TEMPLATE_2(0xAA0001, TRadixSortKernel, float, uchar);
    REGISTER_KERNEL_TEMPLATE_2(0xAA0002, TRadixSortKernel, float, char);
    REGISTER_KERNEL_TEMPLATE_2(0xAA0003, TRadixSortKernel, float, ui16);
    REGISTER_KERNEL_TEMPLATE_2(0xAA0004, TRadixSortKernel, float, i16);
    REGISTER_KERNEL_TEMPLATE_2(0xAA0005, TRadixSortKernel, float, ui32);
    REGISTER_KERNEL_TEMPLATE_2(0xAA0006, TRadixSortKernel, float, i32);
    REGISTER_KERNEL_TEMPLATE_2(0xAA0007, TRadixSortKernel, float, float);

    REGISTER_KERNEL_TEMPLATE_2(0xAA0008, TRadixSortKernel, ui32, uchar);
    REGISTER_KERNEL_TEMPLATE_2(0xAA0009, TRadixSortKernel, ui32, char);
    REGISTER_KERNEL_TEMPLATE_2(0xAA0010, TRadixSortKernel, ui32, ui16);
    REGISTER_KERNEL_TEMPLATE_2(0xAA0011, TRadixSortKernel, ui32, i16);
    REGISTER_KERNEL_TEMPLATE_2(0xAA0012, TRadixSortKernel, ui32, ui32);
    REGISTER_KERNEL_TEMPLATE_2(0xAA0013, TRadixSortKernel, ui32, i32);
    REGISTER_KERNEL_TEMPLATE_2(0xAA0014, TRadixSortKernel, ui32, float);
    REGISTER_KERNEL_TEMPLATE_2(0xAA0015, TRadixSortKernel, ui64, i32);
    REGISTER_KERNEL_TEMPLATE_2(0xAA0016, TRadixSortKernel, float, uint2);
    REGISTER_KERNEL_TEMPLATE_2(0xAA0017, TRadixSortKernel, ui64, ui32);
    REGISTER_KERNEL_TEMPLATE_2(0xAA0018, TRadixSortKernel, bool, ui32);
    REGISTER_KERNEL_TEMPLATE_2(0xAA0019, TRadixSortKernel, ui32, ui64);
    REGISTER_KERNEL_TEMPLATE_2(0xAA0020, TRadixSortKernel, ui64, ui64);

    //    REGISTER_KERNEL_TEMPLATE_2(0xAA0015, TRadixSortKernel, i32, uchar);
    //    REGISTER_KERNEL_TEMPLATE_2(0xAA0016, TRadixSortKernel, i32, char);
    //    REGISTER_KERNEL_TEMPLATE_2(0xAA0017, TRadixSortKernel, i32, ui16);
    //    REGISTER_KERNEL_TEMPLATE_2(0xAA0018, TRadixSortKernel, i32, i16);
    //    REGISTER_KERNEL_TEMPLATE_2(0xAA0019, TRadixSortKernel, i32, ui32);
    //    REGISTER_KERNEL_TEMPLATE_2(0xAA0020, TRadixSortKernel, i32, i32);
    //    REGISTER_KERNEL_TEMPLATE_2(0xAA0021, TRadixSortKernel, i32, float);
}
