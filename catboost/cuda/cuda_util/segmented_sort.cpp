#include "segmented_sort.h"

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/kernel.h>
#include <catboost/cuda/cuda_util/kernel/segmented_sort.cuh>
#include <catboost/cuda/cuda_util/kernel/transform.cuh>
#include <catboost/libs/helpers/exception.h>

using NCudaLib::EPtrType;
using NCudaLib::TMirrorMapping;
using NCudaLib::TSingleMapping;
using NCudaLib::TStripeMapping;
using NKernelHost::IMemoryManager;
using NKernelHost::TCudaBufferPtr;
using NKernelHost::TCudaStream;
using NKernelHost::TKernelBase;

// SegmentedRadixSort

namespace {
    template <typename K, typename V>
    class TSegmentedRadixSortKernel: public TKernelBase<NKernel::TSegmentedRadixSortContext> {
    private:
        TCudaBufferPtr<K> Keys;
        TCudaBufferPtr<V> Values;
        TCudaBufferPtr<K> TmpKeys;
        TCudaBufferPtr<V> TmpValues;
        TCudaBufferPtr<const ui32> SegmentStarts;
        TCudaBufferPtr<const ui32> SegmentEnds;
        ui32 PartCount;
        bool CompareGreater;
        ui32 FirstBit;
        ui32 LastBit;

    public:
        using TKernelContext = NKernel::TSegmentedRadixSortContext;

        TSegmentedRadixSortKernel() = default;

        TSegmentedRadixSortKernel(TCudaBufferPtr<K> keys,
                                  TCudaBufferPtr<V> values,
                                  TCudaBufferPtr<K> tmpKeys,
                                  TCudaBufferPtr<V> tmpValues,
                                  TCudaBufferPtr<const ui32> offsets,
                                  ui32 partCount,
                                  bool compareGreater,
                                  ui32 firstBit,
                                  ui32 lastBit)
            : Keys(keys)
            , Values(values)
            , TmpKeys(tmpKeys)
            , TmpValues(tmpValues)
            , SegmentStarts(offsets)
            , PartCount(partCount)
            , CompareGreater(compareGreater)
            , FirstBit(firstBit)
            , LastBit(lastBit)
        {
        }

        TSegmentedRadixSortKernel(TCudaBufferPtr<K> keys,
                                  TCudaBufferPtr<V> values,
                                  TCudaBufferPtr<K> tmpKeys,
                                  TCudaBufferPtr<V> tmpValues,
                                  TCudaBufferPtr<const ui32> offsets,
                                  TCudaBufferPtr<const ui32> ends,
                                  ui32 partCount,
                                  bool compareGreater,
                                  ui32 firstBit,
                                  ui32 lastBit)
            : Keys(keys)
            , Values(values)
            , TmpKeys(tmpKeys)
            , TmpValues(tmpValues)
            , SegmentStarts(offsets)
            , SegmentEnds(ends)
            , PartCount(partCount)
            , CompareGreater(compareGreater)
            , FirstBit(firstBit)
            , LastBit(lastBit)
        {
        }

        TSegmentedRadixSortKernel(TCudaBufferPtr<K> keys,
                                  TCudaBufferPtr<K> tmpKeys,
                                  TCudaBufferPtr<const ui32> offsets,
                                  ui32 partCount,
                                  bool compareGreater,
                                  ui32 firstBit,
                                  ui32 lastBit)
            : Keys(keys)
            , TmpKeys(tmpKeys)
            , Values(TCudaBufferPtr<V>::Nullptr())
            , SegmentStarts(offsets)
            , PartCount(partCount)
            , CompareGreater(compareGreater)
            , FirstBit(firstBit)
            , LastBit(lastBit)
        {
        }

        THolder<TKernelContext> PrepareContext(IMemoryManager& manager) const {
            CB_ENSURE(Keys.Size() == Keys.ObjectCount());
            CB_ENSURE(Keys.Size() < (static_cast<ui64>(1) << 32));

            const ui32 size = static_cast<const ui32>(Keys.Size());
            auto context = MakeHolder<TKernelContext>(FirstBit, LastBit, CompareGreater);
            if (size) {
                //fill temp storage size by cub
                CUDA_SAFE_CALL(NKernel::SegmentedRadixSort((K*)nullptr, (V*)nullptr, (K*)nullptr, (V*)nullptr,
                                                           size, nullptr, nullptr, PartCount, *context, 0));
                context->TempStorage = manager.Allocate<char>(context->TempStorageSize);
            }
            return context;
        }

        void Run(const TCudaStream& stream, TKernelContext& context) const {
            const ui32 size = Keys.Size();

            if (size == 0) {
                return;
            }
            //we need safecall for cub-based routines
            CUDA_SAFE_CALL(NKernel::SegmentedRadixSort(Keys.Get(), Values.Get(), TmpKeys.Get(), TmpValues.Get(), size,
                                                       SegmentStarts.Get(),
                                                       SegmentEnds.Get() != nullptr ? SegmentEnds.Get() : SegmentStarts.Get() + 1,
                                                       PartCount,
                                                       context, stream.GetStream()));
        }

        Y_SAVELOAD_DEFINE(Keys, Values, TmpKeys, TmpValues, SegmentStarts, PartCount, CompareGreater, FirstBit, LastBit, SegmentEnds);
    };
}

template <typename K, typename V, typename TMapping>
static void SegmentedRadixSortImpl(
    TCudaBuffer<K, TMapping>& keys, TCudaBuffer<V, TMapping>& values,
    TCudaBuffer<K, TMapping>& tmpKeys, TCudaBuffer<V, TMapping>& tmpValues,
    const TCudaBuffer<ui32, TMapping>& offsets, ui32 partCount,
    ui32 fistBit, ui32 lastBit,
    bool compareGreater, ui64 stream) {
    using TKernel = TSegmentedRadixSortKernel<K, V>;
    LaunchKernels<TKernel>(keys.NonEmptyDevices(), stream, keys, values, tmpKeys, tmpValues, offsets, partCount, compareGreater, fistBit, lastBit);
}

#define Y_CATBOOST_CUDA_F_IMPL_PROXY(x) \
    Y_CATBOOST_CUDA_F_IMPL x

#define Y_CATBOOST_CUDA_F_IMPL(K, V, TMapping)                                                                                    \
    template <>                                                                                                                   \
    void SegmentedRadixSort<K, V, TMapping>(                                                                                      \
        TCudaBuffer<K, TMapping> & keys, TCudaBuffer<V, TMapping> & values,                                                       \
        TCudaBuffer<K, TMapping> & tmpKeys, TCudaBuffer<V, TMapping> & tmpValues,                                                 \
        const TCudaBuffer<ui32, TMapping>& offsets, ui32 partCount,                                                               \
        ui32 fistBit, ui32 lastBit,                                                                                               \
        bool compareGreater, ui64 stream) {                                                                                       \
        ::SegmentedRadixSortImpl(keys, values, tmpKeys, tmpValues, offsets, partCount, fistBit, lastBit, compareGreater, stream); \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (ui32, ui32, TMirrorMapping),
    (ui32, ui32, TSingleMapping),
    (ui32, ui32, TStripeMapping),
    (float, float, TSingleMapping));

#undef Y_CATBOOST_CUDA_F_IMPL
#undef Y_CATBOOST_CUDA_F_IMPL_PROXY

// SegmentedRadixSort
template <typename K, typename V, typename TMapping>
static void SegmentedRadixSortImpl(
    TCudaBuffer<K, TMapping>& keys, TCudaBuffer<V, TMapping>& values,
    TCudaBuffer<K, TMapping>& tmpKeys, TCudaBuffer<V, TMapping>& tmpValues,
    const TCudaBuffer<ui32, TMapping>& segmentStarts,
    const TCudaBuffer<ui32, TMapping>& segmentEnds,
    ui32 partCount,
    ui32 fistBit, ui32 lastBit,
    bool compareGreater, ui64 stream) {
    using TKernel = TSegmentedRadixSortKernel<K, V>;
    LaunchKernels<TKernel>(keys.NonEmptyDevices(), stream, keys, values, tmpKeys, tmpValues, segmentStarts, segmentEnds, partCount, compareGreater, fistBit, lastBit);
}

#define Y_CATBOOST_CUDA_F_IMPL_PROXY(x) \
    Y_CATBOOST_CUDA_F_IMPL x

#define Y_CATBOOST_CUDA_F_IMPL(K, V, TMapping)                                                                                                       \
    template <>                                                                                                                                      \
    void SegmentedRadixSort<K, V, TMapping>(                                                                                                         \
        TCudaBuffer<K, TMapping> & keys, TCudaBuffer<V, TMapping> & values,                                                                          \
        TCudaBuffer<K, TMapping> & tmpKeys, TCudaBuffer<V, TMapping> & tmpValues,                                                                    \
        const TCudaBuffer<ui32, TMapping>& segmentStarts,                                                                                            \
        const TCudaBuffer<ui32, TMapping>& segmentEnds,                                                                                              \
        ui32 partCount,                                                                                                                              \
        ui32 fistBit, ui32 lastBit,                                                                                                                  \
        bool compareGreater, ui64 stream) {                                                                                                          \
        ::SegmentedRadixSortImpl(keys, values, tmpKeys, tmpValues, segmentStarts, segmentEnds, partCount, fistBit, lastBit, compareGreater, stream); \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (ui32, ui32, TMirrorMapping),
    (ui32, ui32, TSingleMapping),
    (ui32, ui32, TStripeMapping),
    (float, float, TSingleMapping));

#undef Y_CATBOOST_CUDA_F_IMPL
#undef Y_CATBOOST_CUDA_F_IMPL_PROXY

namespace NCudaLib {
    REGISTER_KERNEL_TEMPLATE_2(0xAB0001, TSegmentedRadixSortKernel, ui32, ui32);
}
