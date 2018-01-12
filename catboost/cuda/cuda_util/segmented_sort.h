#pragma once

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/kernel.h>
#include <catboost/cuda/cuda_util/kernel/segmented_sort.cuh>
#include <catboost/cuda/cuda_util/kernel/transform.cuh>
#include <catboost/libs/helpers/exception.h>

namespace NKernelHost {
    template <typename K, typename V>
    class TSegmentedRadixSortKernel: public TKernelBase<NKernel::TSegmentedRadixSortContext> {
    private:
        TCudaBufferPtr<K> Keys;
        TCudaBufferPtr<V> Values;
        TCudaBufferPtr<K> TmpKeys;
        TCudaBufferPtr<V> TmpValues;
        TCudaBufferPtr<const ui32> Offsets;
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
            , Offsets(offsets)
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
            , Offsets(offsets)
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
                context->TempStorage = manager.Allocate<char>(context->TempStorageSize).Get();
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
                                                       Offsets.Get(), Offsets.Get() + 1, PartCount,
                                                       context, stream.GetStream()));
        }

        Y_SAVELOAD_DEFINE(Keys, Values, TmpKeys, TmpValues, Offsets, PartCount, CompareGreater, FirstBit, LastBit);
    };
}

template <typename K, typename V, class TMapping>
inline void SegmentedRadixSort(TCudaBuffer<K, TMapping>& keys, TCudaBuffer<V, TMapping>& values,
                               TCudaBuffer<K, TMapping>& tmpKeys, TCudaBuffer<V, TMapping>& tmpValues,
                               const TCudaBuffer<ui32, TMapping>& offsets, ui32 partCount,
                               ui32 fistBit = 0, ui32 lastBit = sizeof(K) * 8,
                               bool compareGreater = false, ui64 stream = 0) {
    using TKernel = NKernelHost::TSegmentedRadixSortKernel<K, V>;
    LaunchKernels<TKernel>(keys.NonEmptyDevices(), stream, keys, values, tmpKeys, tmpValues, offsets, partCount, compareGreater, fistBit, lastBit);
}
