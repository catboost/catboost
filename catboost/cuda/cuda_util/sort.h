#pragma once

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/kernel.h>
#include <catboost/cuda/cuda_util/kernel/sort.cuh>
#include <catboost/cuda/cuda_util/kernel/transform.cuh>
#include <catboost/libs/helpers/exception.h>

namespace NKernelHost {
    template <typename T>
    struct TValueConversion {
        using TValue = T;
    };

    template <>
    struct TValueConversion<char> {
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
            //don't reorder. memory allocation could move pointers. arcadia cuda support is bad
            //TODO(noxoomo): make temp memory more robust
            auto tmpStorage = manager.Allocate<char>(context.TempStorageSize);

            if (!NeedOnlyTempStorage) {
                auto tempKeys = manager.Allocate<char>(size * sizeof(K));
                if (context.ValueSize) {
                    context.TempValues = manager.Allocate<char>(size * context.ValueSize).Get();
                }
                context.TempKeys = tempKeys.Get();
            }
            context.TempStorage = tmpStorage.Get();
        }

        inline void MakeTempKeysAndValuesPtrs(NKernel::TRadixSortContext& context) const {
            CB_ENSURE(context.UseExternalBufferForTempKeysAndValues);
            CB_ENSURE(TmpKeys.Size() == Keys.Size());
            CB_ENSURE(TmpValues.Size() == Values.Size());
            context.TempKeys = reinterpret_cast<char*>(TmpKeys.Get());
            context.TempValues = reinterpret_cast<char*>(TmpValues.Get());
        }

        THolder<TKernelContext> PrepareContext(IMemoryManager& manager) const {
            CB_ENSURE(Keys.Size() == Keys.ObjectCount());
            CB_ENSURE(Keys.Size() < (static_cast<ui64>(1) << 32));

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
            const ui32 size = Keys.Size();

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

template <typename K, class TMapping>
inline void RadixSort(TCudaBuffer<K, TMapping>& keys, bool compareGreater = false, ui32 stream = 0) {
    using TKernel = NKernelHost::TRadixSortKernel<K, char>;
    LaunchKernels<TKernel>(keys.NonEmptyDevices(), stream, keys, compareGreater);
}

template <typename K, typename V, class TMapping>
inline void RadixSort(TCudaBuffer<K, TMapping>& keys,
                      TCudaBuffer<V, TMapping>& values,
                      bool compareGreater = false, ui32 stream = 0) {
    using TKernel = NKernelHost::TRadixSortKernel<K, V>;
    LaunchKernels<TKernel>(keys.NonEmptyDevices(), stream, keys, values, compareGreater);
}

template <typename K, typename V, class TMapping>
inline void RadixSort(TCudaBuffer<K, TMapping>& keys, TCudaBuffer<V, TMapping>& values,
                      TCudaBuffer<K, TMapping>& tmpKeys, TCudaBuffer<V, TMapping>& tmpValues,
                      ui32 offset = 0,
                      ui32 bits = sizeof(K) * 8,
                      ui64 stream = 0) {
    using TKernel = NKernelHost::TRadixSortKernel<K, V>;
    CB_ENSURE((offset + bits) <= (sizeof(K) * 8));
    LaunchKernels<TKernel>(keys.NonEmptyDevices(), stream, keys, values, false, offset, offset + bits, tmpKeys, tmpValues);
}
