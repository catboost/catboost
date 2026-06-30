#include "sort.cuh"
#include "fill.cuh"
#include <cub/device/device_radix_sort.cuh>
#include <cassert>

namespace NKernel {

    template <typename K, typename V> cudaError_t RadixSort(K* keys, V *values, ui32 size, TRadixSortContext& context, TCudaStream stream)
    {
        cub::DoubleBuffer<K> doubleBufferKeys(keys, context.GetTempKeys<K>());
        cudaError_t error;

        FillBuffer<char>(context.TempStorage, 0, context.TempStorageSize, stream);
        if (context.ValueSize) {
            cub::DoubleBuffer<V> doubleBufferValues(values, context.GetTempValues<V>());

            if (context.Descending) {
                cub::DoubleBuffer<K> inputValues;
                error = cub::DeviceRadixSort::SortPairsDescending(context.TempStorage, context.TempStorageSize,
                                                                  doubleBufferKeys,
                                                                  doubleBufferValues,
                                                                  size,
                                                                  context.FirstBit, context.LastBit,
                                                                  stream);
            } else {
                error = cub::DeviceRadixSort::SortPairs(context.TempStorage, context.TempStorageSize,
                                                        doubleBufferKeys,
                                                        doubleBufferValues,
                                                        size,
                                                        context.FirstBit, context.LastBit,
                                                        stream);
            }
            if (doubleBufferValues.Current() != values) {
                assert(sizeof(V) == context.ValueSize);
                cudaMemcpyAsync(values, doubleBufferValues.Current(), sizeof(V) * size, cudaMemcpyDefault, stream);
            }
        } else {
            if (context.Descending) {
                error = cub::DeviceRadixSort::SortKeysDescending(context.TempStorage, context.TempStorageSize,
                                                                 doubleBufferKeys,
                                                                 size,
                                                                 context.FirstBit, context.LastBit,
                                                                 stream);
            } else {
                error = cub::DeviceRadixSort::SortKeys(context.TempStorage, context.TempStorageSize,
                                                       doubleBufferKeys,
                                                       size,
                                                       context.FirstBit, context.LastBit,
                                                       stream);
            }
        }
        //TODO(noxoomo): error handling
        if (doubleBufferKeys.Current() != keys) {
            cudaMemcpyAsync(keys, doubleBufferKeys.Current(), sizeof(K) * size, cudaMemcpyDefault, stream);
        }
        return error;
    }

    extern template cudaError_t RadixSort(float* keys, uchar* values, ui32 size, TRadixSortContext& context, TCudaStream stream);
    extern template cudaError_t RadixSort(float* keys, ui16* values, ui32 size, TRadixSortContext& context, TCudaStream stream);
    extern template cudaError_t RadixSort(float* keys, ui32* values, ui32 size, TRadixSortContext& context,  TCudaStream stream);
    extern template cudaError_t RadixSort(float* keys, uint2* values, ui32 size, TRadixSortContext& context,  TCudaStream stream);

    extern template cudaError_t RadixSort(ui32* keys, uchar* values, ui32 size, TRadixSortContext& context, TCudaStream stream);
    extern template cudaError_t RadixSort(ui32* keys, ui16* values, ui32 size, TRadixSortContext& context, TCudaStream stream);
    extern template cudaError_t RadixSort(ui32* keys, ui32* values, ui32 size, TRadixSortContext& context,  TCudaStream stream);
    extern template cudaError_t RadixSort(ui64* keys, ui32* values, ui32 size, TRadixSortContext& context,  TCudaStream stream);

    extern template cudaError_t RadixSort(bool* keys, ui32* values, ui32 size, TRadixSortContext& context,  TCudaStream stream);
}
