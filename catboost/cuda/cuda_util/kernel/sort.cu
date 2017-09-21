#include "sort.cuh"
#include <contrib/libs/cub/cub/device/device_radix_sort.cuh>
#include <cassert>

namespace NKernel {

    template <typename K, typename V> cudaError_t RadixSort(K* keys, V *values, ui32 size, TRadixSortContext& context, TCudaStream stream)
    {
        cub::DoubleBuffer<K> doubleBufferKeys(keys, context.GetTempKeys<K>());
        cudaError_t error;

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

    template cudaError_t RadixSort(uchar* keys, uchar* values, uint size, TRadixSortContext& context, TCudaStream stream);
    template cudaError_t RadixSort(uchar* keys, ushort* values, uint size,TRadixSortContext& context,  TCudaStream stream);
    template cudaError_t RadixSort(uchar* keys, uint* values, uint size, TRadixSortContext& context,  TCudaStream stream);

    template cudaError_t RadixSort(ushort* keys, uchar* values, uint size, TRadixSortContext& context, TCudaStream stream);
    template cudaError_t RadixSort(ushort* keys, ushort* values, uint size, TRadixSortContext& context, TCudaStream stream);
    template cudaError_t RadixSort(ushort* keys, uint* values, uint size, TRadixSortContext& context, TCudaStream stream);

    template cudaError_t RadixSort(uint* keys, uchar* values, uint size, TRadixSortContext& context, TCudaStream stream);
    template cudaError_t RadixSort(uint* keys, ushort* values, uint size, TRadixSortContext& context, TCudaStream stream);
    template cudaError_t RadixSort(uint* keys, uint* values, uint size, TRadixSortContext& context,  TCudaStream stream);

    template cudaError_t RadixSort(float* keys, uchar* values, uint size, TRadixSortContext& context, TCudaStream stream);
    template cudaError_t RadixSort(float* keys, ushort* values, uint size, TRadixSortContext& context, TCudaStream stream);
    template cudaError_t RadixSort(float* keys, uint* values, uint size, TRadixSortContext& context,  TCudaStream stream);



}
