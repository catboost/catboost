#include "segmented_sort.cuh"

#include <cub/device/device_segmented_radix_sort.cuh>

namespace NKernel {

    template <typename K, typename V>
    cudaError_t SegmentedRadixSort(K* keys, V* values,
                                   K* tmpKeys, V* tmpValues,
                                   int size,
                                   const ui32* segmentStarts, const ui32* segmentEnds,
                                   int numSegments,
                                   TSegmentedRadixSortContext& context, TCudaStream stream)
    {
        cub::DoubleBuffer<K> doubleBufferKeys(keys, tmpKeys);
        cudaError_t error;
        int* starts = const_cast<int*>((const int*)(segmentStarts));
        int* ends = const_cast<int*>((const int*)(segmentEnds));

        if (values) {
            cub::DoubleBuffer<V> doubleBufferValues(values, tmpValues);

            if (context.Descending) {


                cub::DoubleBuffer<K> inputValues;
                error = cub::DeviceSegmentedRadixSort::SortPairsDescending(context.TempStorage, context.TempStorageSize,
                                                                           doubleBufferKeys,
                                                                           doubleBufferValues,
                                                                           size,
                                                                           numSegments,
                                                                           starts, ends,
                                                                           context.FirstBit, context.LastBit,
                                                                           stream);
            } else {
                error = cub::DeviceSegmentedRadixSort::SortPairs(context.TempStorage, context.TempStorageSize,
                                                                 doubleBufferKeys,
                                                                 doubleBufferValues,
                                                                 size,
                                                                 numSegments,
                                                                 starts, ends,
                                                                 context.FirstBit, context.LastBit,
                                                                 stream);
            }

            if (doubleBufferValues.Current() != values) {
                cudaMemcpyAsync(values, doubleBufferValues.Current(), sizeof(V) * size, cudaMemcpyDefault, stream);
            }
        } else {
            if (context.Descending) {
                error = cub::DeviceSegmentedRadixSort::SortKeysDescending(context.TempStorage, context.TempStorageSize,
                                                                          doubleBufferKeys,
                                                                          size,
                                                                          numSegments,
                                                                          starts, ends,
                                                                          context.FirstBit, context.LastBit,
                                                                          stream);
                } else {
                    error = cub::DeviceSegmentedRadixSort::SortKeys(context.TempStorage, context.TempStorageSize,
                                                                    doubleBufferKeys,
                                                                    size,
                                                                    numSegments,
                                                                    starts, ends,
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


    #define SEGMENTED_RADIX_SORT(Type) \
    template cudaError_t SegmentedRadixSort(Type* keys, Type* values, Type* tmpKeys, Type* tmpValues, int size, \
                                            const ui32* segmentStarts, const ui32* segmentEnds, int segmentsCount, \
                                            TSegmentedRadixSortContext& context, TCudaStream stream);

    SEGMENTED_RADIX_SORT(float)
    SEGMENTED_RADIX_SORT(ui32)

}
