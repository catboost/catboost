

LIBRARY()

SRCS(
   GLOBAL bootstrap.cpp
   GLOBAL compression_helpers_gpu.cpp
   GLOBAL dot_product.cpp
   GLOBAL fill.cpp
   GLOBAL filter.cpp
   GLOBAL gpu_random.cpp
   GLOBAL helpers.cpp
   GLOBAL partitions.cpp
   GLOBAL partitions_reduce.cpp
   GLOBAL reduce.cpp
   GLOBAL reorder_bins.cpp
   GLOBAL scan.cpp
   GLOBAL segmented_scan.cpp
   GLOBAL segmented_sort.cpp
   GLOBAL sort.cpp
   GLOBAL transform.cpp
   kernel/bootstrap.cu
   kernel/compression.cu
   kernel/dot_product.cu
   kernel/fill.cu
   kernel/filter.cu
   kernel/operators.cu
   kernel/partitions.cu
   kernel/random.cu
   kernel/reduce.cu
   kernel/reorder_one_bit.cu
   kernel/scan.cu
   kernel/segmented_scan.cu
   kernel/segmented_sort.cu
   kernel/sort/bool_uchar.cu
   kernel/sort/bool_ui32.cu
   kernel/sort/float_uchar.cu
   kernel/sort/float_ui16.cu
   kernel/sort/float_ui32.cu
   kernel/sort/float_uint2.cu
   kernel/sort/ui32_uchar.cu
   kernel/sort/ui32_ui16.cu
   kernel/sort/ui32_ui32.cu
   kernel/sort/ui64_uchar.cu
   kernel/sort/ui64_ui32.cu
   kernel/transform.cu
   kernel/update_part_props.cu
)

PEERDIR(
    catboost/libs/helpers
    catboost/cuda/cuda_lib
    contrib/libs/cub
)

INCLUDE(${ARCADIA_ROOT}/catboost/cuda/cuda_lib/default_nvcc_flags.make.inc)

END()
