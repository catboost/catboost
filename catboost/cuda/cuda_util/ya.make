LIBRARY()

NO_WERROR()



SRCS(
   kernel/fill.cu
   kernel/scan.cu
   kernel/transform.cu
   kernel/dot_product.cu
   kernel/sort/float_uchar.cu
   kernel/sort/bool_ui32.cu
   kernel/sort/float_ui16.cu
   kernel/sort/float_ui32.cu
   kernel/sort/ui32_uchar.cu
   kernel/sort/ui32_ui16.cu
   kernel/sort/ui32_ui32.cu
   kernel/sort/ui64_ui32.cu
   kernel/sort/float_uint2.cu
   kernel/segmented_sort.cu
   kernel/random.cu
   kernel/bootstrap.cu
   kernel/filter.cu
   kernel/partitions.cu
   kernel/operators.cu
   kernel/compression.cu
   kernel/segmented_scan.cu
   kernel/reduce.cu
   kernel/update_part_props.cu
   GLOBAL partitions_reduce.cpp
   GLOBAL helpers.cpp
   GLOBAL compression_helpers_gpu.cpp
   GLOBAL fill.cpp
   GLOBAL scan.cpp
   GLOBAL bootstrap.cpp
   GLOBAL transform.cpp
   GLOBAL dot_product.cpp
   GLOBAL gpu_random.cpp
   GLOBAL reduce.cpp
   GLOBAL sort.cpp
   GLOBAL segmented_sort.cpp
   GLOBAL segmented_scan.cpp
   GLOBAL reorder_bins.cpp
   GLOBAL partitions.cpp
   GLOBAL filter.cpp
)

PEERDIR(
    catboost/cuda/cuda_lib
    catboost/cuda/utils
    contrib/libs/cub
)

INCLUDE(${ARCADIA_ROOT}/catboost/cuda/cuda_lib/default_nvcc_flags.make.inc)

END()
