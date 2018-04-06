LIBRARY()

NO_WERROR()



SRCS(
   kernel/fill.cu
   kernel/scan.cu
   kernel/transform.cu
   kernel/dot_product.cu
   kernel/sort/float_uchar.cu
   kernel/sort/float_ui16.cu
   kernel/sort/float_ui32.cu
   kernel/sort/ui32_uchar.cu
   kernel/sort/ui32_ui16.cu
   kernel/sort/ui32_ui32.cu
   kernel/segmented_sort.cu
   kernel/random.cu
   kernel/bootstrap.cu
   kernel/filter.cu
   kernel/partitions.cu
   kernel/operators.cu
   kernel/compression.cu
   kernel/segmented_scan.cu
   kernel/reduce.cu
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


CUDA_NVCC_FLAGS(
    --expt-relaxed-constexpr

    -gencode arch=compute_30,code=compute_30
    -gencode arch=compute_35,code=sm_35
    -gencode arch=compute_50,code=compute_50
    -gencode arch=compute_52,code=sm_52
    -gencode arch=compute_60,code=sm_60
    -gencode arch=compute_61,code=sm_61
    -gencode arch=compute_61,code=compute_61
)


END()
