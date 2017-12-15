LIBRARY()

NO_WERROR()



SRCS(
   kernel/fill.cu
   kernel/scan.cu
   kernel/transform.cu
   kernel/dot_product.cu
   kernel/sort.cu
   kernel/segmented_sort.cu
   kernel/random.cu
   kernel/bootstrap.cu
   kernel/filter.cu
   kernel/partitions.cu
   kernel/compression.cu
   kernel/segmented_scan.cu
   kernel/reduce.cu
   fill.cpp
   scan.cpp
   bootstrap.cpp
   transform.cpp
   dot_product.cpp
   gpu_random.cpp
   sort.cpp
   segmented_sort.cpp
   reorder_bins.cpp
   partitions.cpp
   filter.cpp

)

PEERDIR(
    catboost/cuda/cuda_lib
    catboost/cuda/utils
    contrib/libs/cub
)


CUDA_NVCC_FLAGS(
    --expt-relaxed-constexpr
    -std=c++11
    -gencode arch=compute_30,code=compute_30  -gencode arch=compute_35,code=sm_35  -gencode arch=compute_50,code=compute_50
    -gencode arch=compute_52,code=sm_52
    -gencode arch=compute_60,code=sm_60
    -gencode arch=compute_60,code=sm_60
    -gencode arch=compute_61,code=compute_61
)


END()
