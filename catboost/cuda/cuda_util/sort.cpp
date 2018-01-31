#include <catboost/cuda/cuda_util/sort.h>

using namespace NKernelHost;

namespace NCudaLib {
    //TODO(noxoomo): remap on master side
    REGISTER_KERNEL_TEMPLATE_2(0xAA0001, TRadixSortKernel, float, uchar);
    REGISTER_KERNEL_TEMPLATE_2(0xAA0002, TRadixSortKernel, float, char);
    REGISTER_KERNEL_TEMPLATE_2(0xAA0003, TRadixSortKernel, float, ui16);
    REGISTER_KERNEL_TEMPLATE_2(0xAA0004, TRadixSortKernel, float, i16);
    REGISTER_KERNEL_TEMPLATE_2(0xAA0005, TRadixSortKernel, float, ui32);
    REGISTER_KERNEL_TEMPLATE_2(0xAA0006, TRadixSortKernel, float, i32);
    REGISTER_KERNEL_TEMPLATE_2(0xAA0007, TRadixSortKernel, float, float);

    REGISTER_KERNEL_TEMPLATE_2(0xAA0008, TRadixSortKernel, ui32, uchar);
    REGISTER_KERNEL_TEMPLATE_2(0xAA0009, TRadixSortKernel, ui32, char);
    REGISTER_KERNEL_TEMPLATE_2(0xAA0010, TRadixSortKernel, ui32, ui16);
    REGISTER_KERNEL_TEMPLATE_2(0xAA0011, TRadixSortKernel, ui32, i16);
    REGISTER_KERNEL_TEMPLATE_2(0xAA0012, TRadixSortKernel, ui32, ui32);
    REGISTER_KERNEL_TEMPLATE_2(0xAA0013, TRadixSortKernel, ui32, i32);
    REGISTER_KERNEL_TEMPLATE_2(0xAA0014, TRadixSortKernel, ui32, float);

    //    REGISTER_KERNEL_TEMPLATE_2(0xAA0015, TRadixSortKernel, i32, uchar);
    //    REGISTER_KERNEL_TEMPLATE_2(0xAA0016, TRadixSortKernel, i32, char);
    //    REGISTER_KERNEL_TEMPLATE_2(0xAA0017, TRadixSortKernel, i32, ui16);
    //    REGISTER_KERNEL_TEMPLATE_2(0xAA0018, TRadixSortKernel, i32, i16);
    //    REGISTER_KERNEL_TEMPLATE_2(0xAA0019, TRadixSortKernel, i32, ui32);
    //    REGISTER_KERNEL_TEMPLATE_2(0xAA0020, TRadixSortKernel, i32, i32);
    //    REGISTER_KERNEL_TEMPLATE_2(0xAA0021, TRadixSortKernel, i32, float);

}
