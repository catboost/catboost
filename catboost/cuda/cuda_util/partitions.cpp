#include "partitions.h"

using namespace NKernelHost;

namespace NCudaLib {
    REGISTER_KERNEL(0xAAA001, TUpdatePartitionDimensionsKernel);
    REGISTER_KERNEL(0xAAA002, TUpdatePartitionOffsetsKernel);
}
