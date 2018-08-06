#include "partitions_reduce.h"

using namespace NKernelHost;

namespace NCudaLib {
    REGISTER_KERNEL(0xAADDD1, TReducePartitionsKernel);
    REGISTER_KERNEL(0xAADDD2, TReducePartitionsWithOffsetsKernel);
    REGISTER_KERNEL(0xAADDD3, TCastCopyKernel);
}
