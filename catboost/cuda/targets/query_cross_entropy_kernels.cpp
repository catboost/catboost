#include "query_cross_entropy_kernels.h"

using namespace NKernelHost;

namespace NCudaLib {
    REGISTER_KERNEL(0xA138200, TQueryCrossEntropyKernel);
    REGISTER_KERNEL(0xA138201, TComputeQueryLogitMatrixSizesKernel);
    REGISTER_KERNEL(0xA138202, TMakeQueryLogitPairsKernel);
    REGISTER_KERNEL(0xA138204, TMakeIsSingleClassFlagsKernel);
    REGISTER_KERNEL(0xA138205, TFillPairDer2AndRemapPairDocumentsKernel);

}
