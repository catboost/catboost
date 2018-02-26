#include "splitter.h"

using namespace NKernelHost;
namespace NCudaLib {
    REGISTER_KERNEL(0xEAAA00, TWriteCompressedSplitFloatKernel);
    REGISTER_KERNEL(0xEAAA01, TWriteCompressedSplitKernel);
    REGISTER_KERNEL(0xEAAA02, TUpdateBinsKernel);
    REGISTER_KERNEL(0xEAAA03, TUpdateBinsFromCompressedIndexKernel);
}
