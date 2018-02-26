#include "kernels.h"

using namespace NKernelHost;

namespace NCudaLib {
    REGISTER_KERNEL(0xBBA000, TFindBordersKernel)
    REGISTER_KERNEL(0xBBA001, TBinarizeFloatFeatureKernel)
    REGISTER_KERNEL_TEMPLATE(0xBBA002, TWriteCompressedIndexKernel, EPtrType::CudaHost)
    REGISTER_KERNEL_TEMPLATE(0xBBA003, TWriteCompressedIndexKernel, EPtrType::CudaDevice)
}
