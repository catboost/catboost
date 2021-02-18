#include "kernels.h"

using namespace NKernelHost;

namespace NCudaLib {
    REGISTER_KERNEL(0xBBA000, TFindBordersKernel)
    REGISTER_KERNEL(0xBBA001, TBinarizeFloatFeatureKernel)
    REGISTER_KERNEL_TEMPLATE(0xBBA002, TWriteCompressedIndexKernel, EPtrType::CudaHost)
    REGISTER_KERNEL_TEMPLATE(0xBBA003, TWriteCompressedIndexKernel, EPtrType::CudaDevice)
    REGISTER_KERNEL(0xBBAA004, TComputeQueryIdsKernel)
    REGISTER_KERNEL(0xBBAA005, TFillQueryEndMaskKernel)
    REGISTER_KERNEL(0xBBAA006, TCreateKeysForSegmentedDocsSampleKernel)
    REGISTER_KERNEL(0xBBAA007, TFillTakenDocsMaskKernel)
    REGISTER_KERNEL(0xBBAA008, TRemoveQueryMeans)
    REGISTER_KERNEL(0xBBAA009, TRemoveQueryMax)
    REGISTER_KERNEL(0xBBA00A, TWriteLazyCompressedIndexKernel)
    REGISTER_KERNEL(0xBBA00B, TDropAllLoaders)
}
