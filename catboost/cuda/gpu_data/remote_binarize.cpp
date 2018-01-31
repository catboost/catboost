#include "remote_binarize.h"

using namespace NKernelHost;

namespace NCudaLib {
    REGISTER_KERNEL(0xBBA000, TFindBordersKernel)
    REGISTER_KERNEL(0xBBA001, TBinarizeFloatFeatureKernel)
}
