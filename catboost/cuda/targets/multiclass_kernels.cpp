#include "multiclass_kernels.h"

using namespace NKernelHost;

namespace NCudaLib {
    REGISTER_KERNEL(0xA11BB00, TMultiLogitValueAndDerKernel);
    REGISTER_KERNEL(0xA11BB01, TMultiLogitSecondDerKernel);

    REGISTER_KERNEL(0xA11BB02, TMultiClassOneVsAllValueAndDerKernel);
    REGISTER_KERNEL(0xA11BB03, TMultiClassOneVsAllSecondDerKernel);
    REGISTER_KERNEL(0xA11BB04, TBuildConfusionMatrixKernel);

    REGISTER_KERNEL(0xA11BB05, TRMSEWithUncertaintyValueAndDerKernel);
    REGISTER_KERNEL(0xA11BB06, TRMSEWithUncertaintySecondDerKernel);

    REGISTER_KERNEL(0xA11BB07, TMultiCrossEntropyValueAndDerKernel);
    REGISTER_KERNEL(0xA11BB08, TMultiCrossEntropySecondDerKernel);

    REGISTER_KERNEL(0xA11BB09, TMultiRMSEValueAndDerKernel);
    REGISTER_KERNEL(0xA11BB0A, TMultiRMSESecondDerKernel);
}
