#include "compression_helpers_gpu.h"

using namespace NKernelHost;

namespace NCudaLib {
    REGISTER_KERNEL_TEMPLATE_2(0xFAFA01, TDecompressKernel, ui64, EPtrType::CudaHost)
    REGISTER_KERNEL_TEMPLATE_2(0xFAFA02, TDecompressKernel, ui64, EPtrType::CudaDevice)

    REGISTER_KERNEL_TEMPLATE_2(0xFAFA03, TCompressKernel, ui64, EPtrType::CudaHost)
    REGISTER_KERNEL_TEMPLATE_2(0xFAFA04, TCompressKernel, ui64, EPtrType::CudaDevice)

    REGISTER_KERNEL_TEMPLATE_2(0xFAFA05, TGatherFromCompressedKernel, ui64, EPtrType::CudaHost)
    REGISTER_KERNEL_TEMPLATE_2(0xFAFA06, TGatherFromCompressedKernel, ui64, EPtrType::CudaDevice)

    REGISTER_KERNEL_TEMPLATE_2(0xFAFB01, TDecompressKernel, ui32, EPtrType::CudaHost)
    REGISTER_KERNEL_TEMPLATE_2(0xFAFB02, TDecompressKernel, ui32, EPtrType::CudaDevice)

    REGISTER_KERNEL_TEMPLATE_2(0xFAFB03, TCompressKernel, ui32, EPtrType::CudaHost)
    REGISTER_KERNEL_TEMPLATE_2(0xFAFB04, TCompressKernel, ui32, EPtrType::CudaDevice)

    REGISTER_KERNEL_TEMPLATE_2(0xFAFB05, TGatherFromCompressedKernel, ui32, EPtrType::CudaHost)
    REGISTER_KERNEL_TEMPLATE_2(0xFAFB06, TGatherFromCompressedKernel, ui32, EPtrType::CudaDevice)
}
