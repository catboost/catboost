#include "reduce_scatter.h"

namespace NCudaLib {
    REGISTER_KERNEL_TEMPLATE(0x000023, NKernelHost::TShiftMemoryKernel, float);

    REGISTER_STREAM_SECTION_TASK_TEMPLATE(0x010001, TReduceBinaryStreamTask, float);
}
