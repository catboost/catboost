#include "samples_grouping_gpu.h"
namespace NCatboostCuda {
    template class TGpuSamplesGroupingHelper<NCudaLib::TStripeMapping>;
    template class TGpuSamplesGroupingHelper<NCudaLib::TMirrorMapping>;
}
