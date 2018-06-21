#include "slice.h"
#include "cuda_buffer.h"
#include <util/stream/output.h>

template <>
void Out<TSlice>(IOutputStream& o, const TSlice& slice) {
    o.Write("[" + ToString(slice.Left) + "-" + ToString(slice.Right) + "]");
}

namespace NCudaLib {
    #define INST_CUDA_BUFFER(MAPPING)\
    template class TCudaBuffer<float, MAPPING, EPtrType::CudaDevice>;\
    template class TCudaBuffer<const float, MAPPING, EPtrType::CudaDevice>;\
    template class TCudaBuffer<ui32, MAPPING, EPtrType::CudaDevice>;\
    template class TCudaBuffer<const ui32, MAPPING, EPtrType::CudaDevice>;\
    template class TCudaBuffer<ui64, MAPPING, EPtrType::CudaDevice>;\
    template class TCudaBuffer<const ui64, MAPPING, EPtrType::CudaDevice>;\
    template class TCudaBuffer<i32, MAPPING, EPtrType::CudaDevice>;\
    template class TCudaBuffer<const i32, MAPPING, EPtrType::CudaDevice>;\
    template class TCudaBuffer<i64, MAPPING, EPtrType::CudaDevice>;\
    template class TCudaBuffer<const i64, MAPPING, EPtrType::CudaDevice>;\
    template class TCudaBuffer<double, MAPPING, EPtrType::CudaDevice>;\
    template class TCudaBuffer<const double, MAPPING, EPtrType::CudaDevice>;

    INST_CUDA_BUFFER(NCudaLib::TStripeMapping)
    INST_CUDA_BUFFER(NCudaLib::TMirrorMapping)
    INST_CUDA_BUFFER(NCudaLib::TSingleMapping)

    #undef INST_CUDA_BUFFER
}
