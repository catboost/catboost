#include "base.h"
#include "cuda_event.h"


void GetMemoryInfo(int device, size_t* available, size_t* total) {
    *available = *total = 0;
    TDeviceGuard guard(device);
    CUDA_SAFE_CALL(cudaMemGetInfo(available, total))
}


void TCudaStream::WaitEvent(const TCudaEvent& event) const {
    event.StreamWait(*this);
}
