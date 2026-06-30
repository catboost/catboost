#include "base.h"
#include "cuda_event.h"

namespace NCuda {
    void GetMemoryInfo(int device, size_t* available, size_t* total) {
        *available = *total = 0;
        NCuda::TDeviceGuard guard(device);
        CUDA_SAFE_CALL(cudaMemGetInfo(available, total))
    }

    int GetDeviceCount() {
        int deviceCount = 0;
        CUDA_SAFE_CALL(cudaGetDeviceCount(&deviceCount));
        return deviceCount;
    }
}

void TCudaStream::WaitEvent(const TCudaEvent& event) const {
    event.StreamWait(*this);
}

