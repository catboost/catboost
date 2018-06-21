#include "cuda_base.h"

namespace NCudaLib {
    char* CudaHostAllocate(ui64 size) {
        void* ptr = nullptr;
        CUDA_SAFE_CALL(cudaHostAlloc(&ptr, size, cudaHostAllocPortable));
        return static_cast<char*>(ptr);
    }

    void CudaHostFree(char* ptr) {
        CUDA_SAFE_CALL(cudaFreeHost((void*)ptr));
    }

    cudaStream_t TCudaStreamsProvider::NewStream() {
        cudaStream_t stream;
        CUDA_SAFE_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        return stream;
    }
}
