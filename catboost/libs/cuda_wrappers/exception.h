#pragma once

#include <cuda_runtime.h>
#include <util/generic/bt_exception.h>

class TCudaException : public TWithBackTrace<yexception> {
public:
    TCudaException(cudaError_t error)
    : Error_(error) {

    }


    cudaError_t Error() const {
        return Error_;
    }
private:
    cudaError_t Error_;
};
