#pragma once

#include <cuda_runtime.h>
#include <util/generic/yexception.h>

class TCudaException : public TWithBackTrace<yexception> {
public:
    TCudaException(cudaError_t error)
    : Error_(error) {
        cudaGetDevice(&DeviceId_);
    }


    cudaError_t Error() const {
        return Error_;
    }

    int DeviceId() const {
        return DeviceId_;
    }
private:
    cudaError_t Error_;
    int DeviceId_ = -1;
};


class TCudaEnsureException : public TWithBackTrace<yexception> {
public:
    TCudaEnsureException() {
    }
};

#define CUDA_ENSURE_IMPL_1(CONDITION) Y_ENSURE_EX( \
     CONDITION, \
     TCudaEnsureException() << "Condition violated: `" Y_STRINGIZE(CONDITION) "'"sv \
)

#define CUDA_ENSURE_IMPL_2(CONDITION, MESSAGE) Y_ENSURE_EX(CONDITION, TCudaEnsureException() << MESSAGE)

#define CUDA_ENSURE(...) \
    Y_PASS_VA_ARGS(Y_MACRO_IMPL_DISPATCHER_2(__VA_ARGS__, CUDA_ENSURE_IMPL_2, CUDA_ENSURE_IMPL_1)(__VA_ARGS__))
