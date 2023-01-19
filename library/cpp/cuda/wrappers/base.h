#pragma once

#include "kernel.cuh"

#include <library/cpp/cuda/exception/exception.h>

#include <util/datetime/base.h>
#include <util/generic/array_ref.h>

#include <cuda_runtime.h>

#define CUDA_SAFE_CALL(statement)                                                                                    \
    {                                                                                                                \
        cudaError_t errorCode = statement;                                                                           \
        if (errorCode != cudaSuccess && errorCode != cudaErrorCudartUnloading) {                                     \
            ythrow TCudaException(errorCode) << "CUDA error " << (int)errorCode << ": " << cudaGetErrorString(errorCode); \
        }                                                                                                            \
    }

#ifdef _MSC_VER
#define CUDA_DISABLE_4297_WARN __pragma(warning(push)); __pragma(warning(disable:4297))
#define CUDA_RESTORE_WARNINGS __pragma(warning(pop))
#elif defined(__GNUC__) || defined(__clang__)
#define CUDA_DISABLE_4297_WARN _Pragma("GCC diagnostic push") _Pragma("GCC diagnostic ignored \"-Wexceptions\"")
#define CUDA_RESTORE_WARNINGS _Pragma("GCC diagnostic pop")
#else
#define CUDA_DISABLE_4297_WARN
#define CUDA_RESTORE_WARNINGS
#endif

#define CUDA_SAFE_CALL_FOR_DESTRUCTOR(statement)                                                                                    \
    {                                                                                                                \
        cudaError_t errorCode = statement;                                                                           \
        if (errorCode != cudaSuccess && errorCode != cudaErrorCudartUnloading) {                                     \
            if (UncaughtException()) {                                                                               \
                Cerr << "Got CUDA error " << (int)errorCode << ": " << cudaGetErrorString(errorCode);                \
                Cerr << " while processing exception: " << CurrentExceptionMessage() << Endl;                        \
            } else {                                                                                                 \
                CUDA_DISABLE_4297_WARN                                                                               \
                ythrow TCudaException(errorCode) << "CUDA error " << (int)errorCode << ": " << cudaGetErrorString(errorCode); \
                CUDA_RESTORE_WARNINGS                                                                                 \
            }                                                                                                        \
        }                                                                                                            \
    }

class TCudaEvent;

enum class EStreamPriority {
    Default,
    Low,
    High
};

class TCudaStream {
private:
    struct TImpl: public TThrRefBase, public TNonCopyable {
    public:
        ~TImpl() {
            CUDA_SAFE_CALL_FOR_DESTRUCTOR(cudaStreamDestroy(Stream_));
        }

        explicit TImpl(bool nonBlocking, EStreamPriority streamPriority)
            : NonBlocking(nonBlocking)
        {
            if (streamPriority == EStreamPriority::Default) {
                if (nonBlocking) {
                    CUDA_SAFE_CALL(cudaStreamCreateWithFlags(&Stream_, cudaStreamNonBlocking));
                } else {
                    CUDA_SAFE_CALL(cudaStreamCreate(&Stream_));
                }
            } else {
                int leastPriority = 0;
                int greatestPriority = 0;
                CUDA_SAFE_CALL(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
                CUDA_ENSURE(nonBlocking, "non default priority for nonBlocking streams only");
                int priority = leastPriority ? streamPriority == EStreamPriority::Low : greatestPriority;
                CUDA_SAFE_CALL(cudaStreamCreateWithPriority(&Stream_, cudaStreamNonBlocking, priority));
            }
        }

        cudaStream_t Stream_;
        bool NonBlocking = false;
    };

    using TStreamPtr = TIntrusivePtr<TImpl>;

    TCudaStream(TStreamPtr ptr)
        : Stream_(ptr)
    {
    }

    TCudaStream() {
    }

private:
    TStreamPtr Stream_;

public:
    void Synchronize() const {
        CUDA_SAFE_CALL(cudaStreamSynchronize(GetStream()));
    }

    operator cudaStream_t() const {
        return GetStream();
    }

    cudaStream_t GetStream() const {
        return Stream_ ? Stream_->Stream_ : cudaStreamPerThread;
    }

    static TCudaStream ZeroStream() {
        return TCudaStream();
    }

    static TCudaStream NewStream(bool nonBlocking = true, EStreamPriority streamPriority = EStreamPriority::Default) {
        return TCudaStream(new TImpl(nonBlocking, streamPriority));
    }

    bool operator==(const TCudaStream& other) const {
        return GetStream() == other.GetStream();
    }

    void WaitEvent(const TCudaEvent& event) const;
};

inline void DeviceSynchronize() {
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
}

class TDeviceGuard: private TNonCopyable {
public:
    TDeviceGuard(int device) {
        CUDA_SAFE_CALL(cudaGetDevice(&PreviousDevice));
        if (device != PreviousDevice) {
            CUDA_SAFE_CALL(cudaSetDevice(device));
        } else {
            PreviousDevice = -1;
        }
    }

    ~TDeviceGuard() {
        if (PreviousDevice != -1) {
            CUDA_SAFE_CALL_FOR_DESTRUCTOR(cudaSetDevice(PreviousDevice));
        }
    }

private:
    int PreviousDevice = -1;
};


//return 0 if success and 1 otherwise
void GetMemoryInfo(int device, size_t* available, size_t* total);
int GetDeviceCount();

class TProfile: private TNonCopyable {
public:
    TProfile(const TString& message)
        : Message_(message)
    {
        DeviceSynchronize();
        Start = TInstant::Now();
    }

    ~TProfile() {
        DeviceSynchronize();
        Cout << Message_ << " in " << (TInstant::Now() - Start).SecondsFloat() << " seconds" << Endl;
    }

private:
    TString Message_;
    TInstant Start;
};
//

template <class T>
inline void ClearMemoryAsync(TArrayRef<T> data, TCudaStream stream) {
    CUDA_SAFE_CALL(cudaMemsetAsync(reinterpret_cast<char*>(data.data()), 0, data.size() * sizeof(T), stream.GetStream()));
}
