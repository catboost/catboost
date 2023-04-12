#include "cuda_event.h"


bool TCudaEvent::IsComplete() const {
    cudaError_t errorCode = cudaEventQuery(Inner_->Event);
    if (errorCode == cudaSuccess) {
        return true;
    }
    if (errorCode != cudaErrorNotReady) {
        ythrow TCudaException(errorCode) << "CUDA error " << (int)errorCode << ": " << cudaGetErrorString(errorCode);
    }
    return false;
}

void TCudaEvent::WaitComplete() const  {
    CUDA_SAFE_CALL(cudaEventSynchronize(Inner_->Event));
}

void TCudaEvent::StreamWait(const TCudaStream& stream) const  {
    CUDA_SAFE_CALL(cudaStreamWaitEvent(stream, Inner_->Event, 0));
}

void TCudaEvent::Record(cudaStream_t stream) const {
    CUDA_SAFE_CALL(cudaEventRecord(Inner_->Event, stream));
}

void TCudaEvent::Record(const TCudaStream& stream) const {
    Record(stream.GetStream());
}

TCudaEvent TCudaEvent::NewEvent(bool disableTiming)  {
    return TCudaEvent(new Inner(disableTiming));
}
TCudaEvent::TCudaEvent(TIntrusivePtr<Inner> event)
    : Inner_(std::move(event)) {
}

void TCudaEvent::Swap(TCudaEvent& other)  {
    Inner_.Swap(other.Inner_);
}
TCudaEvent::Inner::Inner(bool disableTiming)
    : WithoutTiming(disableTiming) {
    if (disableTiming) {
        CUDA_SAFE_CALL(cudaEventCreateWithFlags(&Event, cudaEventDisableTiming | cudaEventBlockingSync));
    } else {
        CUDA_SAFE_CALL(cudaEventCreate(&Event));
    }
}
