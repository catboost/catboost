#include "stream_pool.h"
#include "cuda_event.h"

void NCudaNN::TStreamPool::RequestSize(ui64 requestedSize)  {
    ui64 currentSize = 1 + HelperStreams_.size();
    for (ui64 i = currentSize; i < requestedSize; ++i) {
        HelperStreams_.push_back(TCudaStream::NewStream());
        SyncEvents_.push_back(TCudaEvent::NewEvent());
    }
}

NCudaNN::TStreamPool::TStreamPool(TCudaStream defaultStream, ui64 initSize)
: DefaultStream_(defaultStream)
, DefaultEvent_(TCudaEvent::NewEvent()) {
    RequestSize(initSize);
}

void NCudaNN::TStreamPool::PrefixWaitForDefault(ui64 size) {
    CUDA_ENSURE(size <= SyncEvents_.size() + 1);
    DefaultEvent_.Record(DefaultStream_);
    for (ui64 i = 1; i < size; ++i) {
        DefaultEvent_.StreamWait(HelperStreams_[i - 1]);
    }
}

void NCudaNN::TStreamPool::DefaultWaitForPrefix(ui64 size) {
    CUDA_ENSURE(size <= SyncEvents_.size() + 1);
    for (ui64 i = 1; i < size; ++i) {
        SyncEvents_[i - 1].Record(HelperStreams_[i - 1]);
        SyncEvents_[i - 1].StreamWait(DefaultStream_);
    }
}
