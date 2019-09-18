#include "stream_pool.h"

void NCudaNN::TStreamPool::RequestSize(ui64 requestedSize)  {
    ui64 currentSize = 1 + HelperStreams_.size();
    for (ui64 i = currentSize; i < requestedSize; ++i) {
        HelperStreams_.push_back(TCudaStream::NewStream());
    }
}
NCudaNN::TStreamPool::TStreamPool(TCudaStream defaultStream, ui64 initSize)
: DefaultStream_(defaultStream) {
    RequestSize(initSize);
}
