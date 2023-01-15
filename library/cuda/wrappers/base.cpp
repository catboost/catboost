#include "base.h"
#include "cuda_event.h"

void TCudaStream::WaitEvent(const TCudaEvent& event) const {
    event.StreamWait(*this);
}
