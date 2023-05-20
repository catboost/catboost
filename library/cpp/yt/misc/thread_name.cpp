#include "thread_name.h"

#include <util/generic/string.h>
#include <util/system/thread.h>

#include <algorithm>
#include <cstring>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

TString TThreadName::ToString() const
{
    // Buffer is zero terminated.
    return Buffer.data();
}

////////////////////////////////////////////////////////////////////////////////

// This function uses cached TThread::CurrentThreadName() result
TThreadName GetCurrentThreadName()
{
    static thread_local TThreadName threadName;
    if (threadName.Length == 0) {
        if (auto name = TThread::CurrentThreadName()) {
            auto length = std::min<int>(TThreadName::BufferCapacity - 1, name.length());
            threadName.Length = length;
            ::memcpy(threadName.Buffer.data(), name.data(), length);
        }
    }
    return threadName;
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
