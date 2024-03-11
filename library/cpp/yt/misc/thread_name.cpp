#include "thread_name.h"

#include <library/cpp/yt/misc/tls.h>

#include <util/generic/string.h>
#include <util/system/thread.h>

#include <algorithm>
#include <cstring>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

TStringBuf TThreadName::ToStringBuf() const
{
    // Buffer is zero terminated.
    return TStringBuf(Buffer.data(), Length);
}

////////////////////////////////////////////////////////////////////////////////

TThreadName::TThreadName(const TString& name)
{
    Length = std::min<int>(TThreadName::BufferCapacity - 1, name.length());
    ::memcpy(Buffer.data(), name.data(), Length);
}

////////////////////////////////////////////////////////////////////////////////

// This function uses cached TThread::CurrentThreadName() result
TThreadName GetCurrentThreadName()
{
    static YT_THREAD_LOCAL(TThreadName) ThreadName;
    auto& threadName = GetTlsRef(ThreadName);

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
