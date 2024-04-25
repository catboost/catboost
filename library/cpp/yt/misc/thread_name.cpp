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
YT_PREVENT_TLS_CACHING TThreadName GetCurrentThreadName()
{
    static thread_local TThreadName ThreadName;

    if (ThreadName.Length == 0) {
        if (auto name = TThread::CurrentThreadName()) {
            auto length = std::min<int>(TThreadName::BufferCapacity - 1, name.length());
            ThreadName.Length = length;
            ::memcpy(ThreadName.Buffer.data(), name.data(), length);
        }
    }
    return ThreadName;
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
