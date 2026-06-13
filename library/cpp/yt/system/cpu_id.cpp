#include "cpu_id.h"

#include <library/cpp/yt/rseq/rseq.h>

#include <library/cpp/yt/misc/tls.h>

#if defined(__linux__)
#include <sched.h>
#endif

namespace NYT::NDetail {

////////////////////////////////////////////////////////////////////////////////

#ifdef YT_RSEQ_AVAILABLE

YT_PREVENT_TLS_CACHING int GetCurrentCpuIdSlow()
{
    if (NRseq::EnsureCurrentThreadRegistered()) {
        auto cpuId = NRseq::ReadField<int>(NRseq::CpuIdFieldOffset);
        if (cpuId >= 0) {
            return cpuId;
        }
    }

    auto cpuId = ::sched_getcpu();
    return cpuId >= 0 ? cpuId : 0;
}

#elif defined(__linux__)

// Linux without a known fast path (e.g. an unsupported arch): use sched_getcpu.
int GetCurrentCpuIdSlow()
{
    auto cpuId = ::sched_getcpu();
    return cpuId >= 0 ? cpuId : 0;
}

#else

// No rseq and no sched_getcpu (darwin / windows): sharding degenerates to one shard.
int GetCurrentCpuIdSlow()
{
    return 0;
}

#endif

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NDetail
