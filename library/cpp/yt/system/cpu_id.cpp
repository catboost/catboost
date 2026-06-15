#include "cpu_id.h"

#include <library/cpp/yt/misc/tls.h>

#ifdef __linux__
#include <library/cpp/yt/rseq/rseq.h>

#include <sched.h>
#endif

namespace NYT::NDetail {

////////////////////////////////////////////////////////////////////////////////

#ifdef __linux__

YT_PREVENT_TLS_CACHING int GetCurrentCpuIdSlow()
{
#ifdef YT_RSEQ_AVAILABLE
    if (NRseq::EnsureCurrentThreadRegistered()) {
        auto cpuId = NRseq::ReadField<int>(NRseq::CpuIdFieldOffset);
        if (cpuId >= 0) {
            return cpuId;
        }
    }
#endif
    // No rseq fast path (unsupported arch, or thread not registered): sched_getcpu.
    auto cpuId = ::sched_getcpu();
    return cpuId >= 0 ? cpuId : 0;
}

#else

// No sched_getcpu (darwin / windows): sharding degenerates to one shard.
int GetCurrentCpuIdSlow()
{
    return 0;
}

#endif

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NDetail
