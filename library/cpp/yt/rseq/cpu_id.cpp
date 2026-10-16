#include "cpu_id.h"
#include "rseq.h"
#include "rseq_area.h"

#include <library/cpp/yt/misc/tls.h>

#include <sched.h>

namespace NYT::NRseq::NDetail {

////////////////////////////////////////////////////////////////////////////////

YT_PREVENT_TLS_CACHING int GetCurrentCpuIdSlow()
{
    if (EnsureCurrentThreadRegistered()) {
        auto cpuId = ReadRseqAreaField<int>(offsetof(TRseqArea, CpuId));
        if (cpuId >= 0) {
            return cpuId;
        }
    }
    auto cpuId = ::sched_getcpu();
    return cpuId >= 0 ? cpuId : 0;
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NRseq::NDetail
