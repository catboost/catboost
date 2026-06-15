#ifndef CPU_ID_INL_H_
#error "Direct inclusion of this file is not allowed, include cpu_id.h"
// For the sake of sane code completion.
#include "cpu_id.h"
#endif
#undef CPU_ID_INL_H_

#ifdef __linux__
#include <library/cpp/yt/rseq/rseq.h>
#endif

#include <util/system/compiler.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

namespace NDetail {

int GetCurrentCpuIdSlow();

} // namespace NDetail

Y_FORCE_INLINE int GetCurrentCpuId()
{
#ifdef YT_RSEQ_AVAILABLE
    // Branch-free read of the rseq cpu_id: the offset always points at a readable
    // field. A data-dependent branch on the offset here would defeat load pipelining.
    auto cpuId = NRseq::ReadField<int>(NRseq::CpuIdFieldOffset);
    // Negative means this thread is not registered yet or rseq is unavailable.
    if (cpuId < 0) [[unlikely]] {
        return NDetail::GetCurrentCpuIdSlow();
    }
    return cpuId;
#else
    return NDetail::GetCurrentCpuIdSlow();
#endif
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
