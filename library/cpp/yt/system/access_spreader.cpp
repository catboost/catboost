#include "access_spreader.h"

#include <library/cpp/yt/misc/tls.h>

#ifdef __linux__
#include <sched.h>
#endif

#include <atomic>
#include <type_traits>

namespace NYT {
namespace {

////////////////////////////////////////////////////////////////////////////////

#ifdef __linux__
constexpr int AccessStripeSeedRefreshPeriod = 32;
#endif

struct TAccessStripeSeedState
{
    ui32 Seed = 0;
    int RemainingUses = 0;
    ui32 FallbackSeed = 0;
    bool FallbackInitialized = false;
};

static_assert(std::is_trivially_destructible_v<TAccessStripeSeedState>);

constinit std::atomic<ui32> NextFallbackSeed = 0;
constinit thread_local TAccessStripeSeedState AccessStripeSeedState;

ui32 GetFallbackSeed(TAccessStripeSeedState* state)
{
    if (!state->FallbackInitialized) {
        state->FallbackSeed = NextFallbackSeed.fetch_add(1, std::memory_order::relaxed);
        state->FallbackInitialized = true;
    }
    return state->FallbackSeed;
}

////////////////////////////////////////////////////////////////////////////////

} // namespace

////////////////////////////////////////////////////////////////////////////////

namespace NDetail {

YT_PREVENT_TLS_CACHING ui32 GetAccessStripeSeed()
{
#ifdef __linux__
    if (AccessStripeSeedState.RemainingUses == 0) {
        auto cpuId = ::sched_getcpu();
        AccessStripeSeedState.Seed = cpuId >= 0
            ? static_cast<ui32>(cpuId)
            : GetFallbackSeed(&AccessStripeSeedState);

        // On the successful sampling path, a migration may leave at most one 32-call
        // sampling interval using a stale stripe. This cadence amortizes sched_getcpu;
        // stripe staleness affects contention only, not correctness.
        AccessStripeSeedState.RemainingUses = AccessStripeSeedRefreshPeriod - 1;
    } else {
        --AccessStripeSeedState.RemainingUses;
    }

    return AccessStripeSeedState.Seed;
#else
    return GetFallbackSeed(&AccessStripeSeedState);
#endif
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NDetail
} // namespace NYT
