#include "memory_usage_tracker.h"

#include "leaky_ref_counted_singleton.h"

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

class TNulSimplelMemoryUsageTracker
    : public ISimpleMemoryUsageTracker
{
public:
    bool Acquire(i64 /*size*/) override
    {
        return false;
    }

    void Release(i64 /*size*/) override
    { }
};

////////////////////////////////////////////////////////////////////////////////

ISimpleMemoryUsageTrackerPtr GetNullSimpleMemoryUsageTracker()
{
    return LeakyRefCountedSingleton<TNulSimplelMemoryUsageTracker>();
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
