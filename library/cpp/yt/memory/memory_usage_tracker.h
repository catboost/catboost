#pragma once

#include "ref.h"

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

struct ISimpleMemoryUsageTracker
    : public TRefCounted
{
    virtual bool Acquire(i64 size) = 0;
    virtual void Release(i64 size) = 0;
    virtual TSharedRef Track(TSharedRef reference, bool keepExistingTracking) = 0;
};

DEFINE_REFCOUNTED_TYPE(ISimpleMemoryUsageTracker)

////////////////////////////////////////////////////////////////////////////////

class TSimpleMemoryUsageTrackerGuard
    : private TNonCopyable
{
public:
    TSimpleMemoryUsageTrackerGuard() = default;
    TSimpleMemoryUsageTrackerGuard(const TSimpleMemoryUsageTrackerGuard& other) = delete;
    TSimpleMemoryUsageTrackerGuard(TSimpleMemoryUsageTrackerGuard&& other);
    ~TSimpleMemoryUsageTrackerGuard();

    TSimpleMemoryUsageTrackerGuard& operator=(const TSimpleMemoryUsageTrackerGuard& other) = delete;
    TSimpleMemoryUsageTrackerGuard& operator=(TSimpleMemoryUsageTrackerGuard&& other);

    static TSimpleMemoryUsageTrackerGuard Build(ISimpleMemoryUsageTrackerPtr tracker);

    void SetSize(i64 size);

    void Release();

private:
    ISimpleMemoryUsageTrackerPtr Tracker_;
    i64 AcquiredSize_ = 0;
    void MoveFrom(TSimpleMemoryUsageTrackerGuard&& other);

};

////////////////////////////////////////////////////////////////////////////////

ISimpleMemoryUsageTrackerPtr GetNullSimpleMemoryUsageTracker();

TSharedRef TrackMemory(
    const ISimpleMemoryUsageTrackerPtr& tracker,
    TSharedRef reference,
    bool keepExistingTracking = false);

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
