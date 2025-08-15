#include "simple_memory_usage_tracker.h"

#include "leaky_ref_counted_singleton.h"

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

class TNullSimpleMemoryUsageTracker
    : public ISimpleMemoryUsageTracker
{
public:
    bool Acquire(i64 /*size*/) override
    {
        return false;
    }

    void Release(i64 /*size*/) override
    { }

    TSharedRef Track(TSharedRef reference, bool /*keepExistingTracking*/) override
    {
        return reference;
    }
};

////////////////////////////////////////////////////////////////////////////////

TSimpleMemoryUsageTrackerGuard::TSimpleMemoryUsageTrackerGuard(TSimpleMemoryUsageTrackerGuard&& other)
{
    MoveFrom(std::move(other));
}

TSimpleMemoryUsageTrackerGuard::~TSimpleMemoryUsageTrackerGuard()
{
    Release();
}

TSimpleMemoryUsageTrackerGuard& TSimpleMemoryUsageTrackerGuard::operator=(TSimpleMemoryUsageTrackerGuard&& other)
{
    if (this != &other) {
        Release();
        MoveFrom(std::move(other));
    }
    return *this;
}

void TSimpleMemoryUsageTrackerGuard::MoveFrom(TSimpleMemoryUsageTrackerGuard&& other)
{
    Tracker_ = other.Tracker_;
    AcquiredSize_ = other.AcquiredSize_;

    other.Tracker_ = nullptr;
    other.AcquiredSize_ = 0;
}

TSimpleMemoryUsageTrackerGuard TSimpleMemoryUsageTrackerGuard::Build(ISimpleMemoryUsageTrackerPtr tracker)
{
    if (!tracker) {
        return {};
    }

    TSimpleMemoryUsageTrackerGuard guard;
    guard.Tracker_ = tracker;
    return guard;
}

void TSimpleMemoryUsageTrackerGuard::Release()
{
    if (Tracker_) {
        if (AcquiredSize_) {
            Tracker_->Release(AcquiredSize_);
        }

        Tracker_.Reset();
        AcquiredSize_ = 0;
    }
}

void TSimpleMemoryUsageTrackerGuard::SetSize(i64 size)
{
    if (!Tracker_) {
        return;
    }

    YT_VERIFY(size >= 0);

    if (size > AcquiredSize_) {
        Tracker_->Acquire(size - AcquiredSize_);
    } else if (size < AcquiredSize_) {
        Tracker_->Release(AcquiredSize_ - size);
    }

    AcquiredSize_ = size;
}

////////////////////////////////////////////////////////////////////////////////

ISimpleMemoryUsageTrackerPtr GetNullSimpleMemoryUsageTracker()
{
    return LeakyRefCountedSingleton<TNullSimpleMemoryUsageTracker>();
}

TSharedRef TrackMemory(
    const ISimpleMemoryUsageTrackerPtr& tracker,
    TSharedRef reference,
    bool keepExistingTracking)
{
    if (!tracker || !reference) {
        return reference;
    }
    return tracker->Track(std::move(reference), keepExistingTracking);
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
