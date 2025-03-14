#include "memory_usage_tracker.h"

#include "leaky_ref_counted_singleton.h"

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

class TNullMemoryUsageTracker
    : public IMemoryUsageTracker
{
public:
    TError TryAcquire(i64 /*size*/) override
    {
        return {};
    }

    TError TryChange(i64 /*size*/) override
    {
        return {};
    }

    bool Acquire(i64 /*size*/) override
    {
        return false;
    }

    void Release(i64 /*size*/) override
    { }

    void SetLimit(i64 /*size*/) override
    { }

    i64 GetLimit()  const override
    {
        return std::numeric_limits<i64>::max();
    }

    i64 GetUsed() const override
    {
        return 0;
    }

    i64 GetFree() const override
    {
        return std::numeric_limits<i64>::max();
    }

    bool IsExceeded() const override
    {
        return false;
    }

    TSharedRef Track(
        TSharedRef reference,
        bool /*keepExistingTracking*/) override
    {
        return reference;
    }

    TErrorOr<TSharedRef> TryTrack(
        TSharedRef reference,
        bool /*keepExistingTracking*/) override
    {
        return reference;
    }
};

////////////////////////////////////////////////////////////////////////////////

IMemoryUsageTrackerPtr GetNullMemoryUsageTracker()
{
    return LeakyRefCountedSingleton<TNullMemoryUsageTracker>();
}

////////////////////////////////////////////////////////////////////////////////

TMemoryUsageTrackerGuard::TMemoryUsageTrackerGuard(TMemoryUsageTrackerGuard&& other)
{
    MoveFrom(std::move(other));
}

TMemoryUsageTrackerGuard::~TMemoryUsageTrackerGuard()
{
    Release();
}

TMemoryUsageTrackerGuard& TMemoryUsageTrackerGuard::operator=(TMemoryUsageTrackerGuard&& other)
{
    if (this != &other) {
        Release();
        MoveFrom(std::move(other));
    }
    return *this;
}

void TMemoryUsageTrackerGuard::MoveFrom(TMemoryUsageTrackerGuard&& other)
{
    Tracker_ = other.Tracker_;
    Size_ = other.Size_;
    AcquiredSize_ = other.AcquiredSize_;
    Granularity_ = other.Granularity_;

    other.Tracker_ = nullptr;
    other.Size_ = 0;
    other.AcquiredSize_ = 0;
    other.Granularity_ = 0;
}

TMemoryUsageTrackerGuard TMemoryUsageTrackerGuard::Build(
    IMemoryUsageTrackerPtr tracker,
    i64 granularity)
{
    if (!tracker) {
        return {};
    }

    TMemoryUsageTrackerGuard guard;
    guard.Tracker_ = tracker;
    guard.Size_ = 0;
    guard.Granularity_ = granularity;
    return guard;
}

TMemoryUsageTrackerGuard TMemoryUsageTrackerGuard::Acquire(
    IMemoryUsageTrackerPtr tracker,
    i64 size,
    i64 granularity)
{
    if (!tracker) {
        return {};
    }

    YT_VERIFY(size >= 0);
    TMemoryUsageTrackerGuard guard;
    guard.Tracker_ = tracker;
    guard.Size_ = size;
    guard.Granularity_ = granularity;
    if (size >= granularity) {
        guard.AcquiredSize_ = size;
        tracker->Acquire(size);
    }
    return guard;
}

TErrorOr<TMemoryUsageTrackerGuard> TMemoryUsageTrackerGuard::TryAcquire(
    IMemoryUsageTrackerPtr tracker,
    i64 size,
    i64 granularity)
{
    if (!tracker) {
        return {};
    }

    YT_VERIFY(size >= 0);

    auto error = tracker->TryAcquire(size);
    if (!error.IsOK()) {
        return error;
    }
    TMemoryUsageTrackerGuard guard;
    guard.Tracker_ = tracker;
    guard.Size_ = size;
    guard.AcquiredSize_ = size;
    guard.Granularity_ = granularity;
    return std::move(guard);
}

void TMemoryUsageTrackerGuard::Release()
{
    if (Tracker_) {
        if (AcquiredSize_) {
            Tracker_->Release(AcquiredSize_);
        }

        ReleaseNoReclaim();
    }
}

void TMemoryUsageTrackerGuard::ReleaseNoReclaim()
{
    Tracker_.Reset();
    Size_ = 0;
    AcquiredSize_ = 0;
    Granularity_ = 0;
}

TMemoryUsageTrackerGuard::operator bool() const
{
    return Tracker_.operator bool();
}

i64 TMemoryUsageTrackerGuard::GetSize() const
{
    return Size_;
}

void TMemoryUsageTrackerGuard::SetSize(i64 size)
{
    auto ignoredError = SetSizeImpl(size, [&] (i64 delta) {
        Tracker_->Acquire(delta);
        return TError{};
    });

    Y_UNUSED(ignoredError);
}

TError TMemoryUsageTrackerGuard::TrySetSize(i64 size)
{
    return SetSizeImpl(size, [&] (i64 delta) {
        return Tracker_->TryAcquire(delta);
    });
}

TError TMemoryUsageTrackerGuard::SetSizeImpl(i64 size, auto acquirer)
{
    if (!Tracker_) {
        return {};
    }

    YT_VERIFY(size >= 0);
    Size_ = size;
    if (std::abs(Size_ - AcquiredSize_) >= Granularity_) {
        if (Size_ > AcquiredSize_) {
            if (auto result = acquirer(Size_ - AcquiredSize_); !result.IsOK()) {
                return result;
            }
        } else {
            Tracker_->Release(AcquiredSize_ - Size_);
        }
        AcquiredSize_ = Size_;
    }

    return {};
}

void TMemoryUsageTrackerGuard::IncreaseSize(i64 sizeDelta)
{
    SetSize(Size_ + sizeDelta);
}

void TMemoryUsageTrackerGuard::DecreaseSize(i64 sizeDelta)
{
    SetSize(Size_ - sizeDelta);
}

TMemoryUsageTrackerGuard TMemoryUsageTrackerGuard::TransferMemory(i64 size)
{
    YT_VERIFY(Size_ >= size);

    auto acquiredDelta = std::min(AcquiredSize_, size);

    Size_ -= size;
    AcquiredSize_ -= acquiredDelta;

    TMemoryUsageTrackerGuard guard;
    guard.Tracker_ = Tracker_;
    guard.Size_ = size;
    guard.AcquiredSize_ = acquiredDelta;
    guard.Granularity_ = Granularity_;
    return std::move(guard);
}

////////////////////////////////////////////////////////////////////////////////

TMemoryTrackedBlob::TMemoryTrackedBlob(
    TBlob&& blob,
    TMemoryUsageTrackerGuard&& guard)
    : Blob_(std::move(blob))
    , Guard_(std::move(guard))
{ }

TMemoryTrackedBlob TMemoryTrackedBlob::Build(
    IMemoryUsageTrackerPtr tracker,
    TRefCountedTypeCookie tagCookie)
{
    YT_VERIFY(tracker);

    return TMemoryTrackedBlob(
        TBlob(tagCookie),
        TMemoryUsageTrackerGuard::Build(tracker));
}

void TMemoryTrackedBlob::Resize(
    i64 size,
    bool initializeStorage)
{
    YT_VERIFY(size >= 0);

    Blob_.Resize(size, initializeStorage);
    Guard_.SetSize(Blob_.Capacity());
}

TError TMemoryTrackedBlob::TryResize(
    i64 size,
    bool initializeStorage)
{
    YT_VERIFY(size >= 0);
    auto result = Guard_.TrySetSize(size);

    if (result.IsOK()) {
        Blob_.Resize(size, initializeStorage);
        return {};
    } else {
        return result;
    }
}

void TMemoryTrackedBlob::Reserve(i64 size)
{
    YT_VERIFY(size >= 0);

    Blob_.Reserve(size);
    Guard_.SetSize(Blob_.Capacity());
}

TError TMemoryTrackedBlob::TryReserve(i64 size)
{
    YT_VERIFY(size >= 0);

    auto result = Guard_.TrySetSize(size);

    if (result.IsOK()) {
        Blob_.Reserve(size);
        return {};
    } else {
        return result;
    }
}

char* TMemoryTrackedBlob::Begin()
{
    return Blob_.Begin();
}

char* TMemoryTrackedBlob::End()
{
    return Blob_.End();
}

size_t TMemoryTrackedBlob::Capacity() const
{
    return Blob_.Capacity();
}

size_t TMemoryTrackedBlob::Size() const
{
    return Blob_.Size();
}

void TMemoryTrackedBlob::Append(TRef ref)
{
    Blob_.Append(ref);
    Guard_.SetSize(Blob_.Capacity());
}

void TMemoryTrackedBlob::Clear()
{
    Blob_.Clear();
    Guard_.SetSize(Blob_.Capacity());
}

////////////////////////////////////////////////////////////////////////////////

TErrorOr<TSharedRef> TryTrackMemory(
    const IMemoryUsageTrackerPtr& tracker,
    TSharedRef reference,
    bool keepExistingTracking)
{
    if (!tracker || !reference) {
        return reference;
    }
    return tracker->TryTrack(std::move(reference), keepExistingTracking);
}

TSharedRef TrackMemory(
    const IMemoryUsageTrackerPtr& tracker,
    TSharedRef reference,
    bool keepExistingTracking)
{
    if (!tracker || !reference) {
        return reference;
    }
    return tracker->Track(std::move(reference), keepExistingTracking);
}

TSharedRefArray TrackMemory(
    const IMemoryUsageTrackerPtr& tracker,
    TSharedRefArray array,
    bool keepExistingTracking)
{
    if (!tracker || !array) {
        return array;
    }
    TSharedRefArrayBuilder builder(array.Size());
    for (const auto& part : array) {
        builder.Add(tracker->Track(part, keepExistingTracking));
    }
    return builder.Finish();
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

