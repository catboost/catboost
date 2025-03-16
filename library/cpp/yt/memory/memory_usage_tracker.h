#pragma once

#include "blob.h"
#include "ref.h"

#include <library/cpp/yt/error/error.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

struct IMemoryUsageTracker
    : public TRefCounted
{
    virtual TError TryAcquire(i64 size) = 0;
    virtual TError TryChange(i64 size) = 0;
    //! Returns true unless overcommit occurred.
    virtual bool Acquire(i64 size) = 0;
    virtual void Release(i64 size) = 0;
    virtual void SetLimit(i64 size) = 0;
    virtual i64 GetLimit() const = 0;
    virtual i64 GetUsed() const = 0;
    virtual i64 GetFree() const = 0;
    virtual bool IsExceeded() const = 0;

    //! Tracks memory used by references.
    /*!
    * Memory tracking is implemented by specific shared ref holders.
    * #Track returns reference with a holder that wraps the old one and also
    * enables accounting memory in memory tracker's internal state.

    * Subsequent #Track calls for this reference drop memory reference tracker's
    * holder unless #keepExistingTracking is true.
    */
    virtual TSharedRef Track(
        TSharedRef reference,
        bool keepHolder = false) = 0;
    virtual TErrorOr<TSharedRef> TryTrack(
        TSharedRef reference,
        bool keepHolder) = 0;
};

DEFINE_REFCOUNTED_TYPE(IMemoryUsageTracker)

////////////////////////////////////////////////////////////////////////////////

struct IReservingMemoryUsageTracker
    : public IMemoryUsageTracker
{
    virtual void ReleaseUnusedReservation() = 0;
    virtual TError TryReserve(i64 size) = 0;
};

DEFINE_REFCOUNTED_TYPE(IReservingMemoryUsageTracker)

////////////////////////////////////////////////////////////////////////////////

IMemoryUsageTrackerPtr GetNullMemoryUsageTracker();

////////////////////////////////////////////////////////////////////////////////

class TMemoryUsageTrackerGuard
    : private TNonCopyable
{
public:
    TMemoryUsageTrackerGuard() = default;
    TMemoryUsageTrackerGuard(const TMemoryUsageTrackerGuard& other) = delete;
    TMemoryUsageTrackerGuard(TMemoryUsageTrackerGuard&& other);
    ~TMemoryUsageTrackerGuard();

    TMemoryUsageTrackerGuard& operator=(const TMemoryUsageTrackerGuard& other) = delete;
    TMemoryUsageTrackerGuard& operator=(TMemoryUsageTrackerGuard&& other);

    static TMemoryUsageTrackerGuard Build(
        IMemoryUsageTrackerPtr tracker,
        i64 granularity = 1);
    static TMemoryUsageTrackerGuard Acquire(
        IMemoryUsageTrackerPtr tracker,
        i64 size,
        i64 granularity = 1);
    static TErrorOr<TMemoryUsageTrackerGuard> TryAcquire(
        IMemoryUsageTrackerPtr tracker,
        i64 size,
        i64 granularity = 1);

    void Release();

    //! Releases the guard but does not return memory to the tracker.
    //! The caller should care about releasing memory itself.
    void ReleaseNoReclaim();

    explicit operator bool() const;

    i64 GetSize() const;
    void SetSize(i64 size);
    TError TrySetSize(i64 size);
    void IncreaseSize(i64 sizeDelta);
    void DecreaseSize(i64 sizeDelta);
    TMemoryUsageTrackerGuard TransferMemory(i64 size);

private:
    IMemoryUsageTrackerPtr Tracker_;
    i64 Size_ = 0;
    i64 AcquiredSize_ = 0;
    i64 Granularity_ = 0;

    void MoveFrom(TMemoryUsageTrackerGuard&& other);
    TError SetSizeImpl(i64 size, auto acquirer);
};

////////////////////////////////////////////////////////////////////////////////

class TMemoryTrackedBlob
{
public:
    static TMemoryTrackedBlob Build(
        IMemoryUsageTrackerPtr tracker,
        TRefCountedTypeCookie tagCookie = GetRefCountedTypeCookie<TDefaultBlobTag>());

    TMemoryTrackedBlob() = default;
    TMemoryTrackedBlob(const TMemoryTrackedBlob& other) = delete;
    TMemoryTrackedBlob(TMemoryTrackedBlob&& other) = default;
    ~TMemoryTrackedBlob() = default;

    TMemoryTrackedBlob& operator=(const TMemoryTrackedBlob& other) = delete;
    TMemoryTrackedBlob& operator=(TMemoryTrackedBlob&& other) = default;

    void Resize(
        i64 size,
        bool initializeStorage = true);

    TError TryResize(
        i64 size,
        bool initializeStorage = true);

    void Reserve(i64 capacity);

    TError TryReserve(i64 capacity);

    char* Begin();

    char* End();

    size_t Capacity() const;

    size_t Size() const;

    void Append(TRef ref);

    void Clear();

private:
    TBlob Blob_;
    TMemoryUsageTrackerGuard Guard_;

    TMemoryTrackedBlob(
        TBlob&& blob,
        TMemoryUsageTrackerGuard&& guard);
};

////////////////////////////////////////////////////////////////////////////////

TErrorOr<TSharedRef> TryTrackMemory(
    const IMemoryUsageTrackerPtr& tracker,
    TSharedRef reference,
    bool keepExistingTracking = false);

TSharedRef TrackMemory(
    const IMemoryUsageTrackerPtr& tracker,
    TSharedRef reference,
    bool keepExistingTracking = false);

TSharedRefArray TrackMemory(
    const IMemoryUsageTrackerPtr& tracker,
    TSharedRefArray array,
    bool keepExistingTracking = false);

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
