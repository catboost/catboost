#pragma once

#define SPIN_LOCK_COUNT_INL_H_
#include "spin_lock_count-inl.h"
#undef SPIN_LOCK_COUNT_INL_H_

namespace NYT::NThreading {

// Tracks the number of spinlocks currently held by the thread.
// Generally useful to ensure that we are not holding a spinlock
// in a given context.
// Tracking is only active in debug builds.

////////////////////////////////////////////////////////////////////////////////

namespace NDetail {

// In order to properly support tracking in your spinlock you have to do two steps:
// 1) Insert RecordSpinlockAcquired and RecordSpinlockReleased inside your
// (Try)Acquire and Release calls.
// 2) (Optional) Write REGISTER_TRACKED_SPIN_LOCK_CLASS(TSpinLock) for your spinlock
// so that you can use algorithms aware of spinlock tracking.

using NPrivate::RecordSpinLockAcquired;
using NPrivate::RecordSpinLockReleased;

template <class TSpinLock>
struct TIsTrackedSpinLock
    : public std::false_type
{ };

#define REGISTER_TRACKED_SPIN_LOCK_CLASS(Name) \
    namespace NDetail { \
    \
    template <> \
    struct TIsTrackedSpinLock<Name> \
        : public std::true_type \
    { }; \
    \
    } \

} // namespace NDetail

////////////////////////////////////////////////////////////////////////////////

template <class T>
concept CTrackedSpinLock = NDetail::TIsTrackedSpinLock<T>::value;
//! Semantic requirements:
//! 1) T must have a method successful call to which begins critical section (e.g. Acquire).
//! 2) T must have a method successful call to which ends critical section (e.g. Release).
//! In (1) and (2) "successful" is expected to have a definition given by T's author.
//! 3) Beggining of a critical section CS must be sequenced before RecordSpinlockAcquired(true) call.
//! 4) RecordSpinlockAcquired(true) call must be sequenced before RecordSpinlockReleased() call.
//! 5) RecordSpinlockReleased() must be sequenced before the ending of the CS.

////////////////////////////////////////////////////////////////////////////////

using NPrivate::VerifyNoSpinLockAffinity;

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NThreading
