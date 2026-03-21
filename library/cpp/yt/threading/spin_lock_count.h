#pragma once

#include "public.h"

namespace NYT::NThreading {

////////////////////////////////////////////////////////////////////////////////
/*
 * Counts the number of spinlocks currently held by the thread.
 * Generally useful to ensure that we are not holding a spinlock in a given context.
 * Tracking is only active in debug builds.
*
 * In order to properly support tracking in your spinlock you have to do two steps:
 * 1) Insert |(Maybe)RecordSpinlockAcquired| and |RecordSpinlockReleased| inside your
 * |(Try)Acquire| and |Release| calls.
 * 2) (Optional) Define |static constexpr Tracked = true| for your spinlock
 * so that you can use algorithms aware of spinlock tracking.
 *
 * Semantic requirements:
 * 1) T must have a method successful call to which begins critical section (e.g. Acquire).
 * 2) T must have a method successful call to which ends critical section (e.g. Release).
 * In (1) and (2) "successful" is expected to have a definition given by T's author.
 * 3) Beginning of a critical section CS must be sequenced before |RecordSpinlockAcquired| call.
 * 4) |RecordSpinlockAcquired| call must be sequenced before |RecordSpinlockReleased| call.
 * 5) |RecordSpinlockReleased| must be sequenced before the ending of the CS.
 */

template <class T>
concept CTracedSpinLock = T::Traced;

int GetActiveSpinLockCount();
void VerifyNoSpinLockAffinity();

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NThreading

#define SPIN_LOCK_COUNT_INL_H_
#include "spin_lock_count-inl.h"
#undef SPIN_LOCK_COUNT_INL_H_
