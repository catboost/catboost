#pragma once

#include "ref.h"

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

//! Poisons an uninitialized slice of memory.
/*
 * In release builds, does nothing.
 * In checked builds, clobbers memory with a garbage pattern.
 * In ASAN builds, does nothing.
 * In MSAN builds, invokes sanitizer poisoning to catch uninit-read.
 */
void PoisonUninitializedMemory(TMutableRef ref);

//! Poisons a freed slice of memory.
/*
 * In release builds, does nothing.
 * In checked builds, clobbers memory with a garbage pattern.
 * In ASAN and MSAN builds, invokes sanitizer poisoning to catch use-after-free.
 */
void PoisonFreedMemory(TMutableRef ref);

//! Indicates that a slice of memory that was previously given to #PoisonFreedMemory
//! has been recycled and can be reused.
/*!
 * In release builds, does nothing.
 * In checked builds, clobbers memory with a garbage pattern.
 * In ASAN builds, invokes sanitizer unpoisoning.
 * In MSAN builds, does nothing (the memory remains poisoned to catch uninit-read).
 */
void RecycleFreedMemory(TMutableRef ref);

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define POISON_INL_H_
#include "poison-inl.h"
#undef POISON_INL_H_
