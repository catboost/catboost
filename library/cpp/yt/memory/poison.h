#pragma once

#include "ref.h"

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

//! In release builds, does nothing.
//! In checked builds, clobbers memory with garbage pattern.
//! In sanitized builds, invokes sanitizer poisoning.
void PoisonMemory(TMutableRef ref);

//! In release builds, does nothing.
//! In checked builds, clobbers memory with (another) garbage pattern.
//! In sanitized builds, invokes sanitizer unpoisoning.
void UnpoisonMemory(TMutableRef ref);

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define POISON_INL_H_
#include "poison-inl.h"
#undef POISON_INL_H_
