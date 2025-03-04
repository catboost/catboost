#pragma once
#ifndef POISON_INL_H_
#error "Direct inclusion of this file is not allowed, include poison.h"
// For the sake of sane code completion.
#include "poison.h"
#endif

#include <util/system/compiler.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

#if defined(_asan_enabled_)

extern "C" {
void __asan_poison_memory_region(void const volatile *addr, size_t size);
void __asan_unpoison_memory_region(void const volatile *addr, size_t size);
} // extern "C"

Y_FORCE_INLINE void PoisonMemory(TMutableRef ref)
{
    __asan_poison_memory_region(ref.data(), ref.size());
}

Y_FORCE_INLINE void UnpoisonMemory(TMutableRef ref)
{
    __asan_unpoison_memory_region(ref.data(), ref.size());
}

#elif defined(_msan_enabled_)

extern "C" {
void __msan_unpoison(const volatile void* a, size_t size);
void __msan_poison(const volatile void* a, size_t size);
} // extern "C"

Y_FORCE_INLINE void PoisonMemory(TMutableRef ref)
{
    __msan_poison(ref.data(), ref.size());
}

Y_FORCE_INLINE void UnpoisonMemory(TMutableRef ref)
{
    __msan_unpoison(ref.data(), ref.size());
}

#elif defined(NDEBUG)

Y_FORCE_INLINE void PoisonMemory(TMutableRef /*ref*/)
{ }

Y_FORCE_INLINE void UnpoisonMemory(TMutableRef /*ref*/)
{ }

#endif

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
