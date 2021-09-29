// Copyright 2019 The TCMalloc Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Used to override malloc routines on systems that define the
// memory allocation routines to be weak symbols in their libc
// (almost all unix-based systems are like this), on gcc, which
// suppports the 'alias' attribute.

#ifndef TCMALLOC_LIBC_OVERRIDE_GCC_AND_WEAK_INL_H_
#define TCMALLOC_LIBC_OVERRIDE_GCC_AND_WEAK_INL_H_

#include <stddef.h>

#include <new>

#include "tcmalloc/tcmalloc.h"

#ifndef __GNUC__
#error libc_override_gcc_and_weak.h is for gcc distributions only.
#endif

// visibility("default") ensures that these symbols are always exported, even
// with -fvisibility=hidden.
#define TCMALLOC_ALIAS(tc_fn) \
  __attribute__((alias(#tc_fn), visibility("default")))

void* operator new(size_t size) noexcept(false)
    TCMALLOC_ALIAS(TCMallocInternalNew);
void operator delete(void* p) noexcept TCMALLOC_ALIAS(TCMallocInternalDelete);
void operator delete(void* p, size_t size) noexcept
    TCMALLOC_ALIAS(TCMallocInternalDeleteSized);
void* operator new[](size_t size) noexcept(false)
    TCMALLOC_ALIAS(TCMallocInternalNewArray);
void operator delete[](void* p) noexcept
    TCMALLOC_ALIAS(TCMallocInternalDeleteArray);
void operator delete[](void* p, size_t size) noexcept
    TCMALLOC_ALIAS(TCMallocInternalDeleteArraySized);
void* operator new(size_t size, const std::nothrow_t& nt) noexcept
    TCMALLOC_ALIAS(TCMallocInternalNewNothrow);
void* operator new[](size_t size, const std::nothrow_t& nt) noexcept
    TCMALLOC_ALIAS(TCMallocInternalNewArrayNothrow);
void operator delete(void* p, const std::nothrow_t& nt) noexcept
    TCMALLOC_ALIAS(TCMallocInternalDeleteNothrow);
void operator delete[](void* p, const std::nothrow_t& nt) noexcept
    TCMALLOC_ALIAS(TCMallocInternalDeleteArrayNothrow);

void* operator new(size_t size, std::align_val_t alignment) noexcept(false)
    TCMALLOC_ALIAS(TCMallocInternalNewAligned);
void* operator new(size_t size, std::align_val_t alignment,
                   const std::nothrow_t&) noexcept
    TCMALLOC_ALIAS(TCMallocInternalNewAligned_nothrow);
void operator delete(void* p, std::align_val_t alignment) noexcept
    TCMALLOC_ALIAS(TCMallocInternalDeleteAligned);
void operator delete(void* p, std::align_val_t alignment,
                     const std::nothrow_t&) noexcept
    TCMALLOC_ALIAS(TCMallocInternalDeleteAligned_nothrow);
void operator delete(void* p, size_t size, std::align_val_t alignment) noexcept
    TCMALLOC_ALIAS(TCMallocInternalDeleteSizedAligned);
void* operator new[](size_t size, std::align_val_t alignment) noexcept(false)
    TCMALLOC_ALIAS(TCMallocInternalNewArrayAligned);
void* operator new[](size_t size, std::align_val_t alignment,
                     const std::nothrow_t&) noexcept
    TCMALLOC_ALIAS(TCMallocInternalNewArrayAligned_nothrow);
void operator delete[](void* p, std::align_val_t alignment) noexcept
    TCMALLOC_ALIAS(TCMallocInternalDeleteArrayAligned);
void operator delete[](void* p, std::align_val_t alignment,
                       const std::nothrow_t&) noexcept
    TCMALLOC_ALIAS(TCMallocInternalDeleteArrayAligned_nothrow);
void operator delete[](void* p, size_t size,
                       std::align_val_t alignemnt) noexcept
    TCMALLOC_ALIAS(TCMallocInternalDeleteArraySizedAligned);

extern "C" {
void* malloc(size_t size) noexcept TCMALLOC_ALIAS(TCMallocInternalMalloc);
void free(void* ptr) noexcept TCMALLOC_ALIAS(TCMallocInternalFree);
void sdallocx(void* ptr, size_t size, int flags) noexcept
    TCMALLOC_ALIAS(TCMallocInternalSdallocx);
void* realloc(void* ptr, size_t size) noexcept
    TCMALLOC_ALIAS(TCMallocInternalRealloc);
void* calloc(size_t n, size_t size) noexcept
    TCMALLOC_ALIAS(TCMallocInternalCalloc);
void cfree(void* ptr) noexcept TCMALLOC_ALIAS(TCMallocInternalCfree);
void* memalign(size_t align, size_t s) noexcept
    TCMALLOC_ALIAS(TCMallocInternalMemalign);
void* aligned_alloc(size_t align, size_t s) noexcept
    TCMALLOC_ALIAS(TCMallocInternalAlignedAlloc);
void* valloc(size_t size) noexcept TCMALLOC_ALIAS(TCMallocInternalValloc);
void* pvalloc(size_t size) noexcept TCMALLOC_ALIAS(TCMallocInternalPvalloc);
int posix_memalign(void** r, size_t a, size_t s) noexcept
    TCMALLOC_ALIAS(TCMallocInternalPosixMemalign);
void malloc_stats(void) noexcept TCMALLOC_ALIAS(TCMallocInternalMallocStats);
int mallopt(int cmd, int value) noexcept
    TCMALLOC_ALIAS(TCMallocInternalMallOpt);
#ifdef TCMALLOC_HAVE_STRUCT_MALLINFO
struct mallinfo mallinfo(void) noexcept
    TCMALLOC_ALIAS(TCMallocInternalMallocInfo);
#endif
size_t malloc_size(void* p) noexcept TCMALLOC_ALIAS(TCMallocInternalMallocSize);
size_t malloc_usable_size(void* p) noexcept
    TCMALLOC_ALIAS(TCMallocInternalMallocSize);
}  // extern "C"

#endif  // TCMALLOC_LIBC_OVERRIDE_GCC_AND_WEAK_INL_H_
