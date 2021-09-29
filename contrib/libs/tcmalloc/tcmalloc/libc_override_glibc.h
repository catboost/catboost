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
// Used to override malloc routines on systems that are using glibc.

#ifndef TCMALLOC_LIBC_OVERRIDE_GLIBC_INL_H_
#define TCMALLOC_LIBC_OVERRIDE_GLIBC_INL_H_

#include <features.h>
#include <stddef.h>

#include "tcmalloc/tcmalloc.h"

#ifndef __GLIBC__
#error libc_override_glibc.h is for glibc distributions only.
#endif

// In glibc, the memory-allocation methods are weak symbols, so we can
// just override them with our own.  If we're using gcc, we can use
// __attribute__((alias)) to do the overriding easily (exception:
// Mach-O, which doesn't support aliases).  Otherwise we have to use a
// function call.
#if !defined(__GNUC__) || defined(__MACH__)

#include "libc_override_redefine.h"

#else  // #if !defined(__GNUC__) || defined(__MACH__)

// If we get here, we're a gcc system, so do all the overriding we do
// with gcc.  This does the overriding of all the 'normal' memory
// allocation.
#include "libc_override_gcc_and_weak.h"

// We also have to do some glibc-specific overriding.  Some library
// routines on RedHat 9 allocate memory using malloc() and free it
// using __libc_free() (or vice-versa).  Since we provide our own
// implementations of malloc/free, we need to make sure that the
// __libc_XXX variants (defined as part of glibc) also point to the
// same implementations.  Since it only matters for redhat, we
// do it inside the gcc #ifdef, since redhat uses gcc.
// TODO(b/134690953): only do this if we detect we're an old enough glibc?

extern "C" {
void* __libc_malloc(size_t size) noexcept
    TCMALLOC_ALIAS(TCMallocInternalMalloc);
void __libc_free(void* ptr) noexcept TCMALLOC_ALIAS(TCMallocInternalFree);
void* __libc_realloc(void* ptr, size_t size) noexcept
    TCMALLOC_ALIAS(TCMallocInternalRealloc);
void* __libc_calloc(size_t n, size_t size) noexcept
    TCMALLOC_ALIAS(TCMallocInternalCalloc);
void __libc_cfree(void* ptr) noexcept TCMALLOC_ALIAS(TCMallocInternalCfree);
void* __libc_memalign(size_t align, size_t s) noexcept
    TCMALLOC_ALIAS(TCMallocInternalMemalign);
void* __libc_valloc(size_t size) noexcept
    TCMALLOC_ALIAS(TCMallocInternalValloc);
void* __libc_pvalloc(size_t size) noexcept
    TCMALLOC_ALIAS(TCMallocInternalPvalloc);
int __posix_memalign(void** r, size_t a, size_t s) noexcept
    TCMALLOC_ALIAS(TCMallocInternalPosixMemalign);
}  // extern "C"

#endif  // #if defined(__GNUC__) && !defined(__MACH__)

// We also have to hook libc malloc.  While our work with weak symbols
// should make sure libc malloc is never called in most situations, it
// can be worked around by shared libraries with the DEEPBIND
// environment variable set.  The below hooks libc to call our malloc
// routines even in that situation.  In other situations, this hook
// should never be called.
extern "C" {
static void* glibc_override_malloc(size_t size, const void* caller) {
  return TCMallocInternalMalloc(size);
}
static void* glibc_override_realloc(void* ptr, size_t size,
                                    const void* caller) {
  return TCMallocInternalRealloc(ptr, size);
}
static void glibc_override_free(void* ptr, const void* caller) {
  TCMallocInternalFree(ptr);
}
static void* glibc_override_memalign(size_t align, size_t size,
                                     const void* caller) {
  return TCMallocInternalMemalign(align, size);
}

// We should be using __malloc_initialize_hook here.  (See
// http://swoolley.org/man.cgi/3/malloc_hook.)  However, this causes weird
// linker errors with programs that link with -static, so instead we just assign
// the vars directly at static-constructor time.  That should serve the same
// effect of making sure the hooks are set before the first malloc call the
// program makes.

// Glibc-2.14 and above make __malloc_hook and friends volatile
#ifndef __MALLOC_HOOK_VOLATILE
#define __MALLOC_HOOK_VOLATILE /**/
#endif

void* (*__MALLOC_HOOK_VOLATILE __malloc_hook)(size_t, const void*) =
    &glibc_override_malloc;
void* (*__MALLOC_HOOK_VOLATILE __realloc_hook)(void*, size_t, const void*) =
    &glibc_override_realloc;
void (*__MALLOC_HOOK_VOLATILE __free_hook)(void*,
                                           const void*) = &glibc_override_free;
void* (*__MALLOC_HOOK_VOLATILE __memalign_hook)(size_t, size_t, const void*) =
    &glibc_override_memalign;

}  // extern "C"

#endif  // TCMALLOC_LIBC_OVERRIDE_GLIBC_INL_H_
