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
// Used on systems that don't have their own definition of
// malloc/new/etc.  (Typically this will be a windows msvcrt.dll that
// has been edited to remove the definitions.)  We can just define our
// own as normal functions.
//
// This should also work on systems were all the malloc routines are
// defined as weak symbols, and there's no support for aliasing.

#ifndef TCMALLOC_LIBC_OVERRIDE_REDEFINE_H_
#define TCMALLOC_LIBC_OVERRIDE_REDEFINE_H_

#include <sys/cdefs.h>

#include <cstddef>
#include <new>

#include "tcmalloc/tcmalloc.h"

void* operator new(size_t size) { return TCMallocInternalNew(size); }
void operator delete(void* p) noexcept { TCMallocInternalDelete(p); }
void* operator new[](size_t size) { return TCMallocInternalNewArray(size); }
void operator delete[](void* p) noexcept { TCMallocInternalDeleteArray(p); }
void* operator new(size_t size, const std::nothrow_t& nt) noexcept {
  return TCMallocInternalNewNothrow(size, nt);
}
void* operator new[](size_t size, const std::nothrow_t& nt) noexcept {
  return TCMallocInternalNewArrayNothrow(size, nt);
}
void operator delete(void* ptr, const std::nothrow_t& nt) noexcept {
  return TCMallocInternalDeleteNothrow(ptr, nt);
}
void operator delete[](void* ptr, const std::nothrow_t& nt) noexcept {
  return TCMallocInternalDeleteArrayNothrow(ptr, nt);
}
extern "C" {
void* malloc(size_t s) noexcept { return TCMallocInternalMalloc(s); }
void free(void* p) noexcept { TCMallocInternalFree(p); }
void sdallocx(void* p, size_t s, int flags) {
  TCMallocInternalSdallocx(p, s, flags);
}
void* realloc(void* p, size_t s) noexcept {
  return TCMallocInternalRealloc(p, s);
}
void* calloc(size_t n, size_t s) noexcept {
  return TCMallocInternalCalloc(n, s);
}
void cfree(void* p) noexcept { TCMallocInternalCfree(p); }
void* memalign(size_t a, size_t s) noexcept {
  return TCMallocInternalMemalign(a, s);
}
void* valloc(size_t s) noexcept { return TCMallocInternalValloc(s); }
void* pvalloc(size_t s) noexcept { return TCMallocInternalPvalloc(s); }
int posix_memalign(void** r, size_t a, size_t s) noexcept {
  return TCMallocInternalPosixMemalign(r, a, s);
}
void malloc_stats(void) noexcept { TCMallocInternalMallocStats(); }
int mallopt(int cmd, int v) noexcept { return TCMallocInternalMallOpt(cmd, v); }
#ifdef HAVE_STRUCT_MALLINFO
struct mallinfo mallinfo(void) noexcept {
  return TCMallocInternalMallocInfo();
}
#endif
size_t malloc_size(void* p) noexcept { return TCMallocInternalMallocSize(p); }
size_t malloc_usable_size(void* p) noexcept {
  return TCMallocInternalMallocSize(p);
}
}  // extern "C"

#endif  // TCMALLOC_LIBC_OVERRIDE_REDEFINE_H_
